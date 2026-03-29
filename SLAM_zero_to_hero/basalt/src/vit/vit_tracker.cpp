/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2022-2024, Collabora Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Implementation for the VIT interface
@author Mateo de Mayo <mateo.demayo@collabora.com>
@author Simon Zeni <simon.zeni@collabora.com>
*/

#include <basalt/io/marg_data_io.h>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/vit/vit_tracker.hpp>
#include <vit_implementation_helper.hpp>
#include "vit_tracker_ui.hpp"

#include <tbb/concurrent_queue.h>
#include <CLI/CLI.hpp>
#include <opencv2/core/mat.hpp>
#include <sophus/se3.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <thread>

static std::array timing_titles = {
    "frame_ts",
    "tracker_received",
    "tracker_pushed",
    "frames_received",  // < Basalt computation starts
    "opticalflow_produced",
    "vio_start",
    "imu_preintegrated",
    "landmarks_updated",
    "optimized",
    "marginalized",
    "pose_produced",  // Basalt computation ends
    "tracker_consumer_received",
    "tracker_consumer_pushed",
    "get_pose",
};

namespace basalt::vit_implementation {

using std::cout;
using std::make_shared;
using std::make_unique;
using std::string;
using std::thread;
using std::vector;

struct Tracker::Implementation {
  static constexpr vit_tracker_extension_set_t exts{
      .has_add_camera_calibration = true,
      .has_add_imu_calibration = true,
      .has_pose_timing = true,
      .has_pose_features = true,
  };
  vit_tracker_extension_set_t enabled_exts{
      .has_add_camera_calibration = true,
      .has_add_imu_calibration = true,
      .has_pose_timing = false,
      .has_pose_features = false,
  };

  // Options parsed from unified config file
  bool show_gui = false;
  string cam_calib_path;
  string config_path;
  string marg_data_path;
  bool print_queue = false;
  bool use_double = false;
  bool deterministic = false;

  // VIO members
  struct {
    bool imu = false;
    vector<bool> cam;
  } calib_data_ready;
  uint32_t cam_count;

  Calibration<double> calib;
  VioConfig vio_config;
  OpticalFlowBase::Ptr opt_flow_ptr;

  VioEstimatorBase::Ptr vio;
  uint32_t expecting_frame = 0;

  // Queues
  std::atomic<bool> running = false;
  tbb::concurrent_bounded_queue<PoseVelBiasState<double>::Ptr> out_state_queue;
  tbb::concurrent_bounded_queue<PoseVelBiasState<double>::Ptr> monado_out_state_queue;
  tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr> *image_data_queue = nullptr;  // Invariant: not null after ctor
  tbb::concurrent_bounded_queue<ImuData<double>::Ptr> *imu_data_queue = nullptr;     // Invariant: not null after ctor

  // Threads
  thread state_consumer_thread;
  thread queues_printer_thread;

  // External Queues
  vit_tracker_ui ui{};
  MargDataSaver::Ptr marg_data_saver;
  // Additional calibration data
  vector<vit::CameraCalibration> added_cam_calibs{};
  vector<vit::ImuCalibration> added_imu_calibs{};

  OpticalFlowInput::Ptr partial_frame;

  explicit Implementation(const vit::Config *config) : show_gui(config->show_ui), cam_count(config->cam_count) {
    cout << "Basalt with cam_count=" << cam_count << ", show_gui=" << show_gui << "\n";

    // Basalt in its current state does not support monocular cameras, although it
    // should be possible to adapt it to do so, see:
    // https://gitlab.com/VladyslavUsenko/basalt/-/issues/2#note_201965760
    // https://gitlab.com/VladyslavUsenko/basalt/-/issues/25#note_362741510
    // https://github.com/DLR-RM/granite
    ASSERT(cam_count > 1, "Basalt doesn't support running with %d cameras", cam_count);
    calib_data_ready.cam.resize(cam_count, false);

    if (config->file == nullptr) {
      // For the pipeline to work now, the user will need to use add_cam/imu_calibration
      return;
    }

    load_unified_config(config->file);

    vio_config.load(config_path);
    load_calibration_data(cam_calib_path);

    monado_out_state_queue.set_capacity(32);
  }

  static vit::Result get_supported_extensions(vit::TrackerExtensionSet *out_exts) {
    *out_exts = exts;
    return vit::Result::VIT_SUCCESS;
  }

  vit::Result get_enabled_extensions(vit::TrackerExtensionSet *out_exts) const {
    *out_exts = enabled_exts;
    return vit::Result::VIT_SUCCESS;
  }

  vit::Result enable_extension(vit::TrackerExtension ext, bool enable) {
    auto ext_index = (int64_t)ext;
    if (ext_index >= VIT_TRACKER_EXTENSION_COUNT) {
      std::cout << "Invalid extension: " << ext << std::endl;
      return vit::Result::VIT_ERROR_INVALID_VALUE;
    }

    // TODO: Update VIT to remove NOLINTs
    bool supported = ((bool *)&exts)[ext_index];  // NOLINT: cppcoreguidelines-pro-type-cstyle-cast
    if (!supported) {
      std::cout << "Unsupported extension: " << ext << std::endl;
      return vit::Result::VIT_ERROR_NOT_SUPPORTED;
    }

    bool *enabled = &((bool *)&enabled_exts)[ext_index];  // NOLINT: cppcoreguidelines-pro-type-cstyle-cast
    *enabled = enable;

    return vit::Result::VIT_SUCCESS;
  }

  void load_unified_config(const string &unified_config) {
    CLI::App app{"Options for the Basalt SLAM Tracker"};

    app.add_option("--show-gui", show_gui, "Show GUI");
    app.add_option("--cam-calib", cam_calib_path, "Ground-truth camera calibration used for simulation.")->required();
    app.add_option("--config-path", config_path, "Path to config file.")->required();
    app.add_option("--marg-data", marg_data_path, "Path to folder where marginalization data will be stored.");
    app.add_option("--print-queue", print_queue, "Poll and print for queue sizes.");
    app.add_option("--use-double", use_double, "Whether to use a double or single precision pipeline.");
    app.add_option("--deterministic", deterministic, "Make the pipeline output reproducible (some performance impact)");

    try {
      // While --config-path sets the VIO configuration, --config sets the
      // entire unified Basalt configuration, including --config-path
      app.set_config("--config", unified_config, "Configuration file.", true);
      app.allow_config_extras(false);  // Makes parsing fail on unknown options
      string unique_config = "--config=" + unified_config;
      app.parse(unique_config);
    } catch (const CLI::ParseError &e) {
      app.exit(e);
      ASSERT(false, "Config file error (%s)", unified_config.c_str());
    }

    cout << "Instantiating Basalt SLAM tracker\n";
    cout << "Using config file: " << app["--config"]->as<string>() << "\n";
    cout << app.config_to_str(true, true) << "\n";
  }

  void load_calibration_data(const string &calib_path) {
    std::ifstream os(calib_path, std::ios::binary);
    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
      uint32_t calib_cam_count = calib.intrinsics.size();
      cout << "Loaded camera with " << calib_cam_count << " cameras\n";
      ASSERT_(calib_cam_count == cam_count);
      calib_data_ready.imu = true;
      for (uint32_t i = 0; i < calib_cam_count; i++) {
        calib_data_ready.cam.at(i) = true;
      }
    } else {
      std::cerr << "could not load camera calibration " << calib_path << "\n";
      std::abort();
    }
  }

  void apply_cam_calibration(const vit::CameraCalibration &cam_calib) {
    using Scalar = double;
    size_t i = cam_calib.camera_index;

    const cv::Matx<double, 4, 4> tic(cam_calib.transform);
    Eigen::Matrix3d ric;
    ric << tic(0, 0), tic(0, 1), tic(0, 2), tic(1, 0), tic(1, 1), tic(1, 2), tic(2, 0), tic(2, 1), tic(2, 2);
    Eigen::Quaterniond q(ric);
    Eigen::Vector3d p{tic(0, 3), tic(1, 3), tic(2, 3)};
    ASSERT_(calib.T_i_c.size() == i);
    calib.T_i_c.emplace_back(q, p);

    GenericCamera<double> model;
    const double *d = cam_calib.distortion;
    if (cam_calib.model == vit::CameraDistortion::VIT_CAMERA_DISTORTION_NONE) {
      ASSERT_(cam_calib.distortion_count == 0);
      PinholeCamera<Scalar>::VecN mp;
      mp << cam_calib.fx, cam_calib.fy, cam_calib.cx, cam_calib.cy;
      PinholeCamera pinhole(mp);
      model.variant = pinhole;
    } else if (cam_calib.model == vit::CameraDistortion::VIT_CAMERA_DISTORTION_KB4) {
      ASSERT_(cam_calib.distortion_count == 4);
      KannalaBrandtCamera4<Scalar>::VecN mp;
      mp << cam_calib.fx, cam_calib.fy, cam_calib.cx, cam_calib.cy, d[0], d[1], d[2], d[3];
      KannalaBrandtCamera4 kannala_brandt(mp);
      model.variant = kannala_brandt;
    } else if (cam_calib.model == vit::CameraDistortion::VIT_CAMERA_DISTORTION_RT8) {
      ASSERT_(cam_calib.distortion_count == 9);  // 8 and rpmax
      PinholeRadtan8Camera<Scalar>::VecN mp;
      mp << cam_calib.fx, cam_calib.fy, cam_calib.cx, cam_calib.cy, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7];
      Scalar rpmax = d[8];
      PinholeRadtan8Camera pinhole_radtan8(mp, rpmax);
      model.variant = pinhole_radtan8;
    } else {
      ASSERT(false, "Unsupported camera model (%d)", cam_calib.model);
    }
    ASSERT_(calib.intrinsics.size() == i);
    calib.intrinsics.push_back(model);

    ASSERT_(calib.resolution.size() == i);
    calib.resolution.emplace_back(cam_calib.width, cam_calib.height);

    calib_data_ready.cam.at(i) = true;
  }

  void apply_imu_calibration(const vit::ImuCalibration &imu_calib) {
    using Scalar = double;

    int i = imu_calib.imu_index;
    ASSERT(i == 0, "More than one IMU unsupported (%d)", i);

    static double frequency = -1;
    if (frequency == -1) {
      frequency = imu_calib.frequency;
      calib.imu_update_rate = frequency;
    } else {
      ASSERT(frequency == calib.imu_update_rate, "Unsupported mix of IMU frequencies %lf != %lf", frequency,
             calib.imu_update_rate);
    }

    // Accelerometer calibration

    const vit::InertialCalibration &accel = imu_calib.accel;

    Eigen::Matrix<Scalar, 9, 1> accel_bias_full;
    const cv::Matx<double, 3, 1> abias(accel.offset);
    const cv::Matx<double, 3, 3> atran(accel.transform);

    // TODO: Doing the same as rs_t265.cpp but that's incorrect. We should be doing an LQ decomposition of atran and
    // using L. See https://gitlab.com/VladyslavUsenko/basalt-headers/-/issues/8
    accel_bias_full << abias(0), abias(1), abias(2), atran(0, 0) - 1, atran(1, 0), atran(2, 0), atran(1, 1) - 1,
        atran(2, 1), atran(2, 2) - 1;
    CalibAccelBias<Scalar> accel_bias;
    accel_bias.getParam() = accel_bias_full;
    calib.calib_accel_bias = accel_bias;

    calib.accel_noise_std << accel.noise_std[0], accel.noise_std[1], accel.noise_std[2];
    calib.accel_bias_std << accel.bias_std[0], accel.bias_std[1], accel.bias_std[2];

    // Gyroscope calibration

    const vit::InertialCalibration &gyro = imu_calib.gyro;

    Eigen::Matrix<Scalar, 12, 1> gyro_bias_full;
    const cv::Matx<double, 3, 1> gbias(gyro.offset);
    const cv::Matx<double, 3, 3> gtran(gyro.transform);
    gyro_bias_full << gbias(0), gbias(1), gbias(2), gtran(0, 0) - 1, gtran(1, 0), gtran(2, 0), gtran(0, 1),
        gtran(1, 1) - 1, gtran(2, 1), gtran(0, 2), gtran(1, 2), gtran(2, 2) - 1;
    CalibGyroBias<Scalar> gyro_bias;
    gyro_bias.getParam() = gyro_bias_full;
    calib.calib_gyro_bias = gyro_bias;

    calib.gyro_noise_std << gyro.noise_std[0], gyro.noise_std[1], gyro.noise_std[2];
    calib.gyro_bias_std << gyro.bias_std[0], gyro.bias_std[1], gyro.bias_std[2];

    calib_data_ready.imu = true;
  }

  void print_calibration() {
    std::stringstream ss{};
    {
      cereal::JSONOutputArchive write_to_stream(ss);
      write_to_stream(calib);
    }
    cout << "Calibration: " << ss.str() << "\n";
  }

  void initialize() {
    // Overwrite camera calibration data
    for (const auto &c : added_cam_calibs) {
      apply_cam_calibration(c);
    }

    // Overwrite IMU calibration data
    for (const auto &c : added_imu_calibs) {
      apply_imu_calibration(c);
    }

    ASSERT(calib_data_ready.imu, "Missing IMU calibration");
    for (size_t i = 0; i < calib_data_ready.cam.size(); i++) {
      ASSERT(calib_data_ready.cam.at(i), "Missing cam%zu calibration", i);
    }

    // NOTE: This factory also starts the optical flow
    opt_flow_ptr = OpticalFlowFactory::getOpticalFlow(vio_config, calib);
    opt_flow_ptr->start();
    image_data_queue = &opt_flow_ptr->input_img_queue;
    ASSERT_(image_data_queue != nullptr);

    vio = VioEstimatorFactory::getVioEstimator(vio_config, calib, constants::g, true, use_double);
    imu_data_queue = &vio->imu_data_queue;
    ASSERT_(imu_data_queue != nullptr);

    opt_flow_ptr->output_queue = &vio->vision_data_queue;
    opt_flow_ptr->show_gui = show_gui;
    if (show_gui) {
      ui.initialize(cam_count);
      vio->out_vis_queue = &ui.out_vis_queue;
    };
    vio->out_state_queue = &out_state_queue;
    vio->opt_flow_depth_guess_queue = &opt_flow_ptr->input_depth_queue;
    vio->opt_flow_state_queue = &opt_flow_ptr->input_state_queue;
    vio->opt_flow_lm_bundle_queue = &opt_flow_ptr->input_lm_bundle_queue;

    if (!marg_data_path.empty()) {
      marg_data_saver = make_shared<MargDataSaver>(marg_data_path);
      vio->out_marg_queue = &marg_data_saver->in_marg_queue;
    }
  }

  void start() {
    running = true;
    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    if (show_gui) ui.start(vio->getT_w_i_init(), calib, vio_config, opt_flow_ptr, vio);
    if (!deterministic) state_consumer_thread = thread(&Tracker::Implementation::state_consumer, this);
    if (print_queue) queues_printer_thread = thread(&Tracker::Implementation::queues_printer, this);
  }

  void stop() {
    running = false;
    image_data_queue->push(nullptr);
    imu_data_queue->push(nullptr);
    opt_flow_ptr->input_imu_queue.push(nullptr);

    if (print_queue) queues_printer_thread.join();
    if (!deterministic) state_consumer_thread.join();
    if (show_gui) ui.stop();

    // TODO: There is a segfault when closing monado without starting the stream
    // happens in a lambda from keypoint_vio.cpp and ends at line c1alib_bias.hpp:112
  }

  void finalize() const {
    // Only the OpticalFlow gets started by initialize, finish it with this
    image_data_queue->push(nullptr);
  }

  void reset() { vio->scheduleResetState(); }

  void push_imu_sample(const vit::ImuSample *s) {
    // concurrent_bounded_queue expects Erasable and Allocator named
    // requirements for the type, using a pointer because it already is. This is
    // done in the others examples as well but it is far from optimal.
    auto data = make_shared<ImuData<double>>();
    data->t_ns = s->timestamp;
    data->accel = {s->ax, s->ay, s->az};
    data->gyro = {s->wx, s->wy, s->wz};
    imu_data_queue->push(data);
    opt_flow_ptr->input_imu_queue.push(data);
  }

  void push_frame(const vit::ImgSample *s) {
    uint32_t i = s->cam_index;
    ASSERT(expecting_frame == i, "Expected cam%d frame, received cam%d", expecting_frame, i);

    expecting_frame = (expecting_frame + 1) % cam_count;

    if (i == 0) {
      partial_frame = make_shared<OpticalFlowInput>(cam_count);
      partial_frame->show_uimat = ui.get_mat_to_show();

      partial_frame->t_ns = s->timestamp;

      // Initialize stats
      partial_frame->stats.enabled_exts = enabled_exts;
      if (enabled_exts.has_pose_timing) {
        partial_frame->stats.ts = s->timestamp;
        partial_frame->stats.timings.reserve(timing_titles.size());
        partial_frame->stats.timing_titles = timing_titles.data();
        partial_frame->addTime("frame_ts", s->timestamp);
        partial_frame->addTime("tracker_received");
      }

      if (enabled_exts.has_pose_features) {
        partial_frame->stats.features_per_cam.resize(cam_count);
      }
    } else {
      ASSERT(partial_frame->t_ns == s->timestamp, "cam0 and cam%d frame timestamps differ: %ld != %ld", i,
             partial_frame->t_ns, s->timestamp);
    }

    // Forced to use uint16_t here, in place because of cameras with 12-bit grayscale support
    auto &mimg = partial_frame->img_data[i].img;
    mimg = make_shared<ManagedImage<uint16_t>>(s->width, s->height);

    for (uint32_t j = 0; j < s->mask_count; j++) {
      auto &r = s->masks[j];
      partial_frame->masks[i].masks.emplace_back(r.x, r.y, r.w, r.h);
    }

    // TODO figure out format copy to avoid using a cv::Mat
    int type = s->format == VIT_IMAGE_FORMAT_L8 ? CV_8UC1 : CV_8UC3;
    cv::Mat img{(int)s->height, (int)s->width, type, s->data, s->stride};

    for (uint32_t j = 0; j < s->width * s->height; j++) {
      mimg->ptr[j] = img.at<uchar>(j) << 8;
    }

    if (i == cam_count - 1) {
      partial_frame->addTime("tracker_pushed");
      image_data_queue->push(partial_frame);
      if (show_gui) ui.update_last_image(partial_frame);
      if (deterministic) pop_state();
    }
  }

  vit::Result pop_pose(vit::Pose **pose) {
    PoseVelBiasState<double>::Ptr state;

    bool popped = monado_out_state_queue.try_pop(state);

    if (popped) {
      // Discard the pose if the caller doesn't request it.
      if (pose == nullptr) {
        return vit::Result::VIT_SUCCESS;
      }

      // TODO find another way to give the state to the pose implementation
      // without touching the interface
      Pose *p = new Pose();
      p->impl_ = make_unique<Pose::Implementation>(state);
      assert(p->impl_);
      state->input_images->addTime("get_pose");

      *pose = static_cast<vit_pose_t *>(p);
    } else {
      *pose = nullptr;
    }

    return vit::Result::VIT_SUCCESS;
  }

  static vit::Result get_timing_titles(vit_tracker_timing_titles *out_titles) {
    if (!exts.has_pose_timing) {
      return vit::Result::VIT_ERROR_NOT_SUPPORTED;
    }

    out_titles->count = (sizeof(timing_titles) / sizeof(timing_titles[0]));
    out_titles->titles = &timing_titles[0];

    return vit::Result::VIT_SUCCESS;
  }

 private:
  bool pop_state() {
    PoseVelBiasState<double>::Ptr data;
    PoseVelBiasState<double>::Ptr _;
    out_state_queue.pop(data);
    if (data.get() == nullptr) {
      while (!monado_out_state_queue.try_push(nullptr)) monado_out_state_queue.pop(_);
      return false;
    }
    data->input_images->addTime("tracker_consumer_received");

    if (show_gui) ui.log_vio_data(data);

    data->input_images->addTime("tracker_consumer_pushed");
    while (!monado_out_state_queue.try_push(data)) monado_out_state_queue.pop(_);

    return true;
  }

  void state_consumer() {
    while (pop_state()) continue;
  }

  void queues_printer() {
    while (running) {
      cout << "[in] frames: " << image_data_queue->size() << "/" << image_data_queue->capacity() << " \n"
           << "[in] imu: " << imu_data_queue->size() << "/" << imu_data_queue->capacity() << " \n"
           << "[in] depth: " << opt_flow_ptr->input_depth_queue.unsafe_size() << "/-- \n"
           << "[mid] keypoints: " << opt_flow_ptr->output_queue->size() << "/" << opt_flow_ptr->output_queue->capacity()
           << " \n"
           << "[mid] pose: " << out_state_queue.size() << "/" << out_state_queue.capacity() << "\n"
           << "[out] monado queue: " << monado_out_state_queue.size() << "/" << monado_out_state_queue.capacity()
           << "\n"
           << "[out] ui: " << vio->out_vis_queue->size() << "/" << vio->out_vis_queue->capacity() << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    cout << "Finished queues_printer\n";
  }
};

Tracker::Tracker(const vit::Config *config) { impl_ = make_unique<Tracker::Implementation>(config); }

vit::Result Tracker::has_image_format(vit::ImageFormat fmt, bool *out) const {
  switch (fmt) {
    case VIT_IMAGE_FORMAT_L8:
    case VIT_IMAGE_FORMAT_L16:
      *out = true;
      break;
    default:
      std::cerr << "Unknown image format: " << fmt << std::endl;
      break;
  }

  *out = false;
  return vit::Result::VIT_ERROR_INVALID_VALUE;
}

vit::Result Tracker::get_supported_extensions(vit::TrackerExtensionSet *out) const {
  return impl_->get_supported_extensions(out);
}

vit::Result Tracker::get_enabled_extensions(vit::TrackerExtensionSet *out) const {
  return impl_->get_enabled_extensions(out);
}

vit::Result Tracker::enable_extension(vit::TrackerExtension ext, bool enable) {
  return impl_->enable_extension(ext, enable);
}

vit::Result Tracker::start() {
  impl_->initialize();
  impl_->start();
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::stop() {
  impl_->finalize();
  impl_->stop();
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::reset() {
  impl_->reset();
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::is_running(bool *out_running) const {
  *out_running = impl_->running;
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::push_imu_sample(const vit::ImuSample *sample) {
  impl_->push_imu_sample(sample);
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::push_img_sample(const vit::ImgSample *sample) {
  impl_->push_frame(sample);
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::pop_pose(vit::Pose **pose) { return impl_->pop_pose(pose); }

vit::Result Tracker::get_timing_titles(vit_tracker_timing_titles *out_titles) const {
  return impl_->get_timing_titles(out_titles);
}

vit::Result Tracker::add_imu_calibration(const vit::ImuCalibration *calibration) {
  impl_->added_imu_calibs.push_back(*calibration);
  return vit::Result::VIT_SUCCESS;
}

vit::Result Tracker::add_camera_calibration(const vit::CameraCalibration *calibration) {
  impl_->added_cam_calibs.push_back(*calibration);
  return vit::Result::VIT_SUCCESS;
}

struct Pose::Implementation {
  PoseVelBiasState<double>::Ptr state;

  Implementation(const PoseVelBiasState<double>::Ptr &state) : state(state) {}

  ~Implementation() = default;

  void get_data(vit::PoseData *data) const {
    Sophus::SE3d T_w_i = state->T_w_i;

    data->timestamp = state->t_ns;
    data->px = T_w_i.translation().x();
    data->py = T_w_i.translation().y();
    data->pz = T_w_i.translation().z();
    data->ox = T_w_i.unit_quaternion().x();
    data->oy = T_w_i.unit_quaternion().y();
    data->oz = T_w_i.unit_quaternion().z();
    data->ow = T_w_i.unit_quaternion().w();
    data->vx = state->vel_w_i.x();
    data->vy = state->vel_w_i.y();
    data->vz = state->vel_w_i.z();
  }

  vit::Result get_timing(vit::PoseTiming *out_timing) const {
    const vit::TimeStats &stats = state->input_images->stats;
    if (!stats.enabled_exts.has_pose_timing) {
      return vit::Result::VIT_ERROR_NOT_ENABLED;
    }

    out_timing->count = stats.timings.size();
    out_timing->timestamps = stats.timings.data();
    return vit::Result::VIT_SUCCESS;
  }

  vit::Result get_features(uint32_t camera_index, vit::PoseFeatures *out_features) const {
    const vit::TimeStats &stats = state->input_images->stats;
    if (!stats.enabled_exts.has_pose_features) {
      return vit::Result::VIT_ERROR_NOT_ENABLED;
    }

    const std::vector<vit::PoseFeature> &f = stats.features_per_cam.at(camera_index);

    out_features->count = f.size();
    out_features->features = f.data();
    return vit::Result::VIT_SUCCESS;
  }
};

vit::Result Pose::get_data(vit::PoseData *data) const {
  impl_->get_data(data);
  return vit::Result::VIT_SUCCESS;
}

vit::Result Pose::get_timing(vit::PoseTiming *out_timing) const { return impl_->get_timing(out_timing); }

vit::Result Pose::get_features(uint32_t camera_index, vit::PoseFeatures *out_features) const {
  return impl_->get_features(camera_index, out_features);
}

}  // namespace basalt::vit_implementation

vit_result_t vit_tracker_create(const vit_config_t *config, vit_tracker_t **out_tracker) {
  try {
    vit::Tracker *tracker = new basalt::vit_implementation::Tracker(config);
    *out_tracker = tracker;
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << std::endl;
    return VIT_ERROR_ALLOCATION_FAILURE;
  }

  return VIT_SUCCESS;
}
