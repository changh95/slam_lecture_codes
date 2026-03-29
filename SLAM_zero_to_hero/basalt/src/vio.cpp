/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
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
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <magic_enum.hpp>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <basalt/io/dataset_io.h>
#include <basalt/io/marg_data_io.h>
#include <basalt/spline/se3_spline.h>
#include <basalt/utils/assert.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <basalt/calibration/calibration.hpp>

#include <basalt/serialization/headers_serialization.h>

#include <basalt/utils/keypoints.h>
#include <basalt/utils/system_utils.h>
#include <basalt/utils/vio_config.h>
#include <basalt/utils/vis_matrices.h>
#include <basalt/utils/vis_utils.h>
#include <basalt/linearization/landmark_block.hpp>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

// enable the "..."_format(...) string literal
using namespace basalt::literals;
using namespace basalt;
using namespace Eigen;
using pangolin::DataLog;
using pangolin::META_FLAG_READONLY;
using pangolin::Plotter;
using pangolin::Var;
using pangolin::View;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::thread;
using vis::Button;
using UIMAT = vis::UIMAT;

/*
 * basalt_vio_ui 클래스는 VIO 시스템의 UI와 실행 로직을 담당합니다.
 * 이 클래스는 다음 주요 기능을 포함합니다:
 * 1. 이미지와 IMU 데이터 로드 및 처리
 * 2. VIO 알고리즘 실행
 * 3. 결과 시각화 및 분석
 * 4. 다양한 형식으로 궤적 저장
 */
struct basalt_vio_ui : vis::VIOUIBase {
  VioDatasetPtr vio_dataset;
  int64_t start_t_ns = -1;

  DataLog imu_data_log, vio_data_log, error_data_log;
  shared_ptr<Plotter> plotter;

  pangolin::OpenGlRenderState camera;

  tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> out_vis_queue;
  tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> out_state_queue;

  std::vector<int64_t> vio_t_ns;
  Eigen::aligned_vector<Eigen::Vector3d> vio_t_w_i;
  Eigen::aligned_vector<Sophus::SE3d> vio_T_w_i;

  std::vector<int64_t> gt_t_ns;
  Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;

  std::string marg_data_path;
  size_t last_frame_processed = 0;

  tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>> timestamp_to_id;

  std::mutex m;
  std::condition_variable cvar;
  bool step_by_step = false;
  bool deterministic = false;
  size_t max_frames = 0;

  std::atomic<bool> terminate = false;

  bool show_gui = true;
  std::string trajectory_fmt;
  std::string result_path;
  bool trajectory_groundtruth = false;
  bool print_queue = false;
  std::chrono::high_resolution_clock::time_point time_start;
  bool aborted = false;

  thread feed_images_thread;
  thread feed_imu_thread;
  thread vis_thread;
  thread state_consumer_thread;
  thread queues_printer_thread;

  Var<bool> trajectory_menu{"ui.Trajectory Menu", false, true};
  Var<string> trajectory_menu_title{"trajectory_menu.MENU", "Trajectory Menu", META_FLAG_READONLY};
  Var<bool> show_gt{"trajectory_menu.show_gt", true, true};
  Button align_se3_btn{"trajectory_menu.align_se3", [this]() { alignButton(); }};
  Var<bool> euroc_fmt{"trajectory_menu.euroc_fmt", true, true};
  Var<bool> tum_rgbd_fmt{"trajectory_menu.tum_rgbd_fmt", false, true};
  Var<bool> kitti_fmt{"trajectory_menu.kitti_fmt", false, true};
  Var<bool> save_groundtruth{"trajectory_menu.save_groundtruth", false, true};
  Button save_traj_btn{"trajectory_menu.save_traj", [this]() { saveTrajectoryButton(); }};

  Button compute_frames_ate_btn{"curves_menu.compute_frames_ate", [this]() { compute_frames_ate(); }};
  Var<bool> show_frames_ate{"curves_menu.show_frames_ate", false, true};

  Button next_step_btn{"ui.next_step", [this]() { next_step(); }};
  Button prev_step_btn{"ui.prev_step", [this]() { prev_step(); }};

  Var<bool> continue_btn{"ui.continue", false, true};
  Var<bool> continue_fast{"ui.continue_fast", true, true};

  struct OfflineVIOImageView : vis::VIOImageView {
    basalt_vio_ui& vio_ui;
    OfflineVIOImageView(basalt_vio_ui& ui) : VIOImageView(ui), vio_ui(ui) {}
    void Keyboard(View& view, unsigned char key, int x, int y, bool pressed) override {
      bool released = !pressed;
      if (key == ' ' && released) {
        vio_ui.continue_btn = !vio_ui.continue_btn;
        vio_ui.continue_btn.Meta().gui_changed = true;
      } else if (key == '.' && pressed) {
        vio_ui.next_step();
      } else if (key == ',' && pressed) {
        vio_ui.prev_step();
      } else if (key == '>' && pressed) {
        vio_ui.next_step(10);
      } else if (key == '<' && pressed) {
        vio_ui.prev_step(10);
      } else {
        vis::VIOImageView::Keyboard(view, key, x, y, pressed);
      }
    }
  };

  int start(int argc, char** argv) {
    bool print_queue = false;
    std::string cam_calib_path;
    std::string dataset_path;
    std::string dataset_type;
    std::string config_path;
    int num_threads = 0;
    bool use_imu = true;
    bool use_double = false;

    CLI::App app{"Basalt CLI"};

    app.add_option("--show-gui", show_gui, "Show GUI");
    app.add_option("--cam-calib", cam_calib_path, "Ground-truth camera calibration used for simulation.")->required();
    app.add_option("--dataset-path", dataset_path, "Path to dataset.")->required();
    app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")->required();
    app.add_option("--marg-data", marg_data_path, "Path to folder where marginalization data will be stored.");
    app.add_option("--print-queue", print_queue, "Print queue.");
    app.add_option("--config-path", config_path, "Path to config file.");
    app.add_option("--result-path", result_path, "Path to result file where the system will write RMSE ATE.");
    app.add_option("--num-threads", num_threads, "Number of threads.");
    app.add_option("--step-by-step", step_by_step, "Whether to wait for manual input before running the dataset");
    app.add_option("--save-trajectory", trajectory_fmt, "Save trajectory. Supported formats <tum, euroc, kitti>");
    app.add_option("--save-groundtruth", trajectory_groundtruth, "In addition to trajectory, save also ground turth");
    app.add_option("--use-imu", use_imu, "Use IMU: visual-inertial vs visual-only pipeline");
    app.add_option("--use-double", use_double, "Use double not float.");
    app.add_option("--deterministic", deterministic, "Make the pipeline output reproducible (some performance impact)");
    app.add_option("--max-frames", max_frames, "Limit number of frames to process from dataset (0 means unlimited)");

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
      return app.exit(e);
    }

    // global thread limit is in effect until global_control object is destroyed
    std::unique_ptr<tbb::global_control> tbb_global_control;
    if (num_threads > 0) {
      tbb_global_control =
          std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);
    }

    if (!config_path.empty()) {
      config.load(config_path);

      if (config.vio_enforce_realtime) {
        config.vio_enforce_realtime = false;
        std::cout << "The option config.vio_enforce_realtime was enabled, "
                     "but it should only be used with the live executables (supply "
                     "images at a constant framerate). This executable runs on the "
                     "datasets and processes images as fast as it can, so the option "
                     "will be disabled. "
                  << std::endl;
      }
    }

    load_data(cam_calib_path);

    {
      basalt::DatasetIoInterfacePtr dataset_io = basalt::DatasetIoFactory::getDatasetIo(dataset_type);

      dataset_io->read(dataset_path);

      vio_dataset = dataset_io->get_data();
      start_t_ns = vio_dataset->get_image_timestamps().front();

      show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
      show_frame.Meta().gui_changed = true;

      /*
       * [특징점 추적 및 옵티컬 플로우 부분]
       * Basalt 시스템은 direct method를 사용하여 특징점을 추적합니다.
       * 옵티컬 플로우는 특징점 추출과 추적을 담당합니다.
       * 실제 구현은 basalt/optical_flow/ 디렉토리에 있습니다.
       * - PatchOpticalFlow: 패치 기반 추적 (basalt/optical_flow/patch_optical_flow.h)
       * - 특징점 선택, 추적, 스케일 공간 피라미드 등의 기능 포함
       */
      opt_flow = basalt::OpticalFlowFactory::getOpticalFlow(config, calib);
      opt_flow->start();

      for (size_t i = 0; i < vio_dataset->get_gt_pose_data().size(); i++) {
        gt_t_ns.push_back(vio_dataset->get_gt_timestamps()[i]);
        gt_t_w_i.push_back(vio_dataset->get_gt_pose_data()[i].translation());
      }
    }

    {
      /*
       * [VIO 알고리즘 초기화 부분]
       * 여기서 VIO 알고리즘의 실제 인스턴스가 생성됩니다.
       * VioEstimatorFactory를 통해 설정에 따라 적절한 VIO 알고리즘 인스턴스를 생성합니다.
       * 실제 알고리즘 구현은 basalt/vi_estimator/ 디렉토리에 있습니다.
       * - KeypointVioEstimator: 키포인트 기반 VIO (basalt/vi_estimator/keypoint_vio.h)
       * - stereo-VIO, mono-VIO 등 다양한 구현체가 있습니다.
       */
      vio = basalt::VioEstimatorFactory::getVioEstimator(config, calib, basalt::constants::g, use_imu, use_double);
      vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

      opt_flow->output_queue = &vio->vision_data_queue;
      opt_flow->show_gui = show_gui;
      if (show_gui) vio->out_vis_queue = &out_vis_queue;
      vio->out_state_queue = &out_state_queue;
      vio->opt_flow_depth_guess_queue = &opt_flow->input_depth_queue;
      vio->opt_flow_state_queue = &opt_flow->input_state_queue;
      vio->opt_flow_lm_bundle_queue = &opt_flow->input_lm_bundle_queue;
    }

    basalt::MargDataSaver::Ptr marg_data_saver;

    if (!marg_data_path.empty()) {
      marg_data_saver = make_shared<basalt::MargDataSaver>(marg_data_path);
      vio->out_marg_queue = &marg_data_saver->in_marg_queue;

      // Save gt.
      {
        std::string p = marg_data_path + "/gt.cereal";
        std::ofstream os(p, std::ios::binary);

        {
          cereal::BinaryOutputArchive archive(os);
          archive(gt_t_ns);
          archive(gt_t_w_i);
        }
        os.close();
      }
    }

    vio_data_log.Clear();

    feed_images_thread = thread([this]() { feed_images(); });
    feed_imu_thread = thread([this]() { feed_imu(); });

    if (show_gui)
      vis_thread = thread([&]() {
        basalt::VioVisualizationData::Ptr data;

        while (true) {
          out_vis_queue.pop(data);

          if (data.get()) {
            vis_map[data->t_ns] = data;
          } else {
            break;
          }
        }

        std::cout << "Finished t3" << std::endl;
      });

    if (!deterministic) {
      state_consumer_thread = thread([&]() {
        while (pop_state()) continue;
      });
    }

    if (print_queue) {
      queues_printer_thread = thread([&]() {
        while (!terminate) {
          print_queue_fn();
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      });
    }

    time_start = std::chrono::high_resolution_clock::now();

    // record if we close the GUI before VIO is finished.
    aborted = false;

    if (show_gui) run_ui();

    return 0;
  }

  bool pop_state() {
    /*
     * [VIO 상태 처리 함수]
     * 이 함수는 VIO 알고리즘에서 계산된 상태(pose, velocity, bias 등)를 처리합니다.
     * VIO 알고리즘은 최적화 후 새로운 상태를 out_state_queue에 넣고,
     * 이 함수는 그 결과를 가져와 궤적과 그래프 데이터를 업데이트합니다.
     * 시각화 및 결과 저장에 사용되는 vio_t_ns, vio_t_w_i, vio_T_w_i 배열이 여기서 업데이트됩니다.
     * 최종 ATE(Absolute Trajectory Error) 계산에도 이 데이터가 사용됩니다.
     */
    basalt::PoseVelBiasState<double>::Ptr data;
    out_state_queue.pop(data);

    if (data.get() == nullptr) return false;

    int64_t t_ns = data->t_ns;

    Sophus::SE3d T_w_i = data->T_w_i;
    Eigen::Vector3d vel_w_i = data->vel_w_i;
    Eigen::Vector3d bg = data->bias_gyro;
    Eigen::Vector3d ba = data->bias_accel;

    vio_t_ns.emplace_back(data->t_ns);
    vio_t_w_i.emplace_back(T_w_i.translation());
    vio_T_w_i.emplace_back(T_w_i);

    if (show_gui) {
      std::vector<float> vals;
      vals.push_back((t_ns - start_t_ns) * 1e-9);

      for (int i = 0; i < 3; i++) vals.push_back(vel_w_i[i]);
      for (int i = 0; i < 3; i++) vals.push_back(T_w_i.translation()[i]);
      for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
      for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

      vio_data_log.Log(vals);
    }

    return true;
  }

  void print_queue_fn() {
    std::cout << "opt_flow->input_img_queue " << opt_flow->input_img_queue.size() << " opt_flow->output_queue "
              << opt_flow->output_queue->size() << " out_state_queue " << out_state_queue.size() << " imu_data_queue "
              << vio->imu_data_queue.size() << std::endl;
  }

  void run_ui() {
    if (show_gui) {
      pangolin::CreateWindowAndBind("Main", 1800, 1000, vis::default_win_params);

      glEnable(GL_DEPTH_TEST);

      View& main_display = pangolin::CreateDisplay();
      main_display.SetBounds(0.0, 1.0, UI_WIDTH, 1.0);

      img_view_display = &pangolin::CreateDisplay();
      img_view_display->SetBounds(0.4, 1.0, 0.0, 0.4);
      img_view_display->SetLayout(pangolin::LayoutEqual);

      plotter = make_shared<Plotter>(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01, 0.01);
      plotter->SetBackgroundColour(vis::C_BLUEGREY_DARK());
      plotter->SetAxisColour(vis::C_BLUEGREY_LIGHT());
      plotter->SetTickColour(vis::C_BLUEGREY());
      plot_display = &pangolin::CreateDisplay();
      plot_display->SetBounds(0.0, 0.4, UI_WIDTH, 1.0);
      plot_display->AddDisplay(*plotter);

      blocks_view = make_shared<pangolin::ImageView>();
      blocks_view->UseNN() = true;  // Disable antialiasing, can be toggled with N key
      blocks_view->extern_draw_function = [this](View& /*v*/) { draw_blocks_overlay(); };
      const int DEFAULT_W = 480;
      blocks_display = &pangolin::CreateDisplay();
      blocks_display->SetBounds(0.0, 0.6, UI_WIDTH, pangolin::Attach::Pix(UI_WIDTH_PIX + DEFAULT_W));
      blocks_display->AddDisplay(*blocks_view);
      blocks_display->Show(show_blocks);

      pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, UI_WIDTH);
      menus.push_back(&trajectory_menu);
      menus_str.emplace_back("trajectory_menu");
      for (size_t i = 0; i < menus.size(); i++) {
        pangolin::CreatePanel(menus_str[i]).SetBounds(0.0, 0.4, 0.0, UI_WIDTH);
        pangolin::Display(menus_str[i]).Show(*menus[i]);
      }

      while (img_view.size() < calib.intrinsics.size()) {
        auto iv = make_shared<OfflineVIOImageView>(*this);
        iv->UseNN() = true;  // Disable antialiasing, can be toggled with N key

        size_t idx = img_view.size();
        img_view.push_back(iv);

        img_view_display->AddDisplay(*iv);
        iv->extern_draw_function = [idx, this](View& v) {
          return draw_image_overlay(dynamic_cast<pangolin::ImageView&>(v), idx);
        };
      }

      Eigen::Vector3d cam_p(-0.5, -3, -5);
      cam_p = vio->getT_w_i_init().so3() * calib.T_i_c[0].so3() * cam_p;

      camera = pangolin::OpenGlRenderState(
          pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
          pangolin::ModelViewLookAt(cam_p[0], cam_p[1], cam_p[2], 0, 0, 0, pangolin::AxisZ));

      View& display3D = pangolin::CreateDisplay();
      display3D.SetAspect(-640 / 480.0);
      display3D.SetBounds(0.4, 1.0, 0.4, 1.0);
      display3D.SetHandler(new pangolin::Handler3D(camera));

      display3D.extern_draw_function = [this](View& v) { draw_scene(v); };

      main_display.AddDisplay(*img_view_display);
      main_display.AddDisplay(display3D);

      while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        size_t frame_id = show_frame;
        int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];
        auto it = vis_map.find(t_ns);
        if (follow) {
          if (it != vis_map.end()) {
            Sophus::SE3d T_w_i;
            if (!it->second->states.empty()) {
              T_w_i = it->second->states.rbegin()->second;
            } else if (!it->second->frames.empty()) {
              T_w_i = it->second->frames.rbegin()->second;
            }
            T_w_i.so3() = Sophus::SO3d();

            camera.Follow(T_w_i.matrix());
          }
        }

        display3D.Activate(camera);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        img_view_display->Activate();
        if (fixed_depth.GuiChanged() && vio->opt_flow_depth_guess_queue != nullptr) {
          vio->opt_flow_depth_guess_queue->push(fixed_depth);
          depth_guess = fixed_depth;
        } else if (it != vis_map.end() && it->second->opt_flow_res && it->second->opt_flow_res->input_images) {
          depth_guess = it->second->opt_flow_res->input_images->depth_guess;
        }

        if (show_frame.GuiChanged()) {
          for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
            auto frame_id = static_cast<size_t>(show_frame);
            int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

            std::vector<basalt::ImageData> img_vec = vio_dataset->get_image_data(timestamp);

            pangolin::GlPixFormat fmt;
            fmt.glformat = GL_LUMINANCE;
            fmt.gltype = GL_UNSIGNED_SHORT;
            fmt.scalable_internal_format = GL_LUMINANCE16;

            if (img_vec[cam_id].img.get())
              img_view[cam_id]->SetImage(img_vec[cam_id].img->ptr, img_vec[cam_id].img->w, img_vec[cam_id].img->h,
                                         img_vec[cam_id].img->pitch, fmt);
          }
          if (follow_highlight) do_follow_highlight(true, false);

          if (show_blocks) do_show_blocks();

          draw_plots();
        }

        if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() || show_est_ba.GuiChanged() ||
            show_est_bg.GuiChanged() || show_frames_ate.GuiChanged()) {
          draw_plots();
        }

        if (highlight_landmarks.GuiChanged() || filter_highlights.GuiChanged() || show_highlights.GuiChanged() ||
            show_ids.GuiChanged()) {
          highlights = vis::parse_selection(highlight_landmarks);
          filter_highlights = filter_highlights && !highlights.empty();
          for (const auto& [ts, vis] : vis_map) vis->invalidate_mat_imgs();
          if (show_blocks) do_show_blocks();
        }

        if (mat_to_show.GuiChanged()) {
          mat_name = std::string(magic_enum::enum_name((UIMAT)mat_to_show.Get()));
          if (show_blocks) do_show_blocks();
        }

        if (follow_highlight.GuiChanged()) {
          follow_highlight = follow_highlight && !highlights.empty();
          do_follow_highlight(follow_highlight, true);
        }

        if (euroc_fmt.GuiChanged()) {
          euroc_fmt = true;
          tum_rgbd_fmt = false;
          kitti_fmt = false;
        }

        if (tum_rgbd_fmt.GuiChanged()) {
          tum_rgbd_fmt = true;
          euroc_fmt = false;
          kitti_fmt = false;
        }

        if (kitti_fmt.GuiChanged()) {
          kitti_fmt = true;
          euroc_fmt = false;
          tum_rgbd_fmt = false;
        }

        Var<bool>* selected_menu = nullptr;
        for (Var<bool>* menu : menus) {
          if (menu->GuiChanged()) {
            selected_menu = menu;
            break;
          }
        }
        if (selected_menu != nullptr) {
          for (size_t i = 0; i < menus.size(); i++) {
            *menus[i] = menus[i] != selected_menu ? false : *selected_menu;
            pangolin::Display(menus_str[i]).Show(*menus[i]);
          }
        }

        //      if (record) {
        //        main_display.RecordOnRender(
        //            "ffmpeg:[fps=50,bps=80000000,unique_filename]///tmp/"
        //            "vio_screencap.avi");
        //        record = false;
        //      }

        pangolin::FinishFrame();

        if (continue_btn) {
          if (!next_step()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (continue_fast) {
          int64_t t_ns = vio->last_processed_t_ns;
          if (timestamp_to_id.count(t_ns)) {
            show_frame = timestamp_to_id[t_ns];
            show_frame.Meta().gui_changed = true;
          }

          if (vio->finished) {
            continue_fast = false;
          }
        }
      }

      // If GUI closed but VIO not yet finished --> abort input queues, which in
      // turn aborts processing
      if (!vio->finished) {
        std::cout << "GUI closed but odometry still running --> aborting.\n";
        print_queue_fn();  // print queue size at time of aborting
        terminate = true;
        aborted = true;
      }
    }
  }

  void stop() {
    // wait first for vio to complete processing
    vio->maybe_join();

    // input threads will abort when vio is finished, but might be stuck in full
    // push to full queue, so drain queue now
    vio->drain_input_queues();
    opt_flow->drain_input_queues();

    // join input threads
    feed_images_thread.join();
    feed_imu_thread.join();

    // std::cout << "Data input finished, terminate auxiliary threads.";
    terminate = true;

    // join other threads
    if (show_gui) vis_thread.join();
    if (!deterministic) state_consumer_thread.join();
    if (print_queue) queues_printer_thread.join();

    // after joining all threads, print final queue sizes.
    if (print_queue) {
      std::cout << "Final queue sizes:" << std::endl;
      print_queue_fn();
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    const double duration_total = std::chrono::duration<double>(time_end - time_start).count();

    // TODO: remove this unconditional call (here for debugging);
    const double ate_rmse = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);
    vio->debug_finalize();
    std::cout << "Total runtime: {:.3f}s\n"_format(duration_total);

    {
      basalt::ExecutionStats stats;
      stats.add("exec_time_s", duration_total);
      stats.add("ate_rmse", ate_rmse);
      stats.add("ate_num_kfs", vio_t_w_i.size());
      stats.add("num_frames", vio_dataset->get_image_timestamps().size());

      {
        basalt::MemoryInfo mi;
        if (get_memory_info(mi)) {
          stats.add("resident_memory_peak", mi.resident_memory_peak);
        }
      }

      stats.save_json("stats_vio.json");
    }

    if (!aborted && !trajectory_fmt.empty()) {
      std::cout << "Saving trajectory..." << std::endl;

      if (trajectory_fmt == "kitti") {
        kitti_fmt = true;
        euroc_fmt = false;
        tum_rgbd_fmt = false;
      }
      if (trajectory_fmt == "euroc") {
        euroc_fmt = true;
        kitti_fmt = false;
        tum_rgbd_fmt = false;
      }
      if (trajectory_fmt == "tum") {
        tum_rgbd_fmt = true;
        euroc_fmt = false;
        kitti_fmt = false;
      }

      save_groundtruth = trajectory_groundtruth;

      saveTrajectoryButton();
    }

    if (!aborted && !result_path.empty()) {
      double error = basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i);

      auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start);

      std::ofstream os(result_path);
      {
        cereal::JSONOutputArchive ar(os);
        ar(cereal::make_nvp("rms_ate", error));
        ar(cereal::make_nvp("num_frames", vio_dataset->get_image_timestamps().size()));
        ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
      }
      os.close();
    }
  }

  VioVisualizationData::Ptr get_curr_vis_data() override {
    int64_t curr_ts = vio_dataset->get_image_timestamps().at(show_frame);
    auto it = vis_map.find(curr_ts);
    if (it == vis_map.end()) return nullptr;
    VioVisualizationData::Ptr curr_vis_data = it->second;
    return curr_vis_data;
  }

  // Feed functions
  void feed_images() {
    /*
     * [이미지 피딩 쓰레드]
     * 이 함수는 독립된 쓰레드에서 실행되며 이미지 데이터를 VIO 시스템에 공급합니다.
     * 데이터셋에서 이미지를 로드하여 OpticalFlow 모듈의 input_img_queue에 넣습니다.
     * OpticalFlow 모듈은 특징점 추출 및 추적 작업을 수행한 후 결과를 VIO 알고리즘에 전달합니다.
     * 이 데이터 흐름이 Basalt의 프론트엔드-백엔드 구조의 핵심입니다.
     */
    std::cout << "Started input_data thread " << std::endl;

    int NUM_CAMS = calib.intrinsics.size();
    for (size_t i = 0; i < vio_dataset->get_image_timestamps().size(); i++) {
      if (vio->finished || terminate || (max_frames > 0 && i >= max_frames)) {
        // stop loop early if we set a limit on number of frames to process
        break;
      }

      if (step_by_step) {
        std::unique_lock<std::mutex> lk(m);
        cvar.wait(lk);
      }

      basalt::OpticalFlowInput::Ptr img(new basalt::OpticalFlowInput(NUM_CAMS));

      img->t_ns = vio_dataset->get_image_timestamps()[i];
      img->img_data = vio_dataset->get_image_data(img->t_ns);

      timestamp_to_id[img->t_ns] = i;

      opt_flow->input_img_queue.push(img);

      if (deterministic) pop_state();  // Wait for the state to be produced
    }

    // Indicate the end of the sequence
    opt_flow->input_img_queue.push(nullptr);

    std::cout << "Finished input_data thread " << std::endl;
  }

  void feed_imu() {
    /*
     * [IMU 데이터 피딩 쓰레드]
     * 이 함수는 독립된 쓰레드에서 실행되며 IMU 데이터를 VIO 시스템에 공급합니다.
     * 자이로스코프(각속도)와 가속도계 데이터를 패키징하여 VIO 알고리즘에 전달합니다.
     * Basalt는 IMU 예적분(preintegration)을 사용하여 이미지 프레임 사이의 IMU 측정값을 결합합니다.
     * 이 과정은 basalt/vi_estimator/ 디렉토리의 imu_integrator.h에 구현되어 있습니다.
     */
    for (size_t i = 0; i < vio_dataset->get_gyro_data().size(); i++) {
      if (vio->finished || terminate) {
        break;
      }

      basalt::ImuData<double>::Ptr data(new basalt::ImuData<double>);
      data->t_ns = vio_dataset->get_gyro_data()[i].timestamp_ns;

      data->accel = vio_dataset->get_accel_data()[i].data;
      data->gyro = vio_dataset->get_gyro_data()[i].data;

      vio->imu_data_queue.push(data);
      opt_flow->input_imu_queue.push(data);
    }
    vio->imu_data_queue.push(nullptr);
    opt_flow->input_imu_queue.push(nullptr);
  }

  void draw_image_overlay(pangolin::ImageView& v, size_t cam_id) {
    UNUSED(v);
    VioVisualizationData::Ptr curr_vis_data = get_curr_vis_data();
    if (curr_vis_data == nullptr) return;

    if (show_obs) do_show_obs(cam_id);
    if (show_flow) do_show_flow(cam_id);
    if (show_highlights) do_show_highlights(cam_id);
    if (show_tracking_guess) do_show_tracking_guess_vio(cam_id, show_frame, vio_dataset, vis_map);
    if (show_matching_guess) do_show_matching_guesses(cam_id);
    if (show_recall_guess) do_show_recall_guesses(cam_id);
    if (show_masks) do_show_masks(cam_id);
    if (show_cam0_proj) do_show_cam0_proj(cam_id, depth_guess);
    if (show_grid) do_show_grid();
    if (show_safe_radius) do_show_safe_radius();
  }

  void draw_scene(View& view) {
    UNUSED(view);
    view.Activate(camera);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glPointSize(3);
    glColor3f(1.0, 0.0, 0.0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor3ubv(cam_color);
    if (!vio_t_w_i.empty()) {
      size_t end = std::min(vio_t_w_i.size(), size_t(show_frame + 1));
      Eigen::aligned_vector<Eigen::Vector3d> sub_gt(vio_t_w_i.begin(), vio_t_w_i.begin() + end);
      pangolin::glDrawLineStrip(sub_gt);
    }

    glColor3ubv(gt_color);
    if (show_gt) pangolin::glDrawLineStrip(gt_t_w_i);

    VioVisualizationData::Ptr curr_vis_data = get_curr_vis_data();
    if (curr_vis_data == nullptr) return;

    for (size_t i = 0; i < calib.T_i_c.size(); i++) {
      if (!curr_vis_data->states.empty()) {
        const auto& [ts, p] = *curr_vis_data->states.rbegin();
        do_render_camera(p * calib.T_i_c[i], i, ts, cam_color);
      } else if (!curr_vis_data->frames.empty()) {
        const auto& [ts, p] = *curr_vis_data->frames.rbegin();
        do_render_camera(p * calib.T_i_c[i], i, ts, cam_color);
      }
    }

    for (const auto& [ts, p] : curr_vis_data->states)
      for (size_t i = 0; i < calib.T_i_c.size(); i++) do_render_camera(p * calib.T_i_c[i], i, ts, state_color);

    for (const auto& [ts, p] : curr_vis_data->frames)
      for (size_t i = 0; i < calib.T_i_c.size(); i++) do_render_camera(p * calib.T_i_c[i], i, ts, pose_color);

    for (const auto& [ts, p] : curr_vis_data->ltframes)
      for (size_t i = 0; i < calib.T_i_c.size(); i++) do_render_camera(p * calib.T_i_c[i], i, ts, vis::BLUE);

    glColor3ubv(pose_color);
    if (!filter_highlights) pangolin::glDrawPoints(curr_vis_data->points);

    Eigen::aligned_vector<Eigen::Vector3d> highlighted_points;
    if (show_highlights || filter_highlights) {
      for (size_t i = 0; i < curr_vis_data->point_ids.size(); i++) {
        Vector3d pos = curr_vis_data->points.at(i);
        int id = curr_vis_data->point_ids.at(i);
        if (is_highlighted(id)) highlighted_points.push_back(pos);
      }
    }

    if (filter_highlights) pangolin::glDrawPoints(highlighted_points);

    if (show_highlights) {
      glColor3ubv(vis::GREEN);
      glPointSize(10);
      pangolin::glDrawPoints(highlighted_points);
    }

    glColor3ubv(pose_color);
    if (show_ids) {
      for (size_t i = 0; i < curr_vis_data->points.size(); i++) {
        Vector3d pos = curr_vis_data->points.at(i);
        int id = curr_vis_data->point_ids.at(i);

        bool highlighted = is_highlighted(id);
        if (filter_highlights && !highlighted) continue;

        if (show_highlights && highlighted) glColor3ubv(vis::GREEN);
        FONT.Text("%d", id).Draw(pos.x(), pos.y(), pos.z());
        if (show_highlights && highlighted) glColor3ubv(pose_color);
      }
    }

    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
  }

  void load_data(const std::string& calib_path) {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
      std::cout << "Loaded camera with " << calib.intrinsics.size() << " cameras" << std::endl;

    } else {
      std::cerr << "could not load camera calibration " << calib_path << std::endl;
      std::abort();
    }
  }

  bool next_step(int steps = 1) {
    if (show_frame < int(vio_dataset->get_image_timestamps().size()) - steps) {
      show_frame = show_frame + steps;
      show_frame.Meta().gui_changed = true;
      cvar.notify_one();
      return true;
    } else {
      return false;
    }
  }

  bool prev_step(int steps = 1) {
    if (show_frame >= steps) {
      show_frame = show_frame - steps;
      show_frame.Meta().gui_changed = true;
      return true;
    } else {
      return false;
    }
  }

  void draw_plots() {
    plotter->ClearSeries();
    plotter->ClearMarkers();

    if (show_est_pos) {
      plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine, vis::C_RED(), "position x", &vio_data_log);
      plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine, vis::C_GREEN(), "position y", &vio_data_log);
      plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine, vis::C_BLUE(), "position z", &vio_data_log);
    }

    if (show_est_vel) {
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine, vis::C_RED_DARK(), "velocity x", &vio_data_log);
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine, vis::C_GREEN_DARK(), "velocity y", &vio_data_log);
      plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine, vis::C_BLUE_DARK(), "velocity z", &vio_data_log);
    }

    if (show_est_bg) {
      plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine, vis::C_RED(), "gyro bias x", &vio_data_log);
      plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine, vis::C_GREEN(), "gyro bias y", &vio_data_log);
      plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine, vis::C_BLUE(), "gyro bias z", &vio_data_log);
    }

    if (show_est_ba) {
      plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine, vis::C_RED_DARK(), "accel bias x", &vio_data_log);
      plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine, vis::C_GREEN_DARK(), "accel bias y", &vio_data_log);
      plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine, vis::C_BLUE_DARK(), "accel bias z", &vio_data_log);
    }

    if (show_frames_ate) {
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine, vis::C_AMBER(), "ATE [cm]", &error_data_log);
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine, vis::C_PINK(), "ATE diffs [mm]", &error_data_log);
    }

    double t = (vio_dataset->get_image_timestamps()[show_frame] - start_t_ns) * 1e-9;
    plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal, pangolin::Colour::White());
  }

  void alignButton() { basalt::alignSVD(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i); }

  void compute_frames_ate() {
    /*
     * [ATE(Absolute Trajectory Error) 계산 함수]
     * 이 함수는 VIO로 추정한 궤적과 그라운드 트루스 사이의 오차를 계산합니다.
     * 주요 단계:
     * 1. 추정 궤적과 그라운드 트루스 타임스탬프 연관 찾기(associate)
     * 2. 두 궤적 사이의 최적 변환 행렬 계산(get_alignment)
     * 3. 정렬된 궤적 사이의 오차 계산(compute_ate)
     * 4. 결과 로깅 및 시각화
     * 
     * RMSE ATE는 VSLAM/VIO 시스템 평가의 표준 지표 중 하나입니다.
     */
    Eigen::Matrix<int64_t, Eigen::Dynamic, 1> ts{};
    Eigen::Matrix<float, 3, Eigen::Dynamic> est_xyz{};
    Eigen::Matrix<float, 3, Eigen::Dynamic> ref_xyz{};
    int num_assocs = associate(vio_t_ns, vio_t_w_i, gt_t_ns, gt_t_w_i, ts, est_xyz, ref_xyz);

    error_data_log.Clear();
    constexpr int SAMPLE_POINTS = 10000;
    int sample_points = std::min(SAMPLE_POINTS, num_assocs);
    int step = num_assocs / sample_points;

    float prev_err = 0;
    for (int i = 1; i < num_assocs; i += step) {
      Eigen::Matrix4f T_ref_est = get_alignment(est_xyz, ref_xyz, 0, i);
      float error = compute_ate(est_xyz, ref_xyz, T_ref_est, 0, i);
      error_data_log.Log({float((ts(i - 1) - start_t_ns) * 1e-9), error * 1e3f, (error - prev_err) * 1e6f});
      prev_err = error;
    }

    // Ensure last frame error is computed and logged
    int i = num_assocs;
    Eigen::Matrix4f T_ref_est = get_alignment(est_xyz, ref_xyz, 0, i);
    float error = compute_ate(est_xyz, ref_xyz, T_ref_est, 0, i);
    error_data_log.Log({float((ts(i - 1) - start_t_ns) * 1e-9), error * 1e3f, (error - prev_err) * 1e6f});
    prev_err = error;

    std::cout << "[Final ATE]\n";
    std::cout << "T_align\n" << T_ref_est.matrix() << std::endl;
    std::cout << "error " << error << std::endl;
    std::cout << "number of associations " << num_assocs << std::endl;

    show_frames_ate = true;
    show_frames_ate.Meta().gui_changed = true;
  }

  void saveTrajectoryButton() {
    /*
     * [궤적 저장 함수]
     * 이 함수는 VIO로 추정한 카메라 궤적을 파일로 저장합니다.
     * 지원하는 형식:
     * 1. TUM RGB-D: timestamp tx ty tz qx qy qz qw 형식(초 단위 타임스탬프)
     * 2. EuRoC: timestamp, tx, ty, tz, qw, qx, qy, qz 형식(나노초 단위 타임스탬프)
     * 3. KITTI: 3x4 변환 행렬 형식
     * 
     * 이 궤적 파일은 evo 등의 평가 도구로 분석할 수 있습니다.
     * 그라운드 트루스 궤적도 같은 형식으로 저장 가능합니다(save_groundtruth 옵션).
     */
    if (tum_rgbd_fmt) {
      {
        std::ofstream os("trajectory.txt");

        os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

        for (size_t i = 0; i < vio_t_ns.size(); i++) {
          const Sophus::SE3d& pose = vio_T_w_i[i];
          os << std::scientific << std::setprecision(18) << vio_t_ns[i] * 1e-9 << " " << pose.translation().x() << " "
             << pose.translation().y() << " " << pose.translation().z() << " " << pose.unit_quaternion().x() << " "
             << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w()
             << std::endl;
        }

        os.close();
      }

      if (save_groundtruth) {
        std::ofstream os("groundtruth.txt");

        os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

        for (size_t i = 0; i < gt_t_ns.size(); i++) {
          const Eigen::Vector3d& pos = gt_t_w_i[i];
          os << std::scientific << std::setprecision(18) << gt_t_ns[i] * 1e-9 << " " << pos.x() << " " << pos.y() << " "
             << pos.z() << " 0 0 0 1" << std::endl;
        }

        os.close();
      }

      std::cout << "Saved trajectory in TUM RGB-D Dataset format in trajectory.txt" << std::endl;
    } else if (euroc_fmt) {
      std::ofstream os("trajectory.csv");

      os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
            "[],q_RS_x [],q_RS_y [],q_RS_z []"
         << std::endl;

      for (size_t i = 0; i < vio_t_ns.size(); i++) {
        const Sophus::SE3d& pose = vio_T_w_i[i];
        os << std::fixed << std::setprecision(10) << vio_t_ns[i] << "," << pose.translation().x() << ","
           << pose.translation().y() << "," << pose.translation().z() << "," << pose.unit_quaternion().w() << ","
           << pose.unit_quaternion().x() << "," << pose.unit_quaternion().y() << "," << pose.unit_quaternion().z()
           << std::endl;
      }

      std::cout << "Saved trajectory in Euroc Dataset format in trajectory.csv" << std::endl;
    } else {
      std::ofstream os("trajectory_kitti.txt");

      for (size_t i = 0; i < vio_t_ns.size(); i++) {
        Eigen::Matrix<double, 3, 4> mat = vio_T_w_i[i].matrix3x4();
        os << std::scientific << std::setprecision(12) << mat.row(0) << " " << mat.row(1) << " " << mat.row(2) << " "
           << std::endl;
      }

      os.close();

      std::cout << "Saved trajectory in KITTI Dataset format in trajectory_kitti.txt" << std::endl;
    }
  }
};

int main(int argc, char** argv) {
  basalt_vio_ui u{};

  int ret = u.start(argc, argv);
  if (ret != 0) return ret;

  u.stop();

  return 0;
}
