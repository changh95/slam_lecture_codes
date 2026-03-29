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
#pragma once

#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include <Eigen/Geometry>

#include <basalt/utils/vio_config.h>

#include <basalt/image/image_pyr.h>
#include <basalt/imu/imu_types.h>
#include <basalt/io/dataset_io.h>
#include <basalt/optical_flow/patch.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/keypoints.h>
#include <basalt/utils/vis_matrices.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/stereographic_param.hpp>
#include <basalt/utils/sophus_utils.hpp>
#include <vit_implementation_helper.hpp>
#include "sophus/se3.hpp"

#include <tbb/concurrent_queue.h>

namespace basalt {

using Keypoint = Eigen::AffineCompact2f;
using KeypointId = size_t;
using Keypoints = Eigen::aligned_map<KeypointId, Keypoint>;
using KeypointLevels = std::map<KeypointId, size_t>;
using KeypointResponses = std::map<KeypointId, float>;
using LandmarkId = KeypointId;

struct LandmarkBundle {
  using Ptr = std::shared_ptr<LandmarkBundle>;
  int64_t ts = -1;
  Eigen::aligned_vector<LandmarkId> lmids = {};
  Eigen::aligned_vector<Eigen::Vector4f> lms = {};
};

struct OpticalFlowInput {
  using Ptr = std::shared_ptr<OpticalFlowInput>;
  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using UIMAT = vis::UIMAT;

  OpticalFlowInput() = default;

  OpticalFlowInput(int NUM_CAMS) {
    img_data.resize(NUM_CAMS);
    masks.resize(NUM_CAMS);
  }

  int64_t t_ns;
  std::vector<ImageData> img_data;

  // Recorded internal pipeline values for UI playback
  double depth_guess = -1;        //!< Depth guess to use for all features
  bool state_reset = false;       //!< Whether to schedule a state reset in the backend
  std::vector<Masks> masks{};     //!< Regions of the image to ignore
  UIMAT show_uimat = UIMAT::ALL;  //!< Which matrix to compute for the UI

  vit::TimeStats stats;  //!< Keeps track of internal metrics for this t_ns
  void addTime(const char* name, int64_t custom_ts = INT64_MIN) { stats.addTime(name, custom_ts); }
};

struct OpticalFlowResult {
  using Ptr = std::shared_ptr<OpticalFlowResult>;

  int64_t t_ns;
  std::vector<Keypoints> keypoints;
  std::vector<Keypoints> tracking_guesses;
  std::vector<Keypoints> matching_guesses;
  std::vector<Keypoints> recall_guesses;

  std::vector<KeypointLevels> pyramid_levels;
  std::vector<KeypointResponses> keypoint_responses;

  OpticalFlowInput::Ptr input_images;
};

class OpticalFlowBase {
 public:
  using Ptr = std::shared_ptr<OpticalFlowBase>;

  OpticalFlowBase(const VioConfig& conf) : config(conf) {
    input_img_queue.set_capacity(10);
    input_imu_queue.set_capacity(300);
    // patch_coord is initialized in OpticalFlowTyped since we need the Pattern type
    // patch_coord = PatchT::pattern2.template cast<float>();
    depth_guess = config.optical_flow_matching_default_depth;
  }
  virtual ~OpticalFlowBase() { processing_thread->join(); }

  virtual void processingLoop() = 0;

  void start() { processing_thread.reset(new std::thread(&OpticalFlowBase::processingLoop, this)); }

  virtual inline void drain_input_queues() {
    while (!input_img_queue.empty()) {
      OpticalFlowInput::Ptr _;
      input_img_queue.pop(_);
    }
    while (!input_imu_queue.empty()) {
      ImuData<double>::Ptr _;
      input_imu_queue.pop(_);
    }
    while (!input_depth_queue.empty()) {
      double _;
      input_depth_queue.try_pop(_);
    }
    while (!input_state_queue.empty()) {
      PoseVelBiasState<double>::Ptr _;
      input_state_queue.try_pop(_);
    }
  }

  tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr> input_img_queue;
  tbb::concurrent_bounded_queue<ImuData<double>::Ptr> input_imu_queue;
  tbb::concurrent_queue<double> input_depth_queue;
  tbb::concurrent_queue<PoseVelBiasState<double>::Ptr> input_state_queue;
  tbb::concurrent_queue<LandmarkBundle::Ptr> input_lm_bundle_queue;
  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* output_queue = nullptr;

  Eigen::MatrixXf patch_coord;
  double depth_guess = -1;
  LandmarkBundle::Ptr latest_lm_bundle;
  PoseVelBiasState<double>::Ptr latest_state = nullptr;
  PoseVelBiasState<double>::Ptr predicted_state = nullptr;

  bool first_state_arrived = false;
  bool show_gui;  //!< Whether we need to store additional info for the UI

  int64_t t_ns = -1;
  size_t frame_counter = 0;
  KeypointId last_keypoint_id = 0;

  VioConfig config;

  OpticalFlowResult::Ptr transforms;
  std::shared_ptr<std::thread> processing_thread;
  std::shared_ptr<std::vector<ManagedImagePyr<uint16_t>>> old_pyramid;
  std::shared_ptr<std::vector<ManagedImagePyr<uint16_t>>> pyramid;
};

template <typename Scalar, template <typename> typename Pattern>
class OpticalFlowTyped : public OpticalFlowBase {
 public:
  using Ptr = std::shared_ptr<OpticalFlowTyped>;

  using PatchT = OpticalFlowPatch<Scalar, Pattern<Scalar>>;

  using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
  using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

  using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
  using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;

  using SE2 = Sophus::SE2<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;

  OpticalFlowTyped(const VioConfig& conf, const basalt::Calibration<double>& cal)
      : OpticalFlowBase(conf), calib(cal.template cast<Scalar>()) {
    patch_coord = PatchT::pattern2.template cast<float>();

    E.resize(getNumCams());
    for (size_t i = 0; i < getNumCams(); i++) {
      Eigen::Matrix4d Ed;
      SE3 T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j.template cast<double>(), Ed);
      E[i] = Ed.cast<Scalar>();
    }
  }

  virtual ~OpticalFlowTyped() = default;

  size_t getNumCams() const { return calib.intrinsics.size(); }

  const Calibration<Scalar> calib;
  std::vector<Matrix4> E;  // Essential matrix w.r.t. cam 0
};

class OpticalFlowFactory {
 public:
  static OpticalFlowBase::Ptr getOpticalFlow(const VioConfig& config, const Calibration<double>& cam);
};
}  // namespace basalt
