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

#include <basalt/vi_estimator/marg_helper.h>
#include <basalt/vi_estimator/sqrt_keypoint_vio.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optimization/accumulator.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/system_utils.h>
#include <basalt/vi_estimator/sc_ba_base.h>
#include <basalt/utils/cast_utils.hpp>
#include <basalt/utils/format.hpp>
#include <basalt/utils/time_utils.hpp>

#include <basalt/linearization/linearization_base.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <chrono>

namespace basalt {

template <class Scalar_>
SqrtKeypointVioEstimator<Scalar_>::SqrtKeypointVioEstimator(const Eigen::Vector3d& g_,
                                                            const basalt::Calibration<double>& calib_,
                                                            const VioConfig& config_)
    : take_kf(true),
      frames_after_kf(0),
      g(g_.cast<Scalar>()),
      initialized(false),
      schedule_reset(false),
      config(config_),
      lambda(config_.vio_lm_lambda_initial),
      min_lambda(config_.vio_lm_lambda_min),
      max_lambda(config_.vio_lm_lambda_max),
      lambda_vee(2) {
  obs_std_dev = Scalar(config.vio_obs_std_dev);
  huber_thresh = Scalar(config.vio_obs_huber_thresh);
  calib = calib_.cast<Scalar>();

  // Setup marginalization
  marg_data.is_sqrt = config.vio_sqrt_marg;
  marg_data.H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  marg_data.b.setZero(POSE_VEL_BIAS_SIZE);

  // Version without prior
  nullspace_marg_data.is_sqrt = marg_data.is_sqrt;
  nullspace_marg_data.H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  nullspace_marg_data.b.setZero(POSE_VEL_BIAS_SIZE);

  if (marg_data.is_sqrt) {
    // prior on position
    marg_data.H.diagonal().template head<3>().setConstant(std::sqrt(Scalar(config.vio_init_pose_weight)));
    // prior on yaw
    marg_data.H(5, 5) = std::sqrt(Scalar(config.vio_init_pose_weight));

    // small prior to avoid jumps in bias
    marg_data.H.diagonal().template segment<3>(9).array() = std::sqrt(Scalar(config.vio_init_ba_weight));
    marg_data.H.diagonal().template segment<3>(12).array() = std::sqrt(Scalar(config.vio_init_bg_weight));
  } else {
    // prior on position
    marg_data.H.diagonal().template head<3>().setConstant(Scalar(config.vio_init_pose_weight));
    // prior on yaw
    marg_data.H(5, 5) = Scalar(config.vio_init_pose_weight);

    // small prior to avoid jumps in bias
    marg_data.H.diagonal().template segment<3>(9).array() = Scalar(config.vio_init_ba_weight);
    marg_data.H.diagonal().template segment<3>(12).array() = Scalar(config.vio_init_bg_weight);
  }

  std::cout << "marg_H (sqrt:" << marg_data.is_sqrt << ")\n" << marg_data.H << std::endl;

  gyro_bias_sqrt_weight = calib.gyro_bias_std.array().inverse();
  accel_bias_sqrt_weight = calib.accel_bias_std.array().inverse();

  max_states = config.vio_max_states;
  max_kfs = config.vio_max_kfs;

  opt_started = false;

  vision_data_queue.set_capacity(10);
  imu_data_queue.set_capacity(300);
}

template <class Scalar>
void SqrtKeypointVioEstimator<Scalar>::scheduleResetState() {
  schedule_reset = true;
}

template <class Scalar>
void SqrtKeypointVioEstimator<Scalar>::takeLongTermKeyframe() {
  take_ltkf = true;
}

template <class Scalar>
bool SqrtKeypointVioEstimator<Scalar>::resetState(typename IntegratedImuMeasurement<Scalar>::Ptr& meas,
                                                  OpticalFlowResult::Ptr& curr_frame,
                                                  OpticalFlowResult::Ptr& prev_frame) {
  meas = nullptr;
  curr_frame = nullptr;
  prev_frame = nullptr;

  typename ImuData<Scalar>::Ptr data = popFromImuDataQueue();
  if (data == nullptr) return true;
  data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
  data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);

  initialized = false;
  last_processed_t_ns = 0;
  // drain_input_queues();
  frame_states.clear();
  frame_poses.clear();
  lmdb.clear();

  take_kf = true;
  frames_after_kf = 0;
  kf_ids.clear();
  ltkfs.clear();
  imu_meas.clear();
  prev_opt_flow_res.clear();
  num_points_kf.clear();

  // Setup marginalization
  marg_data.is_sqrt = config.vio_sqrt_marg;
  marg_data.order.abs_order_map.clear();
  marg_data.order.items = 0;
  marg_data.order.total_size = 0;
  marg_data.H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  marg_data.b.setZero(POSE_VEL_BIAS_SIZE);

  // Version without prior
  nullspace_marg_data.is_sqrt = marg_data.is_sqrt;
  nullspace_marg_data.order.abs_order_map.clear();
  nullspace_marg_data.order.items = 0;
  nullspace_marg_data.order.total_size = 0;
  nullspace_marg_data.H.setZero(POSE_VEL_BIAS_SIZE, POSE_VEL_BIAS_SIZE);
  nullspace_marg_data.b.setZero(POSE_VEL_BIAS_SIZE);

  if (marg_data.is_sqrt) {
    // prior on position
    marg_data.H.diagonal().template head<3>().setConstant(std::sqrt(Scalar(config.vio_init_pose_weight)));
    // prior on yaw
    marg_data.H(5, 5) = std::sqrt(Scalar(config.vio_init_pose_weight));

    // small prior to avoid jumps in bias
    marg_data.H.diagonal().template segment<3>(9).array() = std::sqrt(Scalar(config.vio_init_ba_weight));
    marg_data.H.diagonal().template segment<3>(12).array() = std::sqrt(Scalar(config.vio_init_bg_weight));
  } else {
    // prior on position
    marg_data.H.diagonal().template head<3>().setConstant(Scalar(config.vio_init_pose_weight));
    // prior on yaw
    marg_data.H(5, 5) = Scalar(config.vio_init_pose_weight);

    // small prior to avoid jumps in bias
    marg_data.H.diagonal().template segment<3>(9).array() = Scalar(config.vio_init_ba_weight);
    marg_data.H.diagonal().template segment<3>(12).array() = Scalar(config.vio_init_bg_weight);
  }

  std::cout << "marg_H (sqrt:" << marg_data.is_sqrt << ")\n" << marg_data.H << std::endl;

  opt_started = false;
  schedule_reset = false;
  return false;
}

template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                                                   const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                                                   const Eigen::Vector3d& ba) {
  initialized = true;
  T_w_i_init = T_w_i.cast<Scalar>();

  last_state_t_ns = t_ns;
  imu_meas[t_ns] = IntegratedImuMeasurement<Scalar>(t_ns, bg.cast<Scalar>(), ba.cast<Scalar>());
  frame_states[t_ns] = PoseVelBiasStateWithLin<Scalar>(t_ns, T_w_i_init, vel_w_i.cast<Scalar>(), bg.cast<Scalar>(),
                                                       ba.cast<Scalar>(), true);
  frame_idx[t_ns] = frame_count++;

  marg_data.order.abs_order_map[t_ns] = std::make_pair(0, POSE_VEL_BIAS_SIZE);
  marg_data.order.total_size = POSE_VEL_BIAS_SIZE;
  marg_data.order.items = 1;

  nullspace_marg_data.order = marg_data.order;

  initialize(bg, ba);
}

template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::initialize(const Eigen::Vector3d& bg_, const Eigen::Vector3d& ba_) {
  /*
   * [VIO 초기화 함수]
   * - 자이로스코프 바이어스(bg)와 가속도계 바이어스(ba)를 초기화하는 함수
   * - 실제 VIO 처리를 담당하는 processing_thread를 생성하고 시작
   */
  Vec3 bg_init = bg_.cast<Scalar>();
  Vec3 ba_init = ba_.cast<Scalar>();

  auto proc_func = [&, bg = bg_init, ba = ba_init] {
    OpticalFlowResult::Ptr prev_frame, curr_frame;
    typename IntegratedImuMeasurement<Scalar>::Ptr meas;

    /*
     * [IMU 노이즈 모델 설정]
     * - 가속도계와 자이로스코프의 이산시간 노이즈 표준편차를 제곱하여 공분산 행렬 생성
     * - 실제 IMU 적분에 사용됨
     */
    const Vec3 accel_cov = calib.dicrete_time_accel_noise_std().array().square();
    const Vec3 gyro_cov = calib.dicrete_time_gyro_noise_std().array().square();

    typename ImuData<Scalar>::Ptr data = popFromImuDataQueue();

    bool run = data != nullptr;  // End VIO otherwise
    if (run) {
      /*
       * [IMU 데이터 보정]
       * - 원본 IMU 데이터에 캘리브레이션 적용
       * - calib_accel_bias, calib_gyro_bias는 basalt/calibration/ 디렉토리에 구현
       */
      data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
      data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
    }

    while (run) {
      bool reset_performed = schedule_reset;
      if (reset_performed) {
        bool exit_requested = resetState(meas, curr_frame, prev_frame);
        if (exit_requested) break;
      }

      vision_data_queue.pop(curr_frame);

      /*
       * [실시간 처리를 위한 프레임 드롭]
       * - 실시간성이 중요한 경우 큐에 있는 오래된 프레임을 건너뜀
       * - 최신 프레임만 처리하여 지연을 방지
       */
      if (config.vio_enforce_realtime) {
        // drop current frame if another frame is already in the queue.
        while (!vision_data_queue.empty()) vision_data_queue.pop(curr_frame);
      }

      if (!curr_frame.get()) {
        break;
      }
      curr_frame->input_images->addTime("vio_start");
      curr_frame->input_images->state_reset = reset_performed;

      // Correct camera time offset
      // curr_frame->t_ns += calib.cam_time_offset_ns;

      if (out_vis_queue) {
        visual_data = std::make_shared<VioVisualizationData>();
        visual_data->t_ns = curr_frame->t_ns;
      }

      /*
       * [VIO 시스템 초기화]
       * - 첫 프레임에서 초기 자세 추정
       * - 중력 방향을 기준으로 초기 회전 설정
       * - 상태 변수(pose, velocity, bias) 초기화
       */
      if (!initialized) {
        while (data->t_ns < curr_frame->t_ns) {
          data = popFromImuDataQueue();
          if (!data) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
          // std::cout << "Skipping IMU data.." << std::endl;
        }

        Vec3 vel_w_i_init;
        vel_w_i_init.setZero();

        T_w_i_init.translation().setZero();
        T_w_i_init.setQuaternion(Eigen::Quaternion<Scalar>::FromTwoVectors(data->accel, Vec3::UnitZ()));

        last_state_t_ns = curr_frame->t_ns;
        imu_meas[last_state_t_ns] = IntegratedImuMeasurement<Scalar>(last_state_t_ns, bg, ba);
        frame_states[last_state_t_ns] =
            PoseVelBiasStateWithLin<Scalar>(last_state_t_ns, T_w_i_init, vel_w_i_init, bg, ba, true);
        frame_idx[last_state_t_ns] = frame_count++;

        marg_data.order.abs_order_map[last_state_t_ns] = std::make_pair(0, POSE_VEL_BIAS_SIZE);
        marg_data.order.total_size = POSE_VEL_BIAS_SIZE;
        marg_data.order.items = 1;

        std::cout << "Setting up filter: t_ns " << last_state_t_ns << std::endl;
        std::cout << "T_w_i\n" << T_w_i_init.matrix() << std::endl;
        std::cout << "vel_w_i " << vel_w_i_init.transpose() << std::endl;

        if (config.vio_debug || config.vio_extended_logging) {
          logMargNullspace();
        }

        initialized = true;
      }

      /*
       * [IMU 사전적분]
       * - 이전 프레임과 현재 프레임 사이의 IMU 측정치 적분
       * - 적분된 측정치는 상태 예측과 최적화에 사용
       * - 실제 구현은 basalt/imu/ 디렉토리의 preintegration.h 참고
       */
      if (prev_frame) {
        // preintegrate measurements

        auto last_state = frame_states.at(last_state_t_ns);

        meas.reset(new IntegratedImuMeasurement<Scalar>(prev_frame->t_ns, last_state.getState().bias_gyro,
                                                        last_state.getState().bias_accel));

        BASALT_ASSERT_MSG(prev_frame->t_ns != curr_frame->t_ns,
                          "duplicate frame timestamps?! zero time delta leads "
                          "to invalid IMU integration.");

        BASALT_ASSERT_MSG(prev_frame->t_ns < curr_frame->t_ns,
                          "frame timestamps not monotonically increasing?! "
                          "are we going 88mph and inside a delorean?");

        while (data->t_ns <= prev_frame->t_ns) {
          data = popFromImuDataQueue();
          if (!data) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
        }

        while (data->t_ns <= curr_frame->t_ns) {
          meas->integrate(*data, accel_cov, gyro_cov);
          data = popFromImuDataQueue();
          if (!data) break;
          data->accel = calib.calib_accel_bias.getCalibrated(data->accel);
          data->gyro = calib.calib_gyro_bias.getCalibrated(data->gyro);
        }

        if (meas->get_start_t_ns() + meas->get_dt_ns() < curr_frame->t_ns) {
          if (!data.get()) break;
          int64_t tmp = data->t_ns;
          data->t_ns = curr_frame->t_ns;
          meas->integrate(*data, accel_cov, gyro_cov);
          data->t_ns = tmp;
        }
      }
      curr_frame->input_images->addTime("imu_preintegrated");

      /*
       * [VIO 측정 업데이트]
       * - 비주얼 트래킹과 IMU 적분 결과를 결합하여 상태 추정
       * - measure() 함수는 basalt/vi_estimator/sqrt_keypoint_vio.cpp에 구현
       */
      bool success = measure(curr_frame, meas);
      if (!success) {
        schedule_reset = true;
        if (out_state_queue) {  // If optimization fails, push an empty state to the output queue
          auto data = std::make_shared<PoseVelBiasState<double>>();
          data->t_ns = curr_frame->t_ns;
          data->input_images = curr_frame->input_images;
          data->input_images->addTime("pose_produced");
          out_state_queue->push(data);
        }
        continue;
      }

      prev_frame = curr_frame;
    }

    if (out_vis_queue) out_vis_queue->push(nullptr);
    if (out_marg_queue) out_marg_queue->push(nullptr);
    if (out_state_queue) out_state_queue->push(nullptr);

    finished = true;

    std::cout << "Finished VIOFilter " << std::endl;
  };

  processing_thread.reset(new std::thread(proc_func));
}

template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::addIMUToQueue(const ImuData<double>::Ptr& data) {
  imu_data_queue.emplace(data);
}

template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::addVisionToQueue(const OpticalFlowResult::Ptr& data) {
  vision_data_queue.push(data);
}

template <class Scalar_>
typename ImuData<Scalar_>::Ptr SqrtKeypointVioEstimator<Scalar_>::popFromImuDataQueue() {
  ImuData<double>::Ptr data;
  imu_data_queue.pop(data);

  if constexpr (std::is_same_v<Scalar, double>) {
    return data;
  } else {
    typename ImuData<Scalar>::Ptr data2;
    if (data) {
      data2.reset(new ImuData<Scalar>);
      *data2 = data->cast<Scalar>();
    }
    return data2;
  }
}

/*
 * [VIO 측정 업데이트 함수]
 * - 비주얼 프론트엔드(optical flow)와 IMU 측정치를 결합하여 상태를 업데이트
 * - 주요 구현 내용:
 *   1. IMU 적분을 통한 상태 예측 (basalt/vi_estimator/imu_integrator.h)
 *   2. 새로운 키프레임 생성 및 랜드마크 관리
 *   3. 번들 조정 최적화 (optimize_and_marg() 함수에서 수행)
 *   4. 상태 및 랜드마크 마진화
 */
template <class Scalar_>
bool SqrtKeypointVioEstimator<Scalar_>::measure(const OpticalFlowResult::Ptr& opt_flow_meas,
                                                const typename IntegratedImuMeasurement<Scalar>::Ptr& meas) {
  stats_sums_.add("frame_id", opt_flow_meas->t_ns).format("none");
  Timer t_total;

  /*
   * [IMU 예측 단계]
   * - IMU 측정치가 있는 경우 상태를 예측
   * - 이전 상태에서 IMU 적분을 통해 현재 상태 예측
   * - 실제 IMU 적분은 basalt/vi_estimator/imu_integrator.h에 구현
   */
  if (meas.get()) {
    BASALT_ASSERT(frame_states[last_state_t_ns].getState().t_ns == meas->get_start_t_ns());
    BASALT_ASSERT(opt_flow_meas->t_ns == meas->get_dt_ns() + meas->get_start_t_ns());
    BASALT_ASSERT(meas->get_dt_ns() > 0);

    PoseVelBiasState<Scalar> next_state = frame_states.at(last_state_t_ns).getState();

    meas->predictState(frame_states.at(last_state_t_ns).getState(), g, next_state);

    last_state_t_ns = opt_flow_meas->t_ns;
    next_state.t_ns = opt_flow_meas->t_ns;

    frame_states[last_state_t_ns] = PoseVelBiasStateWithLin<Scalar>(next_state);
    frame_idx[last_state_t_ns] = frame_count++;

    imu_meas[meas->get_start_t_ns()] = *meas;
  }

  // save results
  prev_opt_flow_res[opt_flow_meas->t_ns] = opt_flow_meas;

  /*
   * [랜드마크 관리 및 데이터 연관]
   * - 기존 랜드마크와 새로운 관측치 매칭
   * - 랜드마크 데이터베이스는 basalt/vi_estimator/landmark_database.h에 구현
   */
  int NUM_CAMS = opt_flow_meas->keypoints.size();
  std::vector<int> connected(NUM_CAMS, 0);
  std::map<int64_t, int> num_points_connected;
  std::vector<std::unordered_set<int>> unconnected_obs(NUM_CAMS);
  for (int i = 0; i < NUM_CAMS; i++) {
    TimeCamId tcid_target(opt_flow_meas->t_ns, i);

    for (const auto& kv_obs : opt_flow_meas->keypoints[i]) {
      int kpt_id = kv_obs.first;

      if (lmdb.landmarkExists(kpt_id)) {
        const TimeCamId& tcid_host = lmdb.getLandmark(kpt_id).host_kf_id;

        KeypointObservation<Scalar> kobs;
        kobs.kpt_id = kpt_id;
        kobs.pos = kv_obs.second.translation().cast<Scalar>();

        lmdb.addObservation(tcid_target, kobs);
        // obs[tcid_host][tcid_target].push_back(kobs);

        if (num_points_connected.count(tcid_host.frame_id) == 0) {
          num_points_connected[tcid_host.frame_id] = 0;
        }
        num_points_connected[tcid_host.frame_id]++;

        connected[i]++;
      } else {
        unconnected_obs[i].emplace(kpt_id);
      }
    }
  }

  /*
   * [키프레임 결정]
   * - 트래킹된 특징점 비율이 임계값보다 낮으면 새로운 키프레임 생성
   * - 키프레임 관리는 basalt/vi_estimator/keyframe_database.h에서 구현
   */
  if (Scalar(connected[0]) / (connected[0] + unconnected_obs[0].size()) < Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;

  if (config.vio_debug) {
    for (int i = 0; i < NUM_CAMS; i++) {
      std::cout << "connected[" << i << "] = " << connected[i] << ", "
                << "unconnected[" << i << "] =" << unconnected_obs[i].size() << std::endl;
    }
  }

  if (take_ltkf) {
    if (!kf_ids.empty()) {  // Move newest kf to ltkfs
      auto last_kf_it = std::prev(kf_ids.end());
      ltkfs.emplace(*last_kf_it);
      kf_ids.erase(last_kf_it);
    }
    take_ltkf = false;
  }

  /*
   * [새로운 랜드마크 생성]
   * - 키프레임에서 새로운 랜드마크 삼각측량
   * - 삼각측량은 basalt/utils/triangulation.h에 구현
   */
  if (take_kf) {
    // Triangulate new points from one of the observations (with sufficient
    // baseline) and make keyframe
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);
    if (visual_data) visual_data->keyframed_idx[last_state_t_ns] = frame_idx.at(last_state_t_ns);

    int num_points_added = 0;
    for (int i = 0; i < NUM_CAMS; i++) {
      TimeCamId tcidl(opt_flow_meas->t_ns, i);

      for (int lm_id : unconnected_obs[i]) {
        if (lmdb.landmarkExists(lm_id)) continue;
        // Find all observations
        std::map<TimeCamId, KeypointObservation<Scalar>> kp_obs;

        for (const auto& kv : prev_opt_flow_res) {
          for (size_t k = 0; k < kv.second->keypoints.size(); k++) {
            auto it = kv.second->keypoints[k].find(lm_id);
            if (it != kv.second->keypoints[k].end()) {
              TimeCamId tcido(kv.first, k);

              KeypointObservation<Scalar> kobs;
              kobs.kpt_id = lm_id;
              kobs.pos = it->second.translation().template cast<Scalar>();

              // obs[tcidl][tcido].push_back(kobs);
              kp_obs[tcido] = kobs;
            }
          }
        }

        // triangulate
        bool valid_kp = false;
        const Scalar min_triang_distance2 =
            Scalar(config.vio_min_triangulation_dist * config.vio_min_triangulation_dist);
        for (const auto& kv_obs : kp_obs) {
          if (valid_kp) break;
          TimeCamId tcido = kv_obs.first;

          const Vec2 p0 = opt_flow_meas->keypoints.at(i).at(lm_id).translation().cast<Scalar>();
          const Vec2 p1 = prev_opt_flow_res[tcido.frame_id]
                              ->keypoints[tcido.cam_id]
                              .at(lm_id)
                              .translation()
                              .template cast<Scalar>();

          Vec4 p0_3d, p1_3d;
          bool valid1 = calib.intrinsics[i].unproject(p0, p0_3d);
          bool valid2 = calib.intrinsics[tcido.cam_id].unproject(p1, p1_3d);
          if (!valid1 || !valid2) continue;

          SE3 T_i0_i1 =
              getPoseStateWithLin(tcidl.frame_id).getPose().inverse() * getPoseStateWithLin(tcido.frame_id).getPose();
          SE3 T_0_1 = calib.T_i_c[i].inverse() * T_i0_i1 * calib.T_i_c[tcido.cam_id];

          if (T_0_1.translation().squaredNorm() < min_triang_distance2) continue;

          Vec4 p0_triangulated = triangulate(p0_3d.template head<3>(), p1_3d.template head<3>(), T_0_1);

          if (p0_triangulated.array().isFinite().all() && p0_triangulated[3] > 0 && p0_triangulated[3] < 3.0) {
            Landmark<Scalar> lm_pos;
            lm_pos.host_kf_id = tcidl;
            lm_pos.direction = StereographicParam<Scalar>::project(p0_triangulated);
            lm_pos.inv_dist = p0_triangulated[3];
            lmdb.addLandmark(lm_id, lm_pos);

            num_points_added++;
            valid_kp = true;
          }
        }

        if (valid_kp) {
          for (const auto& kv_obs : kp_obs) {
            lmdb.addObservation(kv_obs.first, kv_obs.second);
          }
        }
      }
    }

    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }

  /*
   * [최적화 및 마진화]
   * - 번들 조정을 통한 상태 최적화
   * - 오래된 프레임과 랜드마크 마진화
   * - 최적화는 basalt/optimization/bundle_adjustment.h에 구현
   */
  std::unordered_set<KeypointId> lost_landmaks;
  if (config.vio_marg_lost_landmarks) {
    for (const auto& kv : lmdb.getLandmarks()) {
      bool connected = false;
      for (size_t i = 0; i < opt_flow_meas->keypoints.size(); i++) {
        if (opt_flow_meas->keypoints[i].count(kv.first) > 0) connected = true;
      }
      if (!connected) {
        lost_landmaks.emplace(kv.first);
      }
    }
  }
  opt_flow_meas->input_images->addTime("landmarks_updated");

  bool success = optimize_and_marg(opt_flow_meas->input_images, num_points_connected, lost_landmaks);
  if (!success) return false;

  /*
   * [시각화 및 결과 출력]
   * - 특징점 투영 및 시각화 데이터 생성
   * - 시각화는 basalt/utils/vis_utils.h에 구현
   */
  size_t num_cams = opt_flow_meas->keypoints.size();
  bool features_ext = opt_flow_meas->input_images->stats.enabled_exts.has_pose_features;
  bool avg_depth_needed =
      opt_flow_depth_guess_queue && config.optical_flow_matching_guess_type == MatchingGuessType::REPROJ_AVG_DEPTH;

  using Projections = std::vector<Eigen::aligned_vector<Eigen::Vector4d>>;
  std::shared_ptr<Projections> projections = nullptr;
  if (features_ext || out_vis_queue || avg_depth_needed) {
    projections = std::make_shared<Projections>(num_cams);
    computeProjections(*projections, last_state_t_ns);
  }

  if (out_state_queue) {
    PoseVelBiasStateWithLin p = frame_states.at(last_state_t_ns);

    typename PoseVelBiasState<double>::Ptr data(new PoseVelBiasState<double>(p.getState().template cast<double>()));

    data->input_images = opt_flow_meas->input_images;
    data->input_images->addTime("pose_produced");

    if (avg_depth_needed) {
      double avg_invdepth = 0;
      double num_features = 0;
      for (const auto& cam_projs : *projections) {
        for (const Eigen::Vector4d& v : cam_projs) avg_invdepth += v.z();
        num_features += cam_projs.size();
      }

      bool valid = avg_invdepth > 0 && num_features > 0;
      float default_depth = config.optical_flow_matching_default_depth;
      double avg_depth = valid ? num_features / avg_invdepth : default_depth;

      if (opt_flow_depth_guess_queue) opt_flow_depth_guess_queue->push(avg_depth);
    }

    if (features_ext) {
      for (size_t i = 0; i < num_cams; i++) {
        for (const Eigen::Vector4d& v : projections->at(i)) {
          vit::PoseFeature lm = {};
          lm.id = v.w();
          lm.u = v.x();
          lm.v = v.y();
          lm.depth = v.z();
          data->input_images->stats.addFeature(i, lm);
        }
      }
    }
    if (out_state_queue) out_state_queue->push(data);
    if (opt_flow_state_queue) opt_flow_state_queue->push(data);
  }

  bool lm_bundle_needed = opt_flow_lm_bundle_queue && config.optical_flow_recall_enable;
  if (lm_bundle_needed) {
    LandmarkBundle::Ptr lmb = std::make_shared<LandmarkBundle>();
    lmb->ts = last_state_t_ns;
    for (const auto& [lmid, lm] : lmdb.getLandmarks()) {
      if (frame_poses.count(lm.host_kf_id.frame_id) == 0) continue;
      SE3 T_w_i = frame_poses.at(lm.host_kf_id.frame_id).getPose();
      SE3 T_i_c = calib.T_i_c[lm.host_kf_id.cam_id];
      SE3 T_w_c = T_w_i * T_i_c;

      Vec4 pt_c = StereographicParam<Scalar>::unproject(lm.direction);
      pt_c *= 1 / lm.inv_dist;  // scale by depth
      pt_c[3] = 1;

      Vec4 pt_w = T_w_c * pt_c;
      lmb->lmids.emplace_back(lmid);
      lmb->lms.emplace_back(pt_w.template cast<float>());
    }
    opt_flow_lm_bundle_queue->push(lmb);
  }

  if (out_vis_queue) {
    for (const auto& [ts, p] : frame_states) visual_data->states[ts] = p.getState().T_w_i.template cast<double>();

    for (const auto& [ts, pstate] : frame_poses) {
      auto& frames = kf_ids.count(ts) ? visual_data->frames : visual_data->ltframes;
      frames[ts] = pstate.getPose().template cast<double>();
    }

    visual_data->frame_idx = frame_idx;

    get_current_points(visual_data->points, visual_data->point_ids);

    visual_data->projections = projections;

    visual_data->opt_flow_res = prev_opt_flow_res[last_state_t_ns];

    out_vis_queue->push(visual_data);
  }

  last_processed_t_ns = last_state_t_ns;

  stats_sums_.add("measure", t_total.elapsed()).format("ms");

  return true;
}

template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::logMargNullspace() {
  nullspace_marg_data.order = marg_data.order;
  if (config.vio_debug) {
    std::cout << "======== Marg nullspace ==========" << std::endl;
    stats_sums_.add("marg_ns", checkMargNullspace());
    std::cout << "=================================" << std::endl;
  } else {
    stats_sums_.add("marg_ns", checkMargNullspace());
  }
  stats_sums_.add("marg_ev", checkMargEigenvalues());
}

template <class Scalar_>
Eigen::VectorXd SqrtKeypointVioEstimator<Scalar_>::checkMargNullspace() const {
  return checkNullspace(nullspace_marg_data, frame_states, frame_poses, config.vio_debug);
}

template <class Scalar_>
Eigen::VectorXd SqrtKeypointVioEstimator<Scalar_>::checkMargEigenvalues() const {
  return checkEigenvalues(nullspace_marg_data, false);
}

template <class Scalar>
bool SqrtKeypointVioEstimator<Scalar>::show_uimat(UIMAT m) const {
  UIMAT showed = prev_opt_flow_res.at(last_state_t_ns)->input_images->show_uimat;

  bool show_none = showed == UIMAT::NONE;
  if (show_none) return false;

  bool ui_enabled = out_vis_queue != nullptr;
  bool show_all = showed == UIMAT::ALL;
  bool show_this = showed == m;
  bool res = ui_enabled && (show_all || show_this);
  return res;
}

// 이 함수는 VIO 시스템의 marginalization을 수행하는 핵심 함수입니다.
// 주요 기능:
// 1. 오래된 keyframe과 state를 제거하여 시스템 크기를 일정하게 유지
// 2. marginalization prior 생성하여 제거된 정보를 보존
template <class Scalar_>
bool SqrtKeypointVioEstimator<Scalar_>::marginalize(const std::map<int64_t, int>& num_points_connected,
                                                    const std::unordered_set<KeypointId>& lost_landmaks) {
  if (!opt_started) return true;

  Timer t_total;

  // 시스템 크기가 max_kfs나 max_states를 초과하면 marginalization 수행
  if (frame_poses.size() > ltkfs.size() + max_kfs || frame_states.size() >= max_states) {
    // Marginalize

    const int states_to_remove = frame_states.size() - max_states + 1;

    auto it = frame_states.cbegin();
    for (int i = 0; i < states_to_remove; i++) it++;
    int64_t last_state_to_marg = it->first;

    // 상태 변수들의 순서를 관리하는 AbsOrderMap 생성
    AbsOrderMap aom;

    // keyframe이 아닌 모든 frame_poses를 제거 대상으로 표시
    std::set<int64_t> poses_to_marg;
    for (const auto& kv : frame_poses) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

      if (kf_ids.count(kv.first) == 0 && ltkfs.count(kv.first) == 0) poses_to_marg.emplace(kv.first);

      // Check that we have the same order as marginalization
      BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_SIZE;
      aom.items++;
    }

    // 제거할 state들을 velocity/bias만 제거할 것과 전체를 제거할 것으로 분류
    std::set<int64_t> states_to_marg_vel_bias;
    std::set<int64_t> states_to_marg_all;
    for (const auto& kv : frame_states) {
      if (kv.first > last_state_to_marg) break;

      if (kv.first != last_state_to_marg) {
        if (kf_ids.count(kv.first) > 0 || ltkfs.count(kv.first) > 0) {
          states_to_marg_vel_bias.emplace(kv.first);
        } else {
          states_to_marg_all.emplace(kv.first);
        }
      }

      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

      // Check that we have the same order as marginalization
      if (aom.items < marg_data.order.abs_order_map.size())
        BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_VEL_BIAS_SIZE;
      aom.items++;
    }

    auto kf_ids_all = kf_ids;
    std::set<int64_t> kfs_to_marg;
    
    // 제거할 keyframe 선택 로직
    // config.vio_kf_marg_criteria에 따라 두 가지 방식 중 하나 사용:
    // 1. KF_MARG_FORWARD_VECTOR: forward vector 방향이 잘 분포되도록 선택
    // 2. KF_MARG_DEFAULT: feature tracking 상태와 keyframe 간 거리 기반으로 선택
    while (kf_ids.size() > max_kfs && !states_to_marg_vel_bias.empty()) {
      int64_t id_to_marg = -1;

      if (config.vio_kf_marg_criteria == KeyframeMargCriteria::KF_MARG_FORWARD_VECTOR) {
        // forward vector 기반 keyframe 선택 로직 구현부
        if (kf_ids.size() > 2 && id_to_marg < 0) {
          std::set<int64_t> all_kfs = ltkfs;
          all_kfs.insert(kf_ids.begin(), kf_ids.end());

          // Do not account for the two most recent frames
          auto last1 = std::prev(kf_ids.end(), 2);
          auto last2 = std::prev(all_kfs.end(), 2);
          Scalar min_score = std::numeric_limits<Scalar>::max();
          int64_t min_score_id = -1;

          auto get_forward_vector2d = [&](int64_t ts) -> Vec2 {
            SE3 T_w_i = frame_poses.at(ts).getPose();
            SE3 T_w_c0 = T_w_i * calib.T_i_c[0];
            Vec3 fwd3d = T_w_c0.so3() * Vec3{0, 0, 1};
            Vec2 fwd2d = fwd3d.template head<2>();
            return fwd2d;
          };

          for (auto it1 = kf_ids.begin(); it1 != last1; ++it1) {
            Vec2 fwd1 = get_forward_vector2d(*it1);
            Scalar score = 0;
            for (auto it2 = all_kfs.begin(); it2 != last2; ++it2) {
              Vec2 fwd2 = get_forward_vector2d(*it2);
              Scalar dot = std::clamp(fwd1.dot(fwd2), Scalar(-1), Scalar(1));  // clamp needed, otherwise acos can fail
              Scalar angle = acos(dot);
              score += angle;
            }

            if (score < min_score) {
              min_score_id = *it1;
              min_score = score;
            }
          }

          id_to_marg = min_score_id;
        }
      } else if (config.vio_kf_marg_criteria == KeyframeMargCriteria::KF_MARG_DEFAULT) {
        // feature tracking 상태와 거리 기반 keyframe 선택 로직 구현부
        if (kf_ids.size() > 2) {
          // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
          auto end_minus_2 = std::prev(kf_ids.end(), 2);

          for (auto it = kf_ids.begin(); it != end_minus_2; ++it) {
            if (num_points_connected.count(*it) == 0 ||
                (num_points_connected.at(*it) / static_cast<float>(num_points_kf.at(*it)) <
                 config.vio_kf_marg_feature_ratio)) {
              id_to_marg = *it;
              break;
            }
          }
        }

        // DSO에서 가져온 score 함수 - 주로 가장 오래된 keyframe을 제거
        if (kf_ids.size() > 2 && id_to_marg < 0) {
          // Note: size > 2 check is to ensure prev(kf_ids.end(), 2) is valid
          auto end_minus_2 = std::prev(kf_ids.end(), 2);

          int64_t last_kf = *kf_ids.crbegin();
          Scalar min_score = std::numeric_limits<Scalar>::max();
          int64_t min_score_id = -1;

          for (auto it1 = kf_ids.begin(); it1 != end_minus_2; ++it1) {
            // small distance to other keyframes --> higher score
            Scalar denom = 0;
            for (auto it2 = kf_ids.begin(); it2 != end_minus_2; ++it2) {
              denom +=
                  1 / ((frame_poses.at(*it1).getPose().translation() - frame_poses.at(*it2).getPose().translation())
                           .norm() +
                       Scalar(1e-5));
            }

            // small distance to latest kf --> lower score
            Scalar score = std::sqrt((frame_poses.at(*it1).getPose().translation() -
                                      frame_states.at(last_kf).getState().T_w_i.translation())
                                         .norm()) *
                           denom;

            if (score < min_score) {
              min_score_id = *it1;
              min_score = score;
            }
          }

          id_to_marg = min_score_id;
        }
      } else {
        BASALT_LOG_FATAL("Unexpected marginalization criteria");
      }

      // if no frame was selected, the logic above is faulty
      BASALT_ASSERT(id_to_marg >= 0);

      kfs_to_marg.emplace(id_to_marg);
      poses_to_marg.emplace(id_to_marg);

      kf_ids.erase(id_to_marg);
    }

    //    std::cout << "marg order" << std::endl;
    //    aom.print_order();

    //    std::cout << "marg prior order" << std::endl;
    //    marg_order.print_order();

    if (config.vio_debug) {
      std::cout << "states_to_remove " << states_to_remove << std::endl;
      std::cout << "poses_to_marg.size() " << poses_to_marg.size() << std::endl;
      std::cout << "states_to_marg.size() " << states_to_marg_all.size() << std::endl;
      std::cout << "state_to_marg_vel_bias.size() " << states_to_marg_vel_bias.size() << std::endl;
      std::cout << "kfs_to_marg.size() " << kfs_to_marg.size() << std::endl;
    }

    Timer t_actual_marg;

    size_t asize = aom.total_size;

    bool is_lin_sqrt = isLinearizationSqrt(config.vio_linearization_type);

    // 선형화된 시스템의 Jacobian과 residual을 저장할 행렬/벡터
    MatX Q2Jp_or_H;
    VecX Q2r_or_b;

    {
      Timer t_linearize;

      // 선형화 옵션 설정
      typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
      lqr_options.lb_options.huber_parameter = huber_thresh;
      lqr_options.lb_options.obs_std_dev = obs_std_dev;
      lqr_options.linearization_type = config.vio_linearization_type;

      // IMU 데이터 준비
      ImuLinData<Scalar> ild = {g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};

      for (const auto& kv : imu_meas) {
        int64_t start_t = kv.second.get_start_t_ns();
        int64_t end_t = kv.second.get_start_t_ns() + kv.second.get_dt_ns();

        if (aom.abs_order_map.count(start_t) == 0 || aom.abs_order_map.count(end_t) == 0) continue;

        ild.imu_meas[kv.first] = &kv.second;
      }

      // 시스템 선형화 및 QR 분해 수행
      std::set<FrameId> fixed_kfs = config.vio_fix_long_term_keyframes ? ltkfs : std::set<FrameId>{};
      auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(this, aom, lqr_options, &marg_data, &ild, &kfs_to_marg,
                                                              &lost_landmaks, last_state_to_marg, &fixed_kfs);

      lqr->linearizeProblem();

      if (show_uimat(UIMAT::JR_M)) visual_data->getj(UIMAT::JR_M).Jr = lqr->getUILandmarkBlocks();

      lqr->performQR();
      if (show_uimat(UIMAT::JR_M_QR)) visual_data->getj(UIMAT::JR_M_QR).Jr = lqr->getUILandmarkBlocks();

      if (is_lin_sqrt && marg_data.is_sqrt) {
        lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
      } else {
        lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);
      }

      stats_sums_.add("marg_linearize", t_linearize.elapsed()).format("ms");
    }

    //    KeypointVioEstimator::linearizeAbsIMU(
    //        aom, accum.getH(), accum.getB(), imu_error, bg_error, ba_error,
    //        frame_states, imu_meas, gyro_bias_weight, accel_bias_weight, g);
    //    linearizeMargPrior(marg_order, marg_sqrt_H, marg_sqrt_b, aom,
    //    accum.getH(),
    //                       accum.getB(), marg_prior_error);

    // marginalization prior 저장 - 다른 모듈에서 사용할 수 있도록
    if (out_marg_queue && !kfs_to_marg.empty()) {
      // int64_t kf_id = *kfs_to_marg.begin();

      {
        MargData::Ptr m(new MargData);
        m->aom = aom;

        if (is_lin_sqrt && marg_data.is_sqrt) {
          m->abs_H = (Q2Jp_or_H.transpose() * Q2Jp_or_H).template cast<double>();
          m->abs_b = (Q2Jp_or_H.transpose() * Q2r_or_b).template cast<double>();
        } else {
          m->abs_H = Q2Jp_or_H.template cast<double>();

          m->abs_b = Q2r_or_b.template cast<double>();
        }

        assign_cast_map_values(m->frame_poses, frame_poses);
        assign_cast_map_values(m->frame_states, frame_states);
        m->kfs_all = kf_ids_all;
        m->kfs_to_marg = kfs_to_marg;
        m->use_imu = true;

        for (int64_t t : m->kfs_all) {
          m->opt_flow_res.emplace_back(prev_opt_flow_res.at(t));
        }

        out_marg_queue->push(m);
      }
    }

    // 유지할 변수와 제거할 변수의 인덱스 구분
    std::set<int> idx_to_keep, idx_to_marg;
    for (const auto& kv : aom.abs_order_map) {
      if (kv.second.second == POSE_SIZE) {
        int start_idx = kv.second.first;
        if (poses_to_marg.count(kv.first) == 0) {
          for (size_t i = 0; i < POSE_SIZE; i++) idx_to_keep.emplace(start_idx + i);
        } else {
          for (size_t i = 0; i < POSE_SIZE; i++) idx_to_marg.emplace(start_idx + i);
        }
      } else {
        BASALT_ASSERT(kv.second.second == POSE_VEL_BIAS_SIZE);
        // state
        int start_idx = kv.second.first;
        if (states_to_marg_all.count(kv.first) > 0) {
          for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++) idx_to_marg.emplace(start_idx + i);
        } else if (states_to_marg_vel_bias.count(kv.first) > 0) {
          for (size_t i = 0; i < POSE_SIZE; i++) idx_to_keep.emplace(start_idx + i);
          for (size_t i = POSE_SIZE; i < POSE_VEL_BIAS_SIZE; i++) idx_to_marg.emplace(start_idx + i);
        } else {
          BASALT_ASSERT(kv.first == last_state_to_marg);
          for (size_t i = 0; i < POSE_VEL_BIAS_SIZE; i++) idx_to_keep.emplace(start_idx + i);
        }
      }
    }

    if (config.vio_debug) {
      std::cout << "keeping " << idx_to_keep.size() << " marg " << idx_to_marg.size() << " total " << asize
                << std::endl;
      std::cout << "last_state_to_marg " << last_state_to_marg << " frame_poses " << frame_poses.size()
                << " frame_states " << frame_states.size() << std::endl;
    }

    // nullspace 체크를 위한 추가 계산 (디버깅용)
    if (config.vio_debug || config.vio_extended_logging) {
      MatX Q2Jp_or_H_nullspace;
      VecX Q2r_or_b_nullspace;

      typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
      lqr_options.lb_options.huber_parameter = huber_thresh;
      lqr_options.lb_options.obs_std_dev = obs_std_dev;
      lqr_options.linearization_type = config.vio_linearization_type;

      nullspace_marg_data.order = marg_data.order;

      ImuLinData<Scalar> ild = {g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};

      for (const auto& kv : imu_meas) {
        int64_t start_t = kv.second.get_start_t_ns();
        int64_t end_t = kv.second.get_start_t_ns() + kv.second.get_dt_ns();

        if (aom.abs_order_map.count(start_t) == 0 || aom.abs_order_map.count(end_t) == 0) continue;

        ild.imu_meas[kv.first] = &kv.second;
      }

      std::set<FrameId> fixed_kfs = config.vio_fix_long_term_keyframes ? ltkfs : std::set<FrameId>{};
      auto lqr =
          LinearizationBase<Scalar, POSE_SIZE>::create(this, aom, lqr_options, &nullspace_marg_data, &ild, &kfs_to_marg,
                                                       &lost_landmaks, last_state_to_marg, &fixed_kfs);

      lqr->linearizeProblem();
      lqr->performQR();

      if (is_lin_sqrt && marg_data.is_sqrt) {
        lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
      } else {
        lqr->get_dense_H_b(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace);
      }

      MatX nullspace_sqrt_H_new;
      VecX nullspace_sqrt_b_new;

      if (is_lin_sqrt && marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep,
                                                        idx_to_marg, nullspace_sqrt_H_new, nullspace_sqrt_b_new);
      } else if (marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqToSqrt(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
                                                      nullspace_sqrt_H_new, nullspace_sqrt_b_new);
      } else {
        MargHelper<Scalar>::marginalizeHelperSqToSq(Q2Jp_or_H_nullspace, Q2r_or_b_nullspace, idx_to_keep, idx_to_marg,
                                                    nullspace_sqrt_H_new, nullspace_sqrt_b_new);
      }

      nullspace_marg_data.H = nullspace_sqrt_H_new;
      nullspace_marg_data.b = nullspace_sqrt_b_new;
    }

    // 실제 marginalization 수행
    MatX marg_H_new;
    VecX marg_b_new;

    {
      Timer t;
      if (is_lin_sqrt && marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
                                                        marg_b_new);
      } else if (marg_data.is_sqrt) {
        MargHelper<Scalar>::marginalizeHelperSqToSqrt(Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
                                                      marg_b_new);
      } else {
        MargHelper<Scalar>::marginalizeHelperSqToSq(Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
                                                    marg_b_new);
      }

      stats_sums_.add("marg_helper", t.elapsed()).format("ms");
    }

    {
      BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() == false);
      frame_states.at(last_state_to_marg).setLinTrue();
    }

    // 제거된 state들 정리
    for (const int64_t id : states_to_marg_all) {
      if (visual_data) visual_data->marginalized_idx[id] = frame_idx.at(id);
      frame_states.erase(id);
      frame_idx.erase(id);
      imu_meas.erase(id);
      prev_opt_flow_res.erase(id);
    }

    for (const int64_t id : states_to_marg_vel_bias) {
      const PoseVelBiasStateWithLin<Scalar>& state = frame_states.at(id);
      PoseStateWithLin<Scalar> pose(state);

      frame_poses[id] = pose;
      frame_states.erase(id);
      imu_meas.erase(id);
    }

    for (const int64_t id : poses_to_marg) {
      if (visual_data) visual_data->marginalized_idx[id] = frame_idx.at(id);
      frame_poses.erase(id);
      frame_idx.erase(id);
      prev_opt_flow_res.erase(id);
    }

    // landmark database 업데이트
    lmdb.removeKeyframes(kfs_to_marg, poses_to_marg, states_to_marg_all);

    if (config.vio_marg_lost_landmarks) {
      for (const auto& lm_id : lost_landmaks) lmdb.removeLandmark(lm_id);
    }

    // 새로운 marginalization prior 생성
    AbsOrderMap marg_order_new;

    for (const auto& kv : frame_poses) {
      marg_order_new.abs_order_map[kv.first] = std::make_pair(marg_order_new.total_size, POSE_SIZE);

      marg_order_new.total_size += POSE_SIZE;
      marg_order_new.items++;
    }

    {
      marg_order_new.abs_order_map[last_state_to_marg] = std::make_pair(marg_order_new.total_size, POSE_VEL_BIAS_SIZE);
      marg_order_new.total_size += POSE_VEL_BIAS_SIZE;
      marg_order_new.items++;
    }

    marg_data.H = marg_H_new;
    marg_data.b = marg_b_new;
    marg_data.order = marg_order_new;

    if (show_uimat(UIMAT::HB_M)) {
      visual_data->geth(UIMAT::HB_M).H = std::make_shared<Eigen::MatrixXf>(marg_H_new.template cast<float>());
      visual_data->geth(UIMAT::HB_M).b = std::make_shared<Eigen::VectorXf>(marg_b_new.template cast<float>());
      visual_data->geth(UIMAT::HB_M).aom = std::make_shared<AbsOrderMap>(marg_order_new);
    }

    BASALT_ASSERT(size_t(marg_data.H.cols()) == marg_data.order.total_size);

    // Quadratic prior and "delta" of the current state to the original
    // linearization point give cost function
    //
    //    P(x) = 0.5 || J*(delta+x) + r ||^2.
    //
    // For marginalization this has been linearized at x=0 to give
    // linearization
    //
    //    P(x) = 0.5 || J*x + (J*delta + r) ||^2,
    //
    // with Jacobian J and residual J*delta + r.
    //
    // After marginalization, we recover the original form of the
    // prior. We are left with linearization (in sqrt form)
    //
    //    Pnew(x) = 0.5 || Jnew*x + res ||^2.
    //
    // To recover the original form with delta-independent r, we set
    //
    //    Pnew(x) = 0.5 || Jnew*(delta+x) + (res - Jnew*delta) ||^2,
    //
    // and thus rnew = (res - Jnew*delta).

    VecX delta;
    computeDelta(marg_data.order, delta);
    marg_data.b -= marg_data.H * delta;

    if (config.vio_debug || config.vio_extended_logging) {
      VecX delta;
      computeDelta(marg_data.order, delta);
      nullspace_marg_data.b -= nullspace_marg_data.H * delta;
    }

    stats_sums_.add("marg", t_actual_marg.elapsed()).format("ms");

    if (config.vio_debug) {
      std::cout << "marginalizaon done!!" << std::endl;
    }

    if (config.vio_debug || config.vio_extended_logging) {
      Timer t;
      logMargNullspace();
      stats_sums_.add("marg_log", t.elapsed()).format("ms");
    }

    //    std::cout << "new marg prior order" << std::endl;
    //    marg_order.print_order();
  }

  stats_sums_.add("marginalize", t_total.elapsed()).format("ms");

  return true;
}

template <class Scalar_>
bool SqrtKeypointVioEstimator<Scalar_>::optimize() {
  if (config.vio_debug) {
    std::cout << "=================================" << std::endl;
  }

  /*
   * [VIO 최적화 함수]
   * - 비주얼-관성 최적화를 수행하는 핵심 함수
   * - Levenberg-Marquardt 알고리즘을 사용한 비선형 최적화 구현
   * - 주요 구성:
   *   1. 상태 변수 순서 결정 (AbsOrderMap)
   *   2. 잔차 선형화 (LinearizationBase)
   *   3. QR 분해를 통한 랜드마크 marginalization
   *   4. Schur complement를 이용한 카메라 포즈 최적화
   */

  if (opt_started || frame_states.size() > 4) {
    opt_started = true;

    // harcoded configs
    // bool scale_Jp = config.vio_scale_jacobian && is_qr_solver();
    // bool scale_Jl = config.vio_scale_jacobian && is_qr_solver();

    // timing
    ExecutionStats stats;
    Timer timer_total;
    Timer timer_iteration;

    /*
     * [상태 변수 순서 결정]
     * - 타임스탬프 기준 오름차순으로 정렬
     * - frame_poses: 카메라 포즈만 있는 프레임 (POSE_SIZE = 6)
     * - frame_states: 포즈, 속도, 바이어스가 있는 프레임 (POSE_VEL_BIAS_SIZE = 15)
     */
    AbsOrderMap aom;

    for (const auto& kv : frame_poses) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_SIZE);

      // Check that we have the same order as marginalization
      BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_SIZE;
      aom.items++;
    }

    for (const auto& kv : frame_states) {
      aom.abs_order_map[kv.first] = std::make_pair(aom.total_size, POSE_VEL_BIAS_SIZE);

      // Check that we have the same order as marginalization
      if (aom.items < marg_data.order.abs_order_map.size())
        BASALT_ASSERT(marg_data.order.abs_order_map.at(kv.first) == aom.abs_order_map.at(kv.first));

      aom.total_size += POSE_VEL_BIAS_SIZE;
      aom.items++;
    }

    // TODO: Check why we get better accuracy with old SC loop. Possible
    // culprits:
    // - different initial lambda (based on previous iteration)
    // - no landmark damping
    // - outlier removal after 4 iterations?
    lambda = Scalar(config.vio_lm_lambda_initial);

    // record stats
    stats.add("num_cams", this->frame_poses.size()).format("count");
    stats.add("num_lms", this->lmdb.numLandmarks()).format("count");
    stats.add("num_obs", this->lmdb.numObservations()).format("count");

    /*
     * [랜드마크 블록 설정]
     * - 랜드마크 관련 잔차 선형화를 위한 설정
     * - Huber robust cost function 파라미터 설정
     * - 관측 표준편차 설정
     */
    typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
    lqr_options.lb_options.huber_parameter = huber_thresh;
    lqr_options.lb_options.obs_std_dev = obs_std_dev;
    lqr_options.linearization_type = config.vio_linearization_type;

    std::unique_ptr<LinearizationBase<Scalar, POSE_SIZE>> lqr;

    /*
     * [IMU 데이터 설정]
     * - 중력, 자이로/가속도계 바이어스 가중치 설정
     * - IMU 측정치 저장
     */
    ImuLinData<Scalar> ild = {g, gyro_bias_sqrt_weight, accel_bias_sqrt_weight, {}};
    for (const auto& kv : imu_meas) {
      ild.imu_meas[kv.first] = &kv.second;
    }

    std::set<FrameId> fixed_kfs = config.vio_fix_long_term_keyframes ? ltkfs : std::set<FrameId>{};

    {
      Timer t;
      lqr = LinearizationBase<Scalar, POSE_SIZE>::create(this, aom, lqr_options, &marg_data, &ild, nullptr, nullptr,
                                                         std::numeric_limits<int64_t>::max(), &fixed_kfs);
      stats.add("allocateLMB", t.reset()).format("ms");
      lqr->log_problem_stats(stats);
    }

    bool terminated = false;
    bool converged = false;
    std::string message;

    /*
     * [Levenberg-Marquardt 최적화 반복]
     * - 최대 반복 횟수까지 최적화 수행
     * - 각 반복에서:
     *   1. 잔차 선형화
     *   2. QR 분해로 랜드마크 marginalization
     *   3. Schur complement로 카메라 포즈 업데이트
     *   4. 댐핑 파라미터(lambda) 조정
     */
    int it = 0;
    int it_rejected = 0;
    for (; it <= config.vio_max_iterations && !terminated;) {
      if (it > 0) {
        timer_iteration.reset();
      }

      Scalar error_total = 0;
      VecX Jp_column_norm2;

      {
        // TODO: execution could be done staged

        Timer t;

        // linearize residuals
        bool numerically_valid;
        error_total = lqr->linearizeProblem(&numerically_valid);
        if (show_uimat(UIMAT::JR)) visual_data->getj(UIMAT::JR).Jr = lqr->getUILandmarkBlocks();

        if (!numerically_valid) {
          std::cerr << "did not expect numerical failure during linearization\n";
          return false;
        }

        stats.add("linearizeProblem", t.reset()).format("ms");

        //        // compute pose jacobian norm squared for Jacobian scaling
        //        if (scale_Jp) {
        //          Jp_column_norm2 = lqr->getJp_diag2();
        //          stats.add("getJp_diag2", t.reset()).format("ms");
        //        }

        //        // scale landmark jacobians
        //        if (scale_Jl) {
        //          lqr->scaleJl_cols();
        //          stats.add("scaleJl_cols", t.reset()).format("ms");
        //        }

        // marginalize points in place
        lqr->performQR();
        if (show_uimat(UIMAT::JR_QR)) visual_data->getj(UIMAT::JR_QR).Jr = lqr->getUILandmarkBlocks();

        stats.add("performQR", t.reset()).format("ms");
      }

      if (config.vio_debug) {
        // TODO: num_points debug output missing
        std::cout << "[LINEARIZE] Error: " << error_total << " num points " << std::endl;
        std::cout << "Iteration " << it << " " << error_total << std::endl;
      }

      // compute pose jacobian scaling
      //      VecX jacobian_scaling;
      //      if (scale_Jp) {
      //        // TODO: what about jacobian scaling for SC solver?

      //        // ceres uses 1.0 / (1.0 + sqrt(SquaredColumnNorm))
      //        // we use 1.0 / (eps + sqrt(SquaredColumnNorm))
      //        jacobian_scaling = (lqr_options.lb_options.jacobi_scaling_eps +
      //                            Jp_column_norm2.array().sqrt())
      //                               .inverse();
      //      }
      // if (config.vio_debug) {
      //   std::cout << "\t[INFO] Stage 1" << std::endl;
      //}

      /*
       * [Levenberg-Marquardt 내부 반복]
       * - 댐핑 파라미터(lambda) 조정을 위한 backtracking
       * - 비용 함수 감소 여부에 따라 스텝 수용/거부 결정
       */
      for (int j = 0; it <= config.vio_max_iterations && !terminated; j++) {
        if (j > 0) {
          timer_iteration.reset();
          if (config.vio_debug) {
            std::cout << "Iteration " << it << ", backtracking" << std::endl;
          }
        }

        {
          // Timer t;

          // TODO: execution could be done staged

          //          // set (updated) damping for poses
          //          if (config.vio_lm_pose_damping_variant == 0) {
          //            lqr->setPoseDamping(lambda);
          //            stats.add("setPoseDamping", t.reset()).format("ms");
          //          }

          //          // scale landmark Jacobians only on the first inner
          //          iteration. if (scale_Jp && j == 0) {
          //            lqr->scaleJp_cols(jacobian_scaling);
          //            stats.add("scaleJp_cols", t.reset()).format("ms");
          //          }

          //          // set (updated) damping for landmarks
          //          if (config.vio_lm_landmark_damping_variant == 0) {
          //            lqr->setLandmarkDamping(lambda);
          //            stats.add("setLandmarkDamping", t.reset()).format("ms");
          //          }
        }

        // if (config.vio_debug) {
        //   std::cout << "\t[INFO] Stage 2 " << std::endl;
        // }

        /*
         * [선형 시스템 풀이]
         * - Schur complement로 얻은 카메라 포즈에 대한 선형 시스템
         * - LDLT 분해로 증분 계산
         */
        VecX inc;
        {
          Timer t;

          // get dense reduced camera system
          MatX H;
          VecX b;

          lqr->get_dense_H_b(H, b);

          if (config.vio_fix_long_term_keyframes) {
            for (const int64_t& ts : ltkfs) {
              if (aom.abs_order_map.count(ts) < 0) {
                printf("UNEXPECTED: ltkf ts=%ld not in aom\n", ts);
                continue;
              }
              const auto& [idx, size] = aom.abs_order_map.at(ts);
              H.template block<>(idx, 0, size, H.cols()).setZero();
              H.diagonal().template segment<POSE_SIZE>(idx).array() = 1e20;
              b.template segment<>(idx, size).setZero();
            }
          }

          stats.add("get_dense_H_b", t.reset()).format("ms");

          int iter = 0;
          bool inc_valid = false;
          constexpr int max_num_iter = 3;

          while (iter < max_num_iter && !inc_valid) {
            VecX Hdiag_lambda = (H.diagonal() * lambda).cwiseMax(min_lambda);
            MatX H_copy = H;
            H_copy.diagonal() += Hdiag_lambda;

            Eigen::LDLT<Eigen::Ref<MatX>> ldlt(H_copy);
            inc = ldlt.solve(b);
            stats.add("solve", t.reset()).format("ms");

            if (!inc.array().isFinite().all()) {
              lambda = lambda_vee * lambda;
              lambda_vee *= vee_factor;
            } else {
              inc_valid = true;
            }
            iter++;
          }

          if (!inc_valid) {
            std::cerr << "Still invalid inc after " << max_num_iter << " iterations." << std::endl;
          }

          if (show_uimat(UIMAT::HB)) {
            visual_data->geth(UIMAT::HB).H = std::make_shared<Eigen::MatrixXf>(H.template cast<float>());
            visual_data->geth(UIMAT::HB).b = std::make_shared<Eigen::VectorXf>(b.template cast<float>());
            visual_data->geth(UIMAT::HB).aom = std::make_shared<AbsOrderMap>(aom);
          }
        }

        // backup state (then apply increment and check cost decrease)
        backup();

        // backsubstitute (with scaled pose increment)
        Scalar l_diff = 0;
        {
          // negate pose increment before point update
          inc = -inc;

          Timer t;
          l_diff = lqr->backSubstitute(inc);
          stats.add("backSubstitute", t.reset()).format("ms");
        }

        // undo jacobian scaling before applying increment to poses
        //        if (scale_Jp) {
        //          inc.array() *= jacobian_scaling.array();
        //        }

        /*
         * [상태 변수 업데이트]
         * - 계산된 증분으로 카메라 포즈와 IMU 상태 업데이트
         */
        for (auto& [frame_id, state] : frame_poses) {
          int idx = aom.abs_order_map.at(frame_id).first;
          state.applyInc(inc.template segment<POSE_SIZE>(idx));
        }

        for (auto& [frame_id, state] : frame_states) {
          int idx = aom.abs_order_map.at(frame_id).first;
          state.applyInc(inc.template segment<POSE_VEL_BIAS_SIZE>(idx));
        }

        // compute stepsize
        Scalar step_norminf = inc.array().abs().maxCoeff();

        /*
         * [Cost function 평가]
         * - 업데이트 후 Cost function계산
         * - vision, IMU, marginalization prior 항 포함
         */
        Scalar after_update_marg_prior_error = 0;
        Scalar after_update_vision_and_inertial_error = 0;

        {
          Timer t;
          computeError(after_update_vision_and_inertial_error);
          computeMargPriorError(marg_data, after_update_marg_prior_error);

          Scalar after_update_imu_error = 0, after_bg_error = 0, after_ba_error = 0;
          ScBundleAdjustmentBase<Scalar>::computeImuError(
              aom, after_update_imu_error, after_bg_error, after_ba_error, frame_states, imu_meas,
              gyro_bias_sqrt_weight.array().square(), accel_bias_sqrt_weight.array().square(), g);

          after_update_vision_and_inertial_error += after_update_imu_error + after_bg_error + after_ba_error;

          stats.add("computerError2", t.reset()).format("ms");
        }

        Scalar after_error_total = after_update_vision_and_inertial_error + after_update_marg_prior_error;

        /*
         * [스텝 평가]
         * - 비용 감소량과 예측된 감소량 비교
         * - 스텝 수용 여부 결정
         * - 댐핑 파라미터 조정
         */
        Scalar f_diff;
        bool step_is_valid = false;
        bool step_is_successful = false;
        Scalar relative_decrease = 0;
        {
          // compute actual cost decrease
          f_diff = error_total - after_error_total;

          relative_decrease = f_diff / l_diff;

          if (config.vio_debug) {
            std::cout << "\t[EVAL] error: {:.4e}, f_diff {:.4e} l_diff {:.4e} "
                         "step_quality {:.2e} step_size {:.2e}\n"_format(after_error_total, f_diff, l_diff,
                                                                         relative_decrease, step_norminf);
          }

          // TODO: consider to remove assert. For now we want to test if we
          // even run into the l_diff <= 0 case ever in practice
          // BASALT_ASSERT_STREAM(l_diff > 0, "l_diff " << l_diff);

          // l_diff <= 0 is a theoretical possibility if the model cost change
          // is tiny and becomes numerically negative (non-positive). It might
          // not occur since our linear systems are not that big (compared to
          // large scale BA for example) and we also abort optimization quite
          // early and usually don't have large damping (== tiny step size).
          step_is_valid = l_diff > 0;
          step_is_successful = step_is_valid && relative_decrease > 0;
        }

        double iteration_time = timer_iteration.elapsed();
        double cumulative_time = timer_total.elapsed();

        stats.add("iteration", iteration_time).format("ms");
        {
          basalt::MemoryInfo mi;
          if (get_memory_info(mi)) {
            stats.add("resident_memory", mi.resident_memory);
            stats.add("resident_memory_peak", mi.resident_memory_peak);
          }
        }

        if (step_is_successful) {
          BASALT_ASSERT(step_is_valid);

          if (config.vio_debug) {
            //          std::cout << "\t[ACCEPTED] lambda:" << lambda
            //                    << " Error: " << after_error_total <<
            //                    std::endl;

            std::cout << "\t[ACCEPTED] error: {:.4e}, lambda: {:.1e}, it_time: "
                         "{:.3f}s, total_time: {:.3f}s\n"
                         ""_format(after_error_total, lambda, iteration_time, cumulative_time);
          }

          lambda *= std::max<Scalar>(Scalar(1.0) / 3, 1 - std::pow<Scalar>(2 * relative_decrease - 1, 3));
          lambda = std::max(min_lambda, lambda);

          lambda_vee = initial_vee;

          it++;

          // check function and parameter tolerance
          if ((f_diff > 0 && f_diff < Scalar(1e-6)) || step_norminf < Scalar(1e-4)) {
            converged = true;
            terminated = true;
          }

          // stop inner lm loop
          break;
        } else {
          std::string reason = step_is_valid ? "REJECTED" : "INVALID";

          if (config.vio_debug) {
            //          std::cout << "\t[REJECTED] lambda:" << lambda
            //                    << " Error: " << after_error_total <<
            //                    std::endl;

            std::cout << "\t[{}] error: {}, lambda: {:.1e}, it_time:"
                         "{:.3f}s, total_time: {:.3f}s\n"
                         ""_format(reason, after_error_total, lambda, iteration_time, cumulative_time);
          }

          lambda = lambda_vee * lambda;
          lambda_vee *= vee_factor;

          //        lambda = std::max(min_lambda, lambda);
          //        lambda = std::min(max_lambda, lambda);

          restore();
          it++;
          it_rejected++;

          if (lambda > max_lambda) {
            terminated = true;
            message = "Solver did not converge and reached maximum damping lambda";
          }
        }
      }
    }

    stats.add("optimize", timer_total.elapsed()).format("ms");
    stats.add("num_it", it).format("count");
    stats.add("num_it_rejected", it_rejected).format("count");

    // TODO: call filterOutliers at least once (also for CG version)

    stats_all_.merge_all(stats);
    stats_sums_.merge_sums(stats);

    if (config.vio_debug) {
      if (!converged) {
        if (terminated) {
          std::cout << "Solver terminated early after {} iterations: {}"_format(it, message);
        } else {
          std::cout << "Solver did not converge after maximum number of {} iterations"_format(it);
        }
      }

      stats.print();

      std::cout << "=================================" << std::endl;
    }
  }

  return true;
}

template <class Scalar_>
bool SqrtKeypointVioEstimator<Scalar_>::optimize_and_marg(const OpticalFlowInput::Ptr& input_images,
                                                          const std::map<int64_t, int>& num_points_connected,
                                                          const std::unordered_set<KeypointId>& lost_landmaks) {
  bool success = true;

  success &= optimize();
  if (!success) return false;
  input_images->addTime("optimized");

  success &= marginalize(num_points_connected, lost_landmaks);
  if (!success) return false;
  input_images->addTime("marginalized");

  return success;
}

template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::debug_finalize() {
  std::cout << "=== stats all ===\n";
  stats_all_.print();
  std::cout << "=== stats sums ===\n";
  stats_sums_.print();

  // save files
  stats_all_.save_json("stats_all.json");
  stats_sums_.save_json("stats_sums.json");
}

// //////////////////////////////////////////////////////////////////
// instatiate templates

#ifdef BASALT_INSTANTIATIONS_DOUBLE
template class SqrtKeypointVioEstimator<double>;
#endif

#ifdef BASALT_INSTANTIATIONS_FLOAT
template class SqrtKeypointVioEstimator<float>;
#endif

}  // namespace basalt
