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

// Original source for multi-scale implementation:
// https://github.com/DLR-RM/granite (MIT license)

#pragma once

#include <memory>
#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>

#include <basalt/imu/preintegration.h>
#include <basalt/utils/imu_types.h>
#include <basalt/utils/keypoints.h>

namespace basalt {

/*
1. 클래스 정의 및 상속: MultiscaleFrameToFrameOpticalFlow 클래스는 OpticalFlowTyped를 상속받아 멀티스케일 광학 흐름을 구현합니다.
생성자: VIO 설정 및 카메라 캘리브레이션 정보를 받아 초기화하며, IMU 노이즈 공분산을 설정합니다.
processingLoop: 메인 루프로, 이미지 프레임과 IMU 데이터를 입력받아 처리합니다. 상태 추정 및 광학 흐름 계산을 담당합니다.
processImu: IMU 데이터를 적분하여 현재 상태를 예측합니다. IMU 바이어스 보정 및 노이즈를 고려한 예측이 이루어집니다.
processFrame: 실제 프레임 처리가 이루어지는 핵심 함수입니다. 이미지 피라미드를 생성하고, 이전 프레임과의 특징점 추적, 새로운 특징점 추가, 불필요한 특징점 제거 등의 작업을 수행합니다.
trackPoints: 두 이미지 피라미드 간의 특징점을 추적하는 함수입니다. 스테레오 매칭 또는 시간적 트래킹에 사용되며, TBB를 사용한 병렬 처리가 눈에 띕니다. 깊이 정보를 활용하여 추적 정확도를 높입니다.
trackPoint: 단일 특징점을 여러 피라미드 레벨에 걸쳐 추적합니다. 상위 레벨(저해상도)에서 시작하여 하위 레벨(고해상도)로 이동하며 정밀도를 높입니다.
trackPointAtLevel: 특정 피라미드 레벨에서 패치 기반의 KLT(Kanade-Lucas-Tomasi) 유사 알고리즘을 사용하여 반복적으로 특징점의 위치를 최적화합니다.
addPoints: 새로운 특징점을 현재 프레임에 추가합니다. 주로 여러 카메라 간의 공통 영역 또는 새로운 영역에서 특징점을 검출합니다.
addPointsForCamera: 특정 카메라 및 특정 피라미드 레벨에서 새로운 특징점을 검출합니다. FAST와 유사한 특징점 검출 알고리즘이 사용될 수 있습니다.
cam0OverlapCellsMasksForCam: 주 카메라(cam0)와 다른 카메라 이미지 간의 중첩 영역을 계산하여 마스크를 생성합니다. 이는 특징점 매칭 효율성을 높이는 데 사용됩니다.
filterPoints: 추적된 특징점들을 필터링합니다. 주로 에피폴라 제약조건(epipolar constraint)을 사용하여 잘못 추적된 특징점을 제거합니다.
filterPointsForCam: 특정 카메라에 대해 에피폴라 오차를 기준으로 특징점을 필터링합니다.
*/

/// @brief 멀티스케일 프레임 대 프레임 광학 흐름(Optical Flow)을 계산하는 클래스입니다.
/// OpticalFlowTyped를 상속하며, 패치를 모든 피라미드 레벨에서 생성할 수 있는 점이 특징입니다.
template <typename Scalar, template <typename> typename Pattern>
class MultiscaleFrameToFrameOpticalFlow final : public OpticalFlowTyped<Scalar, Pattern> {
 public:
  using Vector3d = Eigen::Matrix<double, 3, 1>;

  using typename OpticalFlowTyped<Scalar, Pattern>::PatchT;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector2;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix2;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector3;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix3;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector4;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix4;
  using typename OpticalFlowTyped<Scalar, Pattern>::SE2;
  using typename OpticalFlowTyped<Scalar, Pattern>::SE3;
  using OpticalFlowTyped<Scalar, Pattern>::getNumCams;
  using OpticalFlowTyped<Scalar, Pattern>::calib;
  using OpticalFlowTyped<Scalar, Pattern>::E; // Epipolar 행렬 (스테레오 카메라의 경우)

  using OpticalFlowBase::config;
  using OpticalFlowBase::depth_guess;
  using OpticalFlowBase::first_state_arrived;
  using OpticalFlowBase::frame_counter;
  using OpticalFlowBase::input_depth_queue;
  using OpticalFlowBase::input_img_queue;
  using OpticalFlowBase::input_imu_queue;
  using OpticalFlowBase::input_state_queue;
  using OpticalFlowBase::last_keypoint_id;
  using OpticalFlowBase::latest_state; // 가장 최근의 VIO 상태 (포즈, 속도, IMU 바이어스)
  using OpticalFlowBase::old_pyramid;
  using OpticalFlowBase::output_queue;
  using OpticalFlowBase::patch_coord;
  using OpticalFlowBase::predicted_state; // IMU 예측을 통해 얻은 현재 상태
  using OpticalFlowBase::processing_thread;
  using OpticalFlowBase::pyramid; // 현재 프레임의 이미지 피라미드
  using OpticalFlowBase::show_gui;
  using OpticalFlowBase::t_ns; // 현재 프레임의 타임스탬프
  using OpticalFlowBase::transforms; // 광학 흐름 결과 (특징점, 추적 정보 등)

  /// @brief 생성자. VIO 설정 및 카메라 캘리브레이션 정보를 초기화합니다.
  /// @param conf VIO 설정값
  /// @param cal 카메라 캘리브레이션 데이터
  MultiscaleFrameToFrameOpticalFlow(const VioConfig& conf, const Calibration<double>& cal)
      : OpticalFlowTyped<Scalar, Pattern>(conf, cal),
        accel_cov(cal.dicrete_time_accel_noise_std().array().square()), // 가속도계 노이즈 공분산
        gyro_cov(cal.dicrete_time_gyro_noise_std().array().square()) { // 자이로스코프 노이즈 공분산
    latest_state = std::make_shared<PoseVelBiasState<double>>();
    predicted_state = std::make_shared<PoseVelBiasState<double>>();
  }

  /// @brief 주 처리 루프. 이미지 및 IMU 데이터를 받아 광학 흐름을 계산하고 결과를 출력합니다.
  /// 이 함수는 별도의 스레드에서 실행됩니다.
  void processingLoop() override {
    using std::make_shared;
    OpticalFlowInput::Ptr img; // 입력 이미지 데이터

    while (true) {
      input_img_queue.pop(img); // 이미지 큐에서 데이터를 가져옴 (Blocking)

      if (img == nullptr) { // 종료 신호
        if (output_queue) output_queue->push(nullptr);
        break;
      }
      img->addTime("frames_received");

      // 깊이 추정값 업데이트
      while (input_depth_queue.try_pop(depth_guess)) continue;
      if (show_gui) img->depth_guess = depth_guess;

      // VIO 상태 업데이트
      if (!input_state_queue.empty()) {
        while (input_state_queue.try_pop(latest_state)) continue;  // 최신 상태로 갱신
        first_state_arrived = true;
      } else if (first_state_arrived) {
        // 새로운 상태 정보가 없으면 예측된 상태를 사용
        latest_state = make_shared<PoseVelBiasState<double>>(*predicted_state);
      }

      // IMU 처리 및 상태 예측
      if (first_state_arrived) {
        auto pim = processImu(img->t_ns); // 현재 이미지 시간까지 IMU 데이터 처리
        pim.predictState(*latest_state, constants::g, *predicted_state); // 다음 상태 예측
      }

      processFrame(img->t_ns, img); // 프레임 처리 (광학 흐름 계산)
    }
  }

  /// @brief IMU 데이터를 처리하여 적분하고, 이를 기반으로 현재 상태를 예측합니다.
  /// @param curr_t_ns 현재 프레임의 타임스탬프
  /// @return 적분된 IMU 측정값 (IntegratedImuMeasurement)
  IntegratedImuMeasurement<double> processImu(int64_t curr_t_ns) {
    int64_t prev_t_ns = t_ns; // 이전 프레임의 타임스탬프
    Vector3d bg = latest_state->bias_gyro; // 최근 자이로 바이어스
    Vector3d ba = latest_state->bias_accel; // 최근 가속도계 바이어스
    IntegratedImuMeasurement<double> pim{prev_t_ns, bg, ba}; // IMU preintegration 객체 초기화

    if (input_imu_queue.empty()) return pim; // IMU 데이터 없으면 반환

    auto pop_imu = [&](ImuData<double>::Ptr& data) -> bool {
      input_imu_queue.pop(data);  // IMU 큐에서 데이터 가져옴 (Blocking)
      if (data == nullptr) return false; // 종료 신호

      // IMU 데이터 캘리브레이션
      Vector3 a = calib.calib_accel_bias.getCalibrated(data->accel.cast<Scalar>());
      Vector3 g = calib.calib_gyro_bias.getCalibrated(data->gyro.cast<Scalar>());
      data->accel = a.template cast<double>();
      data->gyro = g.template cast<double>();
      return true;
    };

    typename ImuData<double>::Ptr data = nullptr;
    if (!pop_imu(data)) return pim;

    // 이전 프레임 시간보다 오래된 IMU 데이터는 버림
    while (data->t_ns <= prev_t_ns) {
      if (!pop_imu(data)) return pim;
    }

    // 현재 프레임 시간까지의 IMU 데이터를 적분
    while (data->t_ns <= curr_t_ns) {
      pim.integrate(*data, accel_cov, gyro_cov); // IMU 측정값 적분
      if (!pop_imu(data)) return pim;
    }

    // 마지막 IMU 샘플 처리: 현재 시간(curr_t_ns) 직전의 샘플이 현재 시간에 발생했다고 가정하여 적분
    if (pim.get_start_t_ns() + pim.get_dt_ns() < curr_t_ns) {
      data->t_ns = curr_t_ns;
      pim.integrate(*data, accel_cov, gyro_cov);
    }

    return pim;
  }

  /// @brief 현재 프레임을 처리하여 광학 흐름을 계산합니다.
  /// 이미지 피라미드 생성, 특징점 추적, 새로운 특징점 추가, 특징점 필터링 등의 작업을 수행합니다.
  /// @param curr_t_ns 현재 프레임의 타임스탬프
  /// @param new_img_vec 현재 프레임의 이미지 데이터
  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return; // 유효하지 않은 이미지 데이터
    }

    const size_t num_cams = getNumCams(); // 카메라 개수

    if (t_ns < 0) { // 첫 프레임 처리
      t_ns = curr_t_ns;

      transforms = std::make_shared<OpticalFlowResult>();
      transforms->keypoints.resize(num_cams);
      transforms->pyramid_levels.resize(num_cams);
      transforms->tracking_guesses.resize(num_cams);
      transforms->matching_guesses.resize(num_cams);
      transforms->t_ns = t_ns;

      pyramid = std::make_shared<std::vector<ManagedImagePyr<uint16_t>>>();
      pyramid->resize(num_cams);

      // 병렬로 이미지 피라미드 생성
      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_cams), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels);
        }
      });

      transforms->input_images = new_img_vec;

      addPoints(); // 새로운 특징점 추가
      filterPoints(); // 특징점 필터링 (첫 프레임에서는 의미 없을 수 있음)
    } else { // 이후 프레임 처리
      t_ns = curr_t_ns;

      old_pyramid = pyramid; // 이전 프레임의 피라미드를 old_pyramid로 저장

      // 현재 프레임의 이미지 피라미드 생성
      pyramid = std::make_shared<std::vector<ManagedImagePyr<uint16_t>>>();
      pyramid->resize(num_cams);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_cams), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels);
        }
      });

      OpticalFlowResult::Ptr new_transforms;
      new_transforms = std::make_shared<OpticalFlowResult>();
      new_transforms->keypoints.resize(num_cams);
      new_transforms->tracking_guesses.resize(num_cams);
      new_transforms->matching_guesses.resize(num_cams);
      new_transforms->pyramid_levels.resize(num_cams);
      new_transforms->t_ns = t_ns;

      // 이전 상태(latest_state)와 예측된 현재 상태(predicted_state) 간의 상대 변환 계산
      SE3 T_i1 = latest_state->T_w_i.template cast<Scalar>(); // 이전 IMU 좌표계 -> 월드 좌표계
      SE3 T_i2 = predicted_state->T_w_i.template cast<Scalar>(); // 현재 IMU 좌표계 -> 월드 좌표계
      for (size_t i = 0; i < num_cams; i++) {
        SE3 T_c1 = T_i1 * calib.T_i_c[i]; // 이전 카메라 좌표계 -> 월드 좌표계
        SE3 T_c2 = T_i2 * calib.T_i_c[i]; // 현재 카메라 좌표계 -> 월드 좌표계
        SE3 T_c1_c2 = T_c1.inverse() * T_c2; // 이전 카메라 -> 현재 카메라 상대 변환 (추적 시 사용)

        // 특징점 추적 (이전 프레임 -> 현재 프레임)
        trackPoints(old_pyramid->at(i), pyramid->at(i),                               // 이미지 피라미드 (이전, 현재)
                    transforms->keypoints[i], transforms->pyramid_levels[i],          // 이전 프레임 특징점 및 피라미드 레벨
                    new_transforms->keypoints[i], new_transforms->pyramid_levels[i],  // 현재 프레임 추적된 특징점 및 레벨 (결과)
                    new_transforms->tracking_guesses[i],                              // 추적 초기 추정값 (결과)
                    new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i); // 마스크, 상대 변환, 카메라 인덱스
      }

      transforms = new_transforms; // 추적 결과를 현재 transforms로 업데이트
      transforms->input_images = new_img_vec;

      addPoints(); // 새로운 특징점 추가
      filterPoints(); // 특징점 필터링
    }

    // 설정된 주기에 따라 광학 흐름 결과 출력
    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      transforms->input_images->addTime("opticalflow_produced");
      output_queue->push(transforms);
    }

    frame_counter++;
  }

  /// @brief 두 이미지 피라미드 간에 특징점(keypoints)을 추적합니다.
  /// TBB를 사용하여 병렬로 처리합니다. 양방향 추적 (forward-backward tracking) 및 에피폴라 제약 등을 활용하여 강인한 추적을 수행합니다.
  /// @param pyr_1 이전 이미지 피라미드
  /// @param pyr_2 현재 이미지 피라미드
  /// @param keypoint_map_1 이전 프레임의 특징점 맵 (ID, Affine transform)
  /// @param pyramid_levels_1 이전 프레임 특징점의 피라미드 레벨
  /// @param keypoint_map_2 현재 프레임에서 추적된 특징점 맵 (결과)
  /// @param pyramid_levels_2 현재 프레임 추적된 특징점의 피라미드 레벨 (결과)
  /// @param guesses 추적 초기 추정값 (결과, GUI 표시용)
  /// @param masks1 이전 이미지 마스크
  /// @param masks2 현재 이미지 마스크
  /// @param T_c1_c2 카메라1에서 카메라2로의 상대 변환 (추적 초기값 계산에 사용)
  /// @param cam1 카메라1 인덱스
  /// @param cam2 카메라2 인덱스
  void trackPoints(const ManagedImagePyr<uint16_t>& pyr_1, const ManagedImagePyr<uint16_t>& pyr_2,  //
                   const Keypoints& keypoint_map_1, const KeypointLevels& pyramid_levels_1,         //
                   Keypoints& keypoint_map_2, KeypointLevels& pyramid_levels_2,                     //
                   Keypoints& guesses, const Masks& masks1, const Masks& masks2, const SE3& T_c1_c2, size_t cam1,
                   size_t cam2) const {
    size_t num_points = keypoint_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Keypoint> init_vec; // Keypoint는 Eigen::AffineCompact2f
    std::vector<size_t> pyramid_level;

    ids.reserve(num_points);
    init_vec.reserve(num_points);
    pyramid_level.reserve(num_points);

    // 추적할 특징점들을 벡터에 저장
    for (const auto& [kpid, affine] : keypoint_map_1) {
      ids.push_back(kpid);
      init_vec.push_back(affine);
      pyramid_level.push_back(pyramid_levels_1.at(kpid));
    }

    // 병렬 처리를 위한 결과 저장용 concurrent map
    tbb::concurrent_unordered_map<KeypointId, Keypoint, std::hash<KeypointId>> result, guesses_tbb;
    tbb::concurrent_unordered_map<KeypointId, size_t, std::hash<KeypointId>> pyrlvls_tbb;

    bool tracking = cam1 == cam2; // 동일 카메라면 시간적 추적 (tracking)
    bool matching = cam1 != cam2; // 다른 카메라면 공간적 매칭 (matching)
    MatchingGuessType guess_type = config.optical_flow_matching_guess_type;
    bool match_guess_uses_depth = guess_type != MatchingGuessType::SAME_PIXEL;
    const bool use_depth = tracking || (matching && match_guess_uses_depth); // 깊이 정보 사용 여부
    const double depth = depth_guess; // 추정된 깊이값

    // TBB 병렬 처리 람다 함수
    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r]; // 원본 특징점 위치 (이전 프레임)
        Eigen::AffineCompact2f transform_2 = transform_1; // 추적될 특징점 위치 (현재 프레임), 초기값은 이전과 동일

        auto t1 = transform_1.translation(); // 픽셀 좌표 (이전)
        auto t2 = transform_2.translation(); // 픽셀 좌표 (현재)

        if (masks1.inBounds(t1.x(), t1.y())) continue; // 마스크 영역 내에 있으면 건너뜀

        bool valid = true;

        Eigen::Vector2f off{0, 0}; // 초기 추정값 보정량

        // 깊이 정보를 사용하여 추적 초기값(t2_guess) 계산
        if (use_depth) {
          Vector2 t2_guess;
          Scalar _; // reprojection error (사용 안 함)
          // projectBetweenCams: cam1의 t1 (깊이 depth) -> cam2의 t2_guess 로 투영
          calib.projectBetweenCams(t1, depth, t2_guess, _, T_c1_c2, cam1, cam2);
          off = t2 - t2_guess; // 실제 transform_2.translation()과의 차이 (보정 전)
        }

        t2 -= off;  // This modifies transform_2.translation(), 깊이 기반 초기 추정값으로 업데이트

        if (show_gui) {
          guesses_tbb[id] = transform_2; // GUI 표시용 초기 추정값 저장
        }

        // 초기 추정값이 이미지 경계 내에 있는지 확인
        valid = t2(0) >= 0 && t2(1) >= 0 && t2(0) < pyr_2.lvl(0).w && t2(1) < pyr_2.lvl(0).h;
        if (!valid) continue;

        // 순방향 추적: pyr_1 -> pyr_2
        valid = trackPoint(pyr_1, pyr_2, transform_1, pyramid_level[r], transform_2);
        if (!valid) continue;

        if (masks2.inBounds(t2.x(), t2.y())) continue; // 추적 결과가 마스크 영역 내에 있으면 건너뜀

        // 역방향 추적 (Forward-Backward consistency check)
        Eigen::AffineCompact2f transform_1_recovered = transform_2; // 현재 추적된 위치에서 시작
        auto t1_recovered = transform_1_recovered.translation();

        t1_recovered += off; // 초기 추정값 보정량 다시 더함 (역방향 추적 시에는 원본 이미지에서의 위치이므로)

        // 역방향 추적: pyr_2 -> pyr_1
        valid = trackPoint(pyr_2, pyr_1, transform_2, pyramid_level[r], transform_1_recovered);
        if (!valid) continue;

        // 순방향-역방향 추적 결과 비교
        Scalar dist2 = (t1 - t1_recovered).squaredNorm(); // 원본 위치와 역추적된 위치 간의 거리 제곱

        if (dist2 < config.optical_flow_max_recovered_dist2) { // 허용 오차 이내면 유효한 추적으로 판단
          result[id] = transform_2; // 추적 결과 저장
          pyrlvls_tbb[id] = pyramid_level[r]; // 해당 특징점의 피라미드 레벨 저장
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);
    tbb::parallel_for(range, compute_func); // 병렬 실행

    // 결과 맵 업데이트
    keypoint_map_2.clear();
    keypoint_map_2.insert(result.begin(), result.end());
    guesses.clear();
    guesses.insert(guesses_tbb.begin(), guesses_tbb.end());
    pyramid_levels_2.clear();
    pyramid_levels_2.insert(pyrlvls_tbb.begin(), pyrlvls_tbb.end());
  }

  /// @brief 단일 특징점(keypoint)을 이미지 피라미드를 따라 추적합니다.
  /// 상위 레벨(저해상도)에서 하위 레벨(고해상도)로 내려가면서 정밀하게 위치를 찾습니다 (Coarse-to-fine).
  /// @param old_pyr 이전 이미지 피라미드
  /// @param pyr 현재 이미지 피라미드
  /// @param old_transform 이전 프레임에서의 특징점 위치 (Affine)
  /// @param pyramid_level 해당 특징점이 정의된 피라미드 레벨
  /// @param transform 현재 프레임에서 추적된 특징점 위치 (Affine, 결과)
  /// @return 추적 성공 여부
  inline bool trackPoint(const ManagedImagePyr<uint16_t>& old_pyr, const ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform, const size_t pyramid_level,
                         Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity(); // Affine 변환의 회전/스케일 부분은 초기화

    // 설정된 최상위 레벨부터 해당 특징점의 피라미드 레벨까지 반복
    for (int level = config.optical_flow_levels; level >= int(pyramid_level); level--) {
      const Scalar scale = 1 << unsigned(level); // 현재 레벨의 스케일 팩터 (2^level)

      Eigen::AffineCompact2f transform_tmp = transform; // 현재 레벨에서의 임시 변환

      transform_tmp.translation() /= scale; // 현재 레벨의 좌표계로 변환

      // 이전 이미지의 현재 레벨에서 패치 생성
      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid; // 패치가 유효한지 (이미지 경계 내에 있는지 등)
      if (patch_valid) {
        // 현재 레벨에서 특징점 추적 수행
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform_tmp);
      }

      // 만약 (특징점 레벨 + 1) 에서 패치가 유효하지 않으면, 더 낮은 레벨로 내려가도 의미 없으므로 실패 처리
      // 이는 coarse 레벨에서 이미 추적이 실패했음을 의미.
      if (level == int(pyramid_level) + 1 && !patch_valid) {
        return false;
      }

      transform_tmp.translation() *= scale; // 다시 원본 스케일로 변환

      if (patch_valid) {
        transform = transform_tmp; // 유효하면 현재 변환 업데이트
      }
    }

    // 최종 Affine 변환의 linear part는 원본 특징점의 linear part와 추적된 linear part를 곱하여 설정
    // (여기서는 transform.linear()가 Identity로 시작했으므로 사실상 old_transform.linear()가 됨)
    // 이는 주로 Affine 변화를 추정하는 경우에 의미가 있으며, 여기서는 Identity로 고정됨.
    // TODO: Affine 변환 추정이 실제로 필요한지 확인 필요. 현재 코드는 translation만 추정하는 KLT와 유사.
    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  /// @brief 특정 피라미드 레벨에서 패치 기반으로 특징점을 추적합니다 (예: KLT 알고리즘).
  /// 반복적으로 residual을 최소화하는 위치 변화(increment)를 계산하여 `transform`을 업데이트합니다.
  /// @param img_2 현재 레벨의 이미지 (추적 대상)
  /// @param dp 이전 이미지에서 생성된 패치 정보 (템플릿)
  /// @param transform 현재 레벨에서의 특징점 위치 (Affine, 입출력)
  /// @return 추적 성공 여부
  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2, const PatchT& dp,
                                Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    // 설정된 최대 반복 횟수만큼 최적화 수행
    for (int iteration = 0; patch_valid && iteration < config.optical_flow_max_iterations; iteration++) {
      typename PatchT::VectorP res; // Residual 벡터 (밝기 차이)

      // 현재 transform을 사용하여 패치 패턴(pattern2)을 변환
      // PatchT::pattern2는 패치 중심을 기준으로 한 상대 좌표들
      typename PatchT::Matrix2P transformed_pat = transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation(); // 절대 좌표로 변환

      // 변환된 패턴 위치에서 이미지(img_2)의 밝기 값과 원본 패치(dp)의 밝기 값을 비교하여 residual 계산
      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        // KLT의 핵심: J^T * W * J * dx = -J^T * W * (I_t - I_s)
        // inc = - (J^T J)^-1 J^T res = -H_se2_inv_J_se2_T * res
        // dp.H_se2_inv_J_se2_T는 미리 계산된 (J^T J)^-1 J^T (Hessian 역행렬 * Jacobian 전치)
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res; // 위치 변화량 (SE2 increment: dx, dy, dtheta)

        // NaN 또는 무한대 값 방지
        patch_valid &= inc.array().isFinite().all();

        // 매우 큰 변화량 방지
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          // SE2::exp(inc)를 통해 증분 변환 행렬을 얻고, 현재 transform에 곱하여 업데이트
          // (SE2 그룹 연산: 현재 변환에 작은 변환을 누적)
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2; // 이미지 경계에서 얼마나 안쪽에 있어야 하는지

          // 업데이트된 위치가 이미지 경계 내에 있는지 확인
          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  /// @brief 특정 카메라 및 피라미드 레벨에 대해 새로운 특징점을 검출하여 추가합니다.
  /// 기존 특징점들을 피하고, 그리드 기반으로 균일하게 분포된 특징점을 찾습니다.
  /// @param cam_id 카메라 인덱스
  /// @param level 특징점을 검출할 피라미드 레벨
  /// @return 새로 추가된 특징점(Keypoints)과 해당 피라미드 레벨(KeypointLevels)
  std::pair<Keypoints, KeypointLevels> addPointsForCamera(size_t cam_id, int level) {
    Eigen::aligned_vector<Eigen::Vector2d> pts;  // 현재 해당 레벨 및 인접 레벨에 이미 존재하는 특징점들의 위치
    for (const auto& [kpid, affine] : transforms->keypoints.at(cam_id)) {
      const int point_level = int(transforms->pyramid_levels.at(cam_id).at(kpid));
      // 현재 검출하려는 레벨(level) 및 그 인접 레벨(-1, +1)에 있는 점들을 고려
      // -> 너무 가까운 곳에 새로운 점을 찍지 않기 위함
      if (point_level >= level - 1 && point_level <= level + 1) {
        const Scalar pt_scale = 1 << unsigned(point_level); // 해당 점이 정의된 레벨의 스케일
        pts.emplace_back((affine.translation() / pt_scale).template cast<double>()); // 현재 검출 레벨 스케일로 변환하여 저장
      }
    }

    const Scalar scale = 1 << unsigned(level); // 현재 검출 레벨의 스케일

    KeypointsData kd;  // 검출된 새로운 코너점들
    // detectKeypoints: 이미지에서 코너점 검출 (예: FAST)
    // config.optical_flow_detection_grid_size: 그리드 셀 크기
    // config.optical_flow_detection_num_points_cell: 셀 당 최대 검출 점 개수
    // config.optical_flow_detection_min_threshold, max_threshold: 코너 검출 임계값
    // config.optical_flow_image_safe_radius: 이미지 경계에서 안전 반경
    // transforms->input_images->masks.at(cam_id): 마스크 영역 (검출 제외 영역)
    // pts: 기존 점들 (이 근처는 피해서 검출)
    detectKeypoints(pyramid->at(cam_id).lvl(level), kd, config.optical_flow_detection_grid_size,
                    config.optical_flow_detection_num_points_cell, config.optical_flow_detection_min_threshold,
                    config.optical_flow_detection_max_threshold, config.optical_flow_image_safe_radius,
                    transforms->input_images->masks.at(cam_id), pts);

    //! @note 모든 레벨에서 동일한 그리드 크기를 사용. 스케일링 시 특징점 수는 증가하지만 ATE/RTE 점수는 약간 나빠지거나 좋아짐.

    Keypoints new_kpts; // 새로 추가될 특징점들
    KeypointLevels new_pyramid_levels; // 새로 추가될 특징점들의 피라미드 레벨
    for (auto& corner : kd.corners) {  // 검출된 코너점들을 Keypoint로 변환
      auto transform = Eigen::AffineCompact2f::Identity();
      transform.translation() = corner.cast<Scalar>() * scale; // 원본 이미지 스케일로 변환

      transforms->keypoints.at(cam_id)[last_keypoint_id] = transform; // 전체 특징점 맵에 추가
      transforms->pyramid_levels.at(cam_id)[last_keypoint_id] = level; // 피라미드 레벨 저장
      new_kpts[last_keypoint_id] = transform; // 반환할 맵에도 추가
      new_pyramid_levels[last_keypoint_id] = level;

      last_keypoint_id++; // 다음 특징점 ID 증가
    }

    return std::make_pair(new_kpts, new_pyramid_levels);
  }

  /// @brief 특정 카메라(cam_id) 이미지에서 주 카메라(cam0)와 겹치는 영역에 대한 셀 마스크를 생성합니다.
  /// 깊이 추정값(depth_guess)을 사용하여 cam_id의 셀 중심을 cam0 이미지로 투영하고,
  /// cam0 이미지 경계 내에 있으면 해당 셀을 "겹치는 영역"으로 간주하여 마스크를 만듭니다.
  /// 이는 cam_id에서 특징점을 추가할 때, 이미 cam0과 매칭될 가능성이 높은 영역을 피하거나 우선순위를 낮추는 데 사용될 수 있습니다.
  /// @param cam_id 대상 카메라 인덱스
  /// @return 생성된 마스크 (겹치는 셀들의 집합)
  Masks cam0OverlapCellsMasksForCam(size_t cam_id) {
    int C = config.optical_flow_detection_grid_size;  // 셀 크기

    int w = transforms->input_images->img_data.at(cam_id).img->w; // 대상 카메라 이미지 너비
    int h = transforms->input_images->img_data.at(cam_id).img->h; // 대상 카메라 이미지 높이

    // 그리드 셀의 시작점 계산 (이미지 중앙에 가깝게)
    int x_start = (w % C) / 2;
    int y_start = (h % C) / 2;

    int x_stop = x_start + C * (w / C - 1);
    int y_stop = y_start + C * (h / C - 1);

    int x_first = x_start + C / 2; // 첫 번째 셀 중심 x
    int y_first = y_start + C / 2; // 첫 번째 셀 중심 y

    int x_last = x_stop + C / 2;   // 마지막 셀 중심 x
    int y_last = y_stop + C / 2;   // 마지막 셀 중심 y

    Masks masks; // 결과 마스크
    for (int y = y_first; y <= y_last; y += C) {
      for (int x = x_first; x <= x_last; x += C) {
        Vector2 ci_uv{(Scalar)x, (Scalar)y}; // cam_id의 셀 중심 좌표
        Vector2 c0_uv; // cam0에 투영된 좌표
        Scalar _; // reprojection error (사용 안 함)
        // projectBetweenCams: cam_id의 ci_uv (깊이 depth_guess) -> cam0의 c0_uv 로 투영
        bool projected = calib.projectBetweenCams(ci_uv, depth_guess, c0_uv, _, cam_id, 0);
        // cam0 이미지 경계 내에 있는지 확인
        bool in_bounds = c0_uv.x() >= 0 && c0_uv.x() < calib.intrinsics[0].w() && c0_uv.y() >= 0 && c0_uv.y() < calib.intrinsics[0].h();
        bool valid = projected && in_bounds;
        if (valid) { // 유효하게 투영되면 해당 셀을 마스크에 추가
          Rect cell_mask(x - C / 2.0f, y - C / 2.0f, C, C);
          masks.masks.push_back(cell_mask);
        }
      }
    }
    return masks;
  }

  /// @brief 모든 카메라에 대해 새로운 특징점을 추가합니다.
  /// 먼저 주 카메라(cam0)에서 특징점을 추가하고, 그 다음 다른 카메라(cam_i)에서 cam0과 매칭될 수 있는 특징점을 찾습니다.
  /// 그 후, cam0과 겹치지 않는 영역에서도 추가적으로 특징점을 검출합니다 (config.optical_flow_detection_nonoverlap 설정 시).
  void addPoints() {
    //! @note 가장 작은 피라미드 레벨은 사용하지 않음. 사용 시 ATE/RTE 점수가 약간 나빠지는 경향.
    // config.optical_flow_levels - 1 까지의 레벨에서 특징점 추가
    for (int level = 0; level < config.optical_flow_levels - 1; level++) {
      Masks& ms0 = transforms->input_images->masks.at(0); // cam0의 마스크
      // cam0에서 새로운 특징점 추가 (결과는 kpts0, lvls0에 저장됨)
      auto [kpts0, lvls0] = addPointsForCamera(0, level);

      // 다른 카메라들에 대해 cam0에서 추가된 점들을 매칭 시도
      for (size_t i = 1; i < getNumCams(); i++) {
        Masks& ms = transforms->input_images->masks.at(i); // cam_i의 마스크
        Keypoints& mgs = transforms->matching_guesses.at(i); // cam_i의 매칭 추정값 (GUI용)

        // cam0의 새로운 특징점(kpts0)을 cam_i로 추적/매칭 시도
        auto& pyr0 = pyramid->at(0); // cam0 피라미드
        auto& pyri = pyramid->at(i); // cam_i 피라미드
        Keypoints kpts; // cam_i에서 매칭된 특징점 결과
        KeypointLevels lvls; // 매칭된 특징점의 피라미드 레벨 결과
        SE3 T_c0_ci = calib.T_i_c[0].inverse() * calib.T_i_c[i]; // cam0 -> cam_i 상대 변환
        trackPoints(pyr0, pyri, kpts0, lvls0, kpts, lvls, mgs, ms0, ms, T_c0_ci, 0, i);

        // 매칭된 점들을 cam_i의 전체 특징점 맵에 추가
        transforms->keypoints.at(i).insert(kpts.begin(), kpts.end());
        transforms->pyramid_levels.at(i).insert(lvls.begin(), lvls.end());
      }
    }

    // cam0과 겹치지 않는 영역에서 추가 특징점 검출 (설정된 경우)
    if (!config.optical_flow_detection_nonoverlap) return;
    for (size_t i = 1; i < getNumCams(); i++) {
      Masks& ms = transforms->input_images->masks.at(i);
      // cam_i의 마스크에 "cam0과 겹치는 영역"을 추가 (해당 영역에는 더 이상 점을 찍지 않도록)
      ms += cam0OverlapCellsMasksForCam(i);
      // 모든 레벨에 대해 (겹치지 않는 영역에서) 추가 특징점 검출
      for (int l = 0; l < config.optical_flow_levels - 1; l++) addPointsForCamera(i, l);
    }
  }

  /// @brief 특정 카메라(cam_id)의 특징점들을 에피폴라 제약조건을 사용하여 필터링합니다.
  /// cam0과 cam_id 간에 공통으로 관측되는 특징점들에 대해 에피폴라 오차를 계산하고,
  /// 오차가 큰 특징점들을 제거합니다. 스테레오 카메라 또는 다중 카메라 시스템에서 사용됩니다.
  /// @param cam_id 필터링할 카메라 인덱스 (0이 아닌 카메라)
  void filterPointsForCam(int cam_id) {
    std::set<KeypointId> kp_to_remove; // 제거할 특징점 ID 집합

    std::vector<KeypointId> kpids; // 공통 관측 특징점 ID
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1; // cam0, cam_id에서의 픽셀 좌표

    // cam_id의 모든 특징점에 대해 반복
    for (const auto& [kpid, affine] : transforms->keypoints.at(cam_id)) {
      auto it = transforms->keypoints.at(0).find(kpid); // 해당 ID가 cam0에도 있는지 확인

      if (it != transforms->keypoints.at(0).end()) { // cam0과 cam_id 모두에서 관측된 경우
        proj0.emplace_back(it->second.translation()); // cam0에서의 좌표
        proj1.emplace_back(affine.translation());     // cam_id에서의 좌표
        kpids.emplace_back(kpid);                     // 특징점 ID
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1; // 정규화된 이미지 평면 좌표 (homogeneous)
    std::vector<bool> p3d0_success, p3d1_success;    // unprojection 성공 여부

    // 픽셀 좌표를 정규화된 이미지 평면 좌표로 변환 (unproject)
    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[cam_id].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) { // 양쪽 모두 unprojection 성공 시
        // 에피폴라 제약: p1^T * E * p0 = 0 (여기서는 p3d1^T * E * p3d0)
        // E[cam_id]는 cam0과 cam_id 간의 Essential Matrix (또는 Fundamental Matrix)
        const double epipolar_error = std::abs(p3d0[i].transpose() * E[cam_id] * p3d1[i]);

        // 에피폴라 오차 임계값은 특징점의 피라미드 레벨에 따라 스케일링
        // (저해상도 레벨의 특징점은 오차 허용치가 더 큼)
        const Scalar scale = 1 << transforms->pyramid_levels.at(cam_id).at(kpids[i]);
        if (epipolar_error > config.optical_flow_epipolar_error * scale) {
          kp_to_remove.emplace(kpids[i]); // 오차가 크면 제거 대상
        }
      } else { // unprojection 실패 시 제거 대상
        kp_to_remove.emplace(kpids[i]);
      }
    }

    // 제거 대상으로 표시된 특징점들을 cam_id의 특징점 맵에서 실제로 제거
    for (int id : kp_to_remove) {
      transforms->keypoints.at(cam_id).erase(id);
      transforms->pyramid_levels.at(cam_id).erase(id); // 피라미드 레벨 정보도 함께 제거
    }
  }

  /// @brief 모든 보조 카메라(cam0 이외의 카메라)에 대해 특징점을 필터링합니다.
  void filterPoints() {
    for (size_t i = 1; i < getNumCams(); i++) {
      filterPointsForCam(i);
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  const Vector3d accel_cov; // 가속도계 이산 시간 노이즈 공분산
  const Vector3d gyro_cov;  // 자이로스코프 이산 시간 노이즈 공분산
};

}  // namespace basalt
