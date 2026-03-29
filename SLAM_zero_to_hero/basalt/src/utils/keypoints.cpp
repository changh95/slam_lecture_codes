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

#include <unordered_set>

#include <basalt/utils/build_config.h>
#include <basalt/utils/keypoints.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifndef BASALT_BUILD_SHARED_LIBRARY_ONLY
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#endif

namespace basalt {

// const int PATCH_SIZE = 31; // 패치 크기 정의
const int HALF_PATCH_SIZE = 15; // 패치 크기의 절반
const int EDGE_THRESHOLD = 19;  // 코너 검출 시 이미지 경계와의 최소 거리

// BRIEF 디스크립터 생성을 위한 미리 정의된 패턴 좌표 (x, a쌍)
const static signed char pattern_31_x_a[256] = {
    8,  4,   -11, 7,   2,   1,   -2,  -13, -13, 10,  -13, -11, 7,  -4,  -13, -9,  12,  -3,  -6,  11,  4,   5,   3,   -8,
    -2, -13, -7,  -4,  -10, 5,   5,   1,   9,   4,   2,   -4,  -8, 4,   0,   -13, -3,  -6,  8,   0,   7,   -13, 10,  -6,
    10, -13, -13, 3,   5,   -1,  3,   2,   -13, -13, -13, -7,  6,  -9,  -2,  -12, 3,   -7,  -3,  2,   -11, -1,  5,   -4,
    -9, -12, 10,  7,   -7,  -4,  7,   -7,  -13, -3,  7,   -13, 1,  2,   -4,  -1,  7,   1,   9,   -1,  -13, 7,   12,  6,
    5,  2,   3,   2,   9,   -8,  -11, 1,   6,   2,   6,   3,   7,  -11, -10, -5,  -10, 8,   4,   -10, 4,   -2,  -5,  7,
    -9, -5,  8,   -9,  1,   7,   -2,  11,  -12, 3,   5,   0,   -9, 0,   -1,  5,   3,   -13, -5,  -4,  6,   -7,  -13, 1,
    4,  -2,  2,   -2,  4,   -6,  -3,  7,   4,   -13, 7,   7,   -7, -8,  -13, 2,   10,  -6,  8,   2,   -11, -12, -11, 5,
    -2, -1,  -13, -10, -3,  2,   -9,  -4,  -4,  -6,  6,   -13, 11, 7,   -1,  -4,  -7,  -13, -7,  -8,  -5,  -13, 1,   1,
    9,  5,   -1,  -9,  -1,  -13, 8,   2,   7,   -10, -10, 4,   3,  -4,  5,   4,   -9,  0,   -12, 3,   -10, 8,   -8,  2,
    10, 6,   -7,  -3,  -1,  -3,  -8,  4,   2,   6,   3,   11,  -3, 4,   2,   -10, -13, -13, 6,   0,   -13, -9,  -13, 5,
    2,  -1,  9,   11,  3,   -1,  3,   -13, 5,   8,   7,   -10, 7,  9,   7,   -1};

// BRIEF 디스크립터 생성을 위한 미리 정의된 패턴 좌표 (y, a쌍)
const static signed char pattern_31_y_a[256] = {
    -3,  2,   9,   -12, -13, -7,  -10, -13, -3,  4,   -8,  7,   7,   -5,  2,   0,   -6,  6,   -13, -13, 7,   -3,
    -7,  -7,  11,  12,  3,   2,   -12, -12, -6,  0,   11,  7,   -1,  -12, -5,  11,  -8,  -2,  -2,  9,   12,  9,
    -5,  -6,  7,   -3,  -9,  8,   0,   3,   7,   7,   -10, -4,  0,   -7,  3,   12,  -10, -1,  -5,  5,   -10, -7,
    -2,  9,   -13, 6,   -3,  -13, -6,  -10, 2,   12,  -13, 9,   -1,  6,   11,  7,   -8,  -7,  -3,  -6,  3,   -13,
    1,   -1,  1,   -9,  -13, 7,   -5,  3,   -13, -12, 8,   6,   -12, 4,   12,  12,  -9,  3,   3,   -3,  8,   -5,
    11,  -8,  5,   -1,  -6,  12,  -2,  0,   -8,  -6,  -13, -13, -8,  -11, -8,  -4,  1,   -6,  -9,  7,   5,   -4,
    12,  7,   2,   11,  5,   -4,  9,   -7,  5,   6,   6,   -10, 1,   -2,  -12, -13, 1,   -10, -13, 5,   -2,  9,
    1,   -8,  -4,  11,  6,   4,   -5,  -5,  -3,  -12, -2,  -13, 0,   -3,  -13, -8,  -11, -2,  9,   -3,  -13, 6,
    12,  -11, -3,  11,  11,  -5,  12,  -8,  1,   -12, -2,  5,   -1,  7,   5,   0,   12,  -8,  11,  -3,  -10, 1,
    -11, -13, -13, -10, -8,  -6,  12,  2,   -13, -13, 9,   3,   1,   2,   -10, -13, -12, 2,   6,   8,   10,  -9,
    -13, -7,  -2,  2,   -5,  -9,  -1,  -1,  0,   -11, -4,  -6,  7,   12,  0,   -1,  3,   8,   -6,  -9,  7,   -6,
    5,   -3,  0,   4,   -6,  0,   8,   9,   -4,  4,   3,   -7,  0,   -6};

// BRIEF 디스크립터 생성을 위한 미리 정의된 패턴 좌표 (x, b쌍)
const static signed char pattern_31_x_b[256] = {
    9,  7,  -8,  12,  2,  1,  -2, -11, -12, 11,  -8,  -9,  12, -3, -12, -7, 12,  -2,  -4, 12, 5,   10,  6,   -6,
    -1, -8, -5,  -3,  -6, 6,  7,  4,   11,  4,   4,   -2,  -7, 9,  1,   -8, -2,  -4,  10, 1,  11,  -11, 12,  -6,
    12, -8, -8,  7,   10, 1,  5,  3,   -13, -12, -11, -4,  12, -7, 0,   -7, 8,   -4,  -1, 5,  -5,  0,   5,   -4,
    -9, -8, 12,  12,  -6, -3, 12, -5,  -12, -2,  12,  -11, 12, 3,  -2,  1,  8,   3,   12, -1, -10, 10,  12,  7,
    6,  2,  4,   12,  10, -7, -4, 2,   7,   3,   11,  8,   9,  -6, -5,  -3, -9,  12,  6,  -8, 6,   -2,  -5,  10,
    -8, -5, 9,   -9,  1,  9,  -1, 12,  -6,  7,   10,  2,   -5, 2,  1,   7,  6,   -8,  -3, -3, 8,   -6,  -5,  3,
    8,  2,  12,  0,   9,  -3, -1, 12,  5,   -9,  8,   7,   -7, -7, -12, 3,  12,  -6,  9,  2,  -10, -7,  -10, 11,
    -1, 0,  -12, -10, -2, 3,  -4, -3,  -2,  -4,  6,   -5,  12, 12, 0,   -3, -6,  -8,  -6, -6, -4,  -8,  5,   10,
    10, 10, 1,   -6,  1,  -8, 10, 3,   12,  -5,  -8,  8,   8,  -3, 10,  5,  -4,  3,   -6, 4,  -10, 12,  -6,  3,
    11, 8,  -6,  -3,  -1, -3, -8, 12,  3,   11,  7,   12,  -3, 4,  2,   -8, -11, -11, 11, 1,  -9,  -6,  -8,  8,
    3,  -1, 11,  12,  3,  0,  4,  -10, 12,  9,   8,   -10, 12, 10, 12,  0};

// BRIEF 디스크립터 생성을 위한 미리 정의된 패턴 좌표 (y, b쌍)
const static signed char pattern_31_y_b[256] = {
    5,   -12, 2,   -13, 12,  6,   -4, -8,  -9,  9,   -9,  12,  6,   0,  -3,  5,  -1,  12,  -8,  -8,  1,   -3, 12,  -2,
    -10, 10,  -3,  7,   11,  -7,  -1, -5,  -13, 12,  4,   7,   -10, 12, -13, 2,  3,   -9,  7,   3,   -10, 0,  1,   12,
    -4,  -12, -4,  8,   -7,  -12, 6,  -10, 5,   12,  8,   7,   8,   -6, 12,  5,  -13, 5,   -7,  -11, -13, -1, 2,   12,
    6,   -4,  -3,  12,  5,   4,   2,  1,   5,   -6,  -7,  -12, 12,  0,  -13, 9,  -6,  12,  6,   3,   5,   12, 9,   11,
    10,  3,   -6,  -13, 3,   9,   -6, -8,  -4,  -2,  0,   -8,  3,   -4, 10,  12, 0,   -6,  -11, 7,   7,   12, 2,   12,
    -8,  -2,  -13, 0,   -2,  1,   -4, -11, 4,   12,  8,   8,   -13, 12, 7,   -9, -8,  9,   -3,  -12, 0,   12, -2,  10,
    -4,  -13, 12,  -6,  3,   -5,  1,  -11, -7,  -5,  6,   6,   1,   -8, -8,  9,  3,   7,   -8,  8,   3,   -9, -5,  8,
    12,  9,   -5,  11,  -13, 2,   0,  -10, -7,  9,   11,  5,   6,   -2, 7,   -2, 7,   -13, -8,  -9,  5,   10, -13, -13,
    -1,  -9,  -13, 2,   12,  -10, -6, -6,  -9,  -7,  -13, 5,   -13, -3, -12, -1, 3,   -9,  1,   -8,  9,   12, -5,  7,
    -8,  -12, 5,   9,   5,   4,   3,  12,  11,  -13, 12,  4,   6,   12, 1,   1,  1,   -13, -13, 4,   -2,  -3, -2,  10,
    -9,  -1,  -2,  -8,  5,   10,  5,  5,   11,  -6,  -12, 9,   4,   -2, -2,  -11};

/**
 * @brief 16비트 원본 이미지에서 8비트 이미지로 변환 후, goodFeaturesToTrack을 사용하여 키포인트를 검출합니다.
 *        검출된 키포인트 중 이미지 경계 내에 있는 유효한 키포인트만 저장합니다.
 * @param img_raw 입력 16비트 이미지
 * @param kd 검출된 키포인트 데이터를 저장할 구조체
 * @param num_features 검출할 최대 특징점 수
 */
void detectKeypointsMapping(const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd, int num_features) {
  cv::Mat image(img_raw.h, img_raw.w, CV_8U); // OpenCV Mat 형식으로 이미지 생성

  // 16비트 이미지를 8비트 이미지로 변환 (상위 8비트 사용)
  uint8_t* dst = image.ptr();
  const uint16_t* src = img_raw.ptr;

  for (size_t i = 0; i < img_raw.size(); i++) {
    dst[i] = (src[i] >> 8);
  }

  std::vector<cv::Point2f> points;
  // OpenCV의 goodFeaturesToTrack 함수를 사용하여 코너 검출 (Shi-Tomasi 코너 검출)
  goodFeaturesToTrack(image, points, num_features, 0.01, 8);

  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  // 검출된 포인트들 중 이미지 경계 내에 있는 포인트만 저장
  for (size_t i = 0; i < points.size(); i++) {
    if (img_raw.InBounds(points[i].x, points[i].y, EDGE_THRESHOLD)) {
      kd.corners.emplace_back(points[i].x, points[i].y);
    }
  }
}

/**
 * @brief 이미지를 셀(cell) 단위로 나누고, 각 셀에서 FAST 알고리즘을 사용하여 키포인트를 검출합니다.
 *        셀별로 지정된 개수만큼의 키포인트를 검출하며, 임계값을 조절하여 검출 개수를 맞춥니다.
 *        safe_radius와 masks를 이용해 특정 영역의 키포인트 검출을 제외할 수 있습니다.
 * @param img_raw 입력 16비트 이미지
 * @param kd 검출된 키포인트 데이터를 저장할 구조체
 * @param cells 각 셀에 이미 검출된 특징점의 수를 나타내는 행렬 (0이면 해당 셀에서 검출 진행)
 * @param PATCH_SIZE 각 셀(패치)의 크기
 * @param num_points_cell 셀당 검출할 목표 특징점 수
 * @param min_threshold FAST 검출 시 최소 임계값
 * @param max_threshold FAST 검출 시 최대 임계값
 * @param safe_radius 이미지 중심으로부터 이 반경 이내의 특징점만 사용 (0이면 무시)
 * @param masks 검출에서 제외할 영역을 지정하는 마스크
 */
void detectKeypointsWithCells(const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd,
                              const Eigen::MatrixXi& cells, int PATCH_SIZE, int num_points_cell, int min_threshold,
                              int max_threshold, float safe_radius, const Masks& masks) {
  kd.corners.clear();
  kd.corner_responses.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  // 처리할 이미지 영역의 시작 및 끝 좌표 계산 (패치 단위로 정렬)
  const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
  const size_t x_stop = x_start + PATCH_SIZE * (img_raw.w / PATCH_SIZE - 1);

  const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
  const size_t y_stop = y_start + PATCH_SIZE * (img_raw.h / PATCH_SIZE - 1);

  // 모든 셀에 대해 반복
  for (size_t x = x_start; x <= x_stop; x += PATCH_SIZE) {
    for (size_t y = y_start; y <= y_stop; y += PATCH_SIZE) {
      // 해당 셀에 이미 충분한 특징점이 있거나 (cells > 0), 검출하지 않도록 표시된 경우 건너뜀
      if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0) continue;

      // 현재 셀에 해당하는 부분 이미지(sub-image)를 가져옴
      const basalt::Image<const uint16_t> sub_img_raw = img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

      cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U); // 부분 이미지를 위한 OpenCV Mat

      // 부분 이미지를 16비트에서 8비트로 변환
      for (int r = 0; r < PATCH_SIZE; r++) { // OpenCV Mat의 행(row) 접근을 위해 r 변수 사용
        uchar* sub_ptr = subImg.ptr(r);
        for (int c = 0; c < PATCH_SIZE; c++) { // OpenCV Mat의 열(column) 접근을 위해 c 변수 사용
          sub_ptr[c] = (sub_img_raw(c, r) >> 8); // basalt::Image는 (col, row) 접근
        }
      }

      int points_added = 0;        // 현재 셀에서 추가된 포인트 수
      int threshold = max_threshold; // FAST 임계값 초기화

      // 목표한 포인트 수를 채우거나 임계값이 최소값보다 작아질 때까지 반복
      while (points_added < num_points_cell && threshold >= min_threshold) {
        std::vector<cv::KeyPoint> points;
        // FAST 알고리즘으로 코너 검출
        cv::FAST(subImg, points, threshold);

        // 검출된 포인트를 응답값(response) 기준으로 내림차순 정렬
        std::sort(points.begin(), points.end(),
                  [](const cv::KeyPoint& a, const cv::KeyPoint& b) -> bool { return a.response > b.response; });

        // 정렬된 포인트를 순회하며 목표 개수만큼 추가
        for (size_t i = 0; i < points.size() && points_added < num_points_cell; i++) {
          float full_x = x + points[i].pt.x; // 전체 이미지에서의 x 좌표
          float full_y = y + points[i].pt.y; // 전체 이미지에서의 y 좌표
          // 이미지 중심으로부터의 거리 계산
          float dist_to_center = Eigen::Vector2f{full_x - img_raw.w / 2.0f, full_y - img_raw.h / 2.0f}.norm();

          // safe_radius가 설정되어 있고, 포인트가 반경 밖에 있으면 무시
          if (safe_radius != 0.0 && dist_to_center >= safe_radius) continue;
          // 마스크 영역 내에 있으면 무시
          if (masks.inBounds(full_x, full_y)) continue;
          // 이미지 경계 유효성 검사
          if (!img_raw.InBounds(full_x, full_y, EDGE_THRESHOLD)) continue;

          kd.corners.emplace_back(full_x, full_y);
          kd.corner_responses.emplace_back(points[i].response);
          points_added++;
        }

        threshold /= 2; // 다음 반복을 위해 임계값 감소
      }
    }
  }
}

/**
 * @brief 이미지에서 키포인트를 검출하는 메인 함수. 기존에 검출된 키포인트(`current_points`)를 고려하여
 *        셀 기반의 키포인트 검출(`detectKeypointsWithCells`)을 수행합니다.
 *        이를 통해 키포인트가 특정 영역에 몰리지 않고 이미지 전체에 걸쳐 분포되도록 합니다.
 * @param img_raw 입력 16비트 이미지
 * @param kd 검출된 키포인트 데이터를 저장할 구조체
 * @param PATCH_SIZE 각 셀(패치)의 크기
 * @param num_points_cell 셀당 검출할 목표 특징점 수
 * @param min_threshold FAST 검출 시 최소 임계값
 * @param max_threshold FAST 검출 시 최대 임계값
 * @param safe_radius 이미지 중심으로부터 이 반경 이내의 특징점만 사용 (0이면 무시)
 * @param masks 검출에서 제외할 영역을 지정하는 마스크
 * @param current_points 현재 추적중이거나 이전에 검출된 특징점들의 좌표. 이 점들이 있는 셀은 새로운 특징점 검출을 덜 수행하게 됨.
 */
void detectKeypoints(const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd, int PATCH_SIZE,
                     int num_points_cell, int min_threshold, int max_threshold, float safe_radius, const Masks& masks,
                     const Eigen::aligned_vector<Eigen::Vector2d>& current_points) {
  kd.corners.clear();
  kd.corner_angles.clear();
  kd.corner_descriptors.clear();

  // 처리할 이미지 영역의 시작 및 끝 좌표 계산
  const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
  const size_t x_stop = x_start + PATCH_SIZE * (img_raw.w / PATCH_SIZE - 1);

  const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
  const size_t y_stop = y_start + PATCH_SIZE * (img_raw.h / PATCH_SIZE - 1);

  Eigen::MatrixXi cells; // 각 셀에 있는 기존 특징점의 수를 저장할 행렬
  cells.setZero(img_raw.h / PATCH_SIZE + 1, img_raw.w / PATCH_SIZE + 1);

  // 기존 특징점들이 어느 셀에 위치하는지 계산하여 `cells` 행렬에 기록
  for (const Eigen::Vector2d& p : current_points) {
    if (p[0] >= x_start && p[1] >= y_start && p[0] < x_stop + PATCH_SIZE && p[1] < y_stop + PATCH_SIZE) {
      int x_cell = (p[0] - x_start) / PATCH_SIZE; // 오타 수정: x -> x_cell
      int y_cell = (p[1] - y_start) / PATCH_SIZE; // 오타 수정: y -> y_cell

      cells(y_cell, x_cell)++; // 해당 셀의 특징점 개수 증가
    }
  }

  // 셀 정보를 바탕으로 키포인트 검출 수행
  detectKeypointsWithCells(img_raw, kd, cells, PATCH_SIZE, num_points_cell, min_threshold, max_threshold, safe_radius,
                           masks);
}

/**
 * @brief 검출된 키포인트들의 각도(orientation)를 계산합니다.
 *        회전 불변성을 위해 특징점 주변 픽셀 값의 모멘트를 사용하여 주 방향을 계산합니다.
 * @param img_raw 입력 16비트 이미지
 * @param kd 키포인트 데이터 (코너 좌표 포함, 여기에 각도 정보 추가)
 * @param rotate_features 특징점 회전 계산 여부 (true이면 각도 계산)
 */
void computeAngles(const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd, bool rotate_features) {
  kd.corner_angles.resize(kd.corners.size()); // 각도 저장 공간 할당

  for (size_t i = 0; i < kd.corners.size(); i++) {
    const Eigen::Vector2d& p = kd.corners[i];

    const int cx = p[0]; // 키포인트 중심 x 좌표
    const int cy = p[1]; // 키포인트 중심 y 좌표

    double angle = 0; // 계산될 각도

    if (rotate_features) { // 회전 계산이 활성화된 경우
      double m01 = 0, m10 = 0; // y 방향 모멘트, x 방향 모멘트
      // 패치 내의 픽셀들을 순회하며 모멘트 계산 (IC-Angle 방식과 유사)
      for (int x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; x++) {
        for (int y = -HALF_PATCH_SIZE; y <= HALF_PATCH_SIZE; y++) {
          if (x * x + y * y <= HALF_PATCH_SIZE * HALF_PATCH_SIZE) { // 원형 패치 내의 픽셀만 고려
            double val = img_raw(cx + x, cy + y); // 픽셀 값
            m01 += y * val;
            m10 += x * val;
          }
        }
      }
      angle = atan2(m01, m10); // 아크탄젠트 함수로 각도 계산
    }
    kd.corner_angles[i] = angle; // 계산된 각도 저장
  }
}

/**
 * @brief 검출된 키포인트들에 대해 BRIEF 디스크립터를 계산합니다.
 *        미리 정의된 패턴(pattern_31_x_a/y_a, pattern_31_x_b/y_b)을 사용하여
 *        키포인트 주변의 픽셀 쌍들의 밝기 비교 결과를 이진 문자열로 표현합니다.
 *        계산된 각도(orientation)를 이용해 패턴을 회전시켜 회전 불변성을 갖도록 할 수 있습니다.
 * @param img_raw 입력 16비트 이미지
 * @param kd 키포인트 데이터 (코너 좌표, 각도 정보 포함, 여기에 디스크립터 정보 추가)
 */
void computeDescriptors(const basalt::Image<const uint16_t>& img_raw, KeypointsData& kd) {
  kd.corner_descriptors.resize(kd.corners.size()); // 디스크립터 저장 공간 할당

  for (size_t i = 0; i < kd.corners.size(); i++) {
    std::bitset<256> descriptor; // 256비트 디스크립터

    const Eigen::Vector2d& p = kd.corners[i]; // 키포인트 좌표
    double angle = kd.corner_angles[i];       // 키포인트 각도

    int cx = p[0]; // 키포인트 중심 x
    int cy = p[1]; // 키포인트 중심 y

    Eigen::Rotation2Dd rot(angle);      // 각도를 이용한 2D 회전 행렬 생성
    Eigen::Matrix2d mat_rot = rot.matrix(); // 회전 행렬

    // 256개의 픽셀 쌍에 대해 반복하여 디스크립터 비트 계산
    for (int j = 0; j < 256; j++) { // 변수명 충돌을 피하기 위해 i -> j로 변경
      // 미리 정의된 패턴에서 두 점의 상대 좌표를 가져옴
      Eigen::Vector2d va(pattern_31_x_a[j], pattern_31_y_a[j]);
      Eigen::Vector2d vb(pattern_31_x_b[j], pattern_31_y_b[j]);

      // 키포인트의 각도에 따라 패턴 좌표를 회전시킴
      Eigen::Vector2i vva = (mat_rot * va).array().round().cast<int>();
      Eigen::Vector2i vvb = (mat_rot * vb).array().round().cast<int>();

      // 회전된 두 점의 픽셀 밝기를 비교하여 디스크립터 비트 설정
      // (첫 번째 점의 밝기 < 두 번째 점의 밝기) 이면 1, 아니면 0
      descriptor[j] = img_raw(cx + vva[0], cy + vva[1]) < img_raw(cx + vvb[0], cy + vvb[1]);
    }
    kd.corner_descriptors[i] = descriptor; // 계산된 디스크립터 저장
  }
}

/**
 * @brief 두 이미지의 키포인트 디스크립터들 간의 매칭을 수행하는 헬퍼 함수입니다.
 *        첫 번째 이미지의 각 디스크립터에 대해, 두 번째 이미지에서 가장 유사한 디스크립터(best_idx)와
 *        두 번째로 유사한 디스크립터를 찾습니다.
 *        최상의 매칭 거리가 임계값(`threshold`)보다 작고, 최상의 거리와 차순위 거리의 비율이
 *        `test_dist` (Lowe's ratio test의 역수 개념) 조건을 만족하면 매칭으로 간주합니다.
 * @param corner_descriptors_1 첫 번째 이미지의 디스크립터 집합
 * @param corner_descriptors_2 두 번째 이미지의 디스크립터 집합
 * @param matches 매칭 결과를 저장할 맵 (key: 첫 번째 이미지 디스크립터 인덱스, value: 두 번째 이미지 디스크립터 인덱스)
 * @param threshold 해밍 거리 임계값. 이 값보다 작은 경우에만 매칭 후보로 고려.
 * @param test_dist 최상의 매칭 거리와 차순위 매칭 거리 간의 비율 테스트에 사용되는 값 (best_dist * test_dist <= best2_dist). 일반적으로 1/0.7 또는 1/0.8.
 */
void matchFastHelper(const std::vector<std::bitset<256>>& corner_descriptors_1,
                     const std::vector<std::bitset<256>>& corner_descriptors_2, std::unordered_map<int, int>& matches,
                     int threshold, double test_dist) {
  matches.clear();

  // 첫 번째 이미지의 모든 디스크립터에 대해 반복
  for (size_t i = 0; i < corner_descriptors_1.size(); i++) {
    int best_idx = -1;        // 가장 일치하는 디스크립터의 인덱스 (두 번째 이미지에서)
    int best_dist = 500;      // 가장 작은 해밍 거리 (초기값은 충분히 큰 값)
    int best2_dist = 500;     // 두 번째로 작은 해밍 거리

    // 두 번째 이미지의 모든 디스크립터와 비교
    for (size_t j = 0; j < corner_descriptors_2.size(); j++) {
      // 해밍 거리 계산 (XOR 연산 후 1의 개수 카운트)
      int dist = (corner_descriptors_1[i] ^ corner_descriptors_2[j]).count();

      if (dist <= best_dist) { // 현재 거리가 기존 최단 거리보다 작거나 같으면
        best2_dist = best_dist; // 기존 최단 거리를 두 번째 최단 거리로 업데이트
        best_dist = dist;       // 현재 거리를 최단 거리로 업데이트
        best_idx = j;           // 최단 거리 디스크립터의 인덱스 업데이트
      } else if (dist < best2_dist) { // 현재 거리가 기존 두 번째 최단 거리보다 작으면
        best2_dist = dist;        // 두 번째 최단 거리 업데이트
      }
    }

    // Lowe's ratio test (변형된 형태) 및 거리 임계값 검사
    // 최단 거리가 임계값보다 작고, (최단 거리 * test_dist)가 두 번째 최단 거리보다 작거나 같으면 매칭으로 인정
    if (best_dist < threshold && best_dist * test_dist <= best2_dist) {
      matches.emplace(i, best_idx); // 매칭 결과 저장
    }
  }
}

/**
 * @brief 두 이미지의 키포인트 디스크립터들 간의 상호 매칭(cross-check)을 수행합니다.
 *        `matchFastHelper`를 양방향으로 호출 (1->2, 2->1)하여,
 *        서로가 서로를 가장 좋은 매칭 상대로 지목하는 경우에만 최종 매칭으로 확정합니다.
 * @param corner_descriptors_1 첫 번째 이미지의 디스크립터 집합
 * @param corner_descriptors_2 두 번째 이미지의 디스크립터 집합
 * @param matches 최종 매칭 결과를 저장할 벡터 (pair: <첫 번째 이미지 인덱스, 두 번째 이미지 인덱스>)
 * @param threshold 해밍 거리 임계값
 * @param dist_2_best `matchFastHelper`에 전달될 비율 테스트 값
 */
void matchDescriptors(const std::vector<std::bitset<256>>& corner_descriptors_1,
                      const std::vector<std::bitset<256>>& corner_descriptors_2,
                      std::vector<std::pair<int, int>>& matches, int threshold, double dist_2_best) {
  matches.clear();

  std::unordered_map<int, int> matches_1_2; // 이미지 1 -> 이미지 2 매칭 결과
  std::unordered_map<int, int> matches_2_1; // 이미지 2 -> 이미지 1 매칭 결과

  // 이미지 1에서 이미지 2로의 매칭 수행
  matchFastHelper(corner_descriptors_1, corner_descriptors_2, matches_1_2, threshold, dist_2_best);
  // 이미지 2에서 이미지 1로의 매칭 수행
  matchFastHelper(corner_descriptors_2, corner_descriptors_1, matches_2_1, threshold, dist_2_best);

  // 상호 검증 (Cross-check)
  // matches_1_2에 있는 각 매칭 (i, j)에 대해,
  // matches_2_1에서 j가 다시 i와 매칭되는지 확인
  for (const auto& kv : matches_1_2) {
    // kv.first는 desc1의 인덱스, kv.second는 desc2의 인덱스
    // matches_2_1[kv.second]는 desc2의 kv.second 인덱스에 매칭된 desc1의 인덱스
    if (matches_2_1.count(kv.second) && matches_2_1.at(kv.second) == kv.first) { // at() 사용으로 존재하지 않는 키 접근 방지
      matches.emplace_back(kv.first, kv.second); // 상호 매칭이 확인되면 최종 매칭 리스트에 추가
    }
  }
}

// OpenGV 라이브러리 의존성을 제거하기 위해 빌드 시 비활성화 가능 (더 빠른 빌드를 위함)
#ifndef BASALT_BUILD_SHARED_LIBRARY_ONLY
/**
 * @brief RANSAC 알고리즘을 사용하여 두 키포인트 집합 간의 매칭에서 인라이어(inlier)를 찾고,
 *        상대적인 자세(SE3 변환)를 추정합니다. OpenGV 라이브러리를 사용합니다.
 * @param kd1 첫 번째 이미지의 키포인트 데이터 (3D 좌표 포함)
 * @param kd2 두 번째 이미지의 키포인트 데이터 (3D 좌표 포함)
 * @param ransac_thresh RANSAC에서 인라이어로 판단하기 위한 임계값 (재투영 오차 등)
 * @param ransac_min_inliers 유효한 모델로 간주하기 위한 최소 인라이어 수
 * @param md 매칭 데이터 (입력: 초기 매칭, 출력: RANSAC 후 인라이어 매칭 및 추정된 변환 T_i_j)
 */
void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2, const double ransac_thresh,
                       const int ransac_min_inliers, MatchData& md) {
  md.inliers.clear(); // 인라이어 목록 초기화

  opengv::bearingVectors_t bearingVectors1, bearingVectors2; // OpenGV에서 사용하는 베어링 벡터 타입

  // 초기 매칭된 키포인트들의 3D 좌표(정규화된 이미지 평면 좌표, 즉 베어링 벡터)를 가져옴
  for (size_t i = 0; i < md.matches.size(); i++) {
    // .head<3>()은 (x,y,z) 중 (x,y)만 사용하거나, 정규화된 (x,y,1) 벡터를 의미할 수 있음.
    // 여기서는 카메라 좌표계에서의 3D 방향 벡터로 가정.
    bearingVectors1.push_back(kd1.corners_3d[md.matches[i].first].head<3>());
    bearingVectors2.push_back(kd2.corners_3d[md.matches[i].second].head<3>());
  }

  // OpenGV의 CentralRelativeAdapter 생성 (두 카메라 간의 상대 자세 추정을 위한 어댑터)
  opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors1, bearingVectors2);

  // RANSAC 객체 생성
  opengv::sac::Ransac<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;

  // CentralRelativePoseSacProblem 생성 (상대 자세 추정 문제 정의)
  // 알고리즘으로 STEWENIUS, NISTER, SEVENPT, EIGHTPT 등을 선택 가능
  std::shared_ptr<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> relposeproblem_ptr(
      new opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem(
          adapter, opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::STEWENIUS)); // 5점 알고리즘(Stewenius) 사용

  ransac.sac_model_ = relposeproblem_ptr; // RANSAC에 문제 모델 설정
  ransac.threshold_ = ransac_thresh;      // 인라이어 결정을 위한 임계값 설정
  ransac.max_iterations_ = 100;           // 최대 반복 횟수 설정
  ransac.computeModel();                  // RANSAC 실행하여 모델 추정

  // 비선형 최적화를 통한 모델 개선 및 추가 인라이어 확보
  const size_t num_inliers_ransac = ransac.inliers_.size(); // RANSAC 직후 인라이어 수

  // RANSAC으로 찾은 모델(회전 R, 변위 t)을 어댑터에 설정
  adapter.sett12(ransac.model_coefficients_.topRightCorner<3, 1>());      // t12: 카메라1에서 카메라2로의 변위
  adapter.setR12(ransac.model_coefficients_.topLeftCorner<3, 3>());       // R12: 카메라1에서 카메라2로의 회전

  // RANSAC 인라이어들을 사용하여 비선형 최적화 수행
  const opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  // 최적화된 모델을 사용하여 다시 인라이어들을 선택
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation, ransac.threshold_, ransac.inliers_);

  // 비선형 최적화 후 인라이어 수가 줄어들었는지 확인 (약간의 변동은 예상되므로, 2개 초과 감소 시 경고)
  if (ransac.inliers_.size() + 2 < num_inliers_ransac) {
    std::cout << "Warning: non-linear refinement reduced the relative pose "
                 "ransac inlier count from "
              << num_inliers_ransac << " to " << ransac.inliers_.size() << "." << std::endl;
  }

  // 최종 결과 (변환 행렬의 이동(translation) 부분을 정규화)
  // nonlinear_transformation은 4x4 행렬로, 왼쪽 위 3x3은 회전, 오른쪽 위 3x1은 이동 벡터
  md.T_i_j = Sophus::SE3d(nonlinear_transformation.topLeftCorner<3, 3>(),
                          nonlinear_transformation.topRightCorner<3, 1>().normalized()); // 이동 벡터를 정규화하여 SE(3) 변환 생성

  // 인라이어 수가 최소 요구 조건 이상이면, 인라이어 매칭을 저장
  if ((long)ransac.inliers_.size() >= ransac_min_inliers) {
    for (size_t i = 0; i < ransac.inliers_.size(); i++) {
      // ransac.inliers_[i]는 md.matches의 인덱스
      md.inliers.emplace_back(md.matches[ransac.inliers_[i]]);
    }
  }
}
#endif

}  // namespace basalt