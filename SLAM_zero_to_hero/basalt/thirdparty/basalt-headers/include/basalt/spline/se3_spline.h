/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

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

@file
@brief Uniform B-spline for SE(3)
*/

#pragma once

#include <basalt/spline/rd_spline.h>
#include <basalt/spline/so3_spline.h>
#include <basalt/utils/assert.h>

#include <basalt/calibration/calib_bias.hpp>

#include <array>

namespace basalt {

/// @brief Uniform B-spline for SE(3) of order N. Internally uses an SO(3) (\ref
/// So3Spline) spline for rotation and 3D Euclidean spline (\ref RdSpline) for
/// translation (split representaion).
///
/// See [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860) for more details.
template <int _N, typename _Scalar = double>
class Se3Spline {
 public:
  static constexpr int N = _N;        ///< Order of the spline.
  static constexpr int DEG = _N - 1;  ///< Degree of the spline.

  using MatN = Eigen::Matrix<_Scalar, _N, _N>;
  using VecN = Eigen::Matrix<_Scalar, _N, 1>;
  using VecNp1 = Eigen::Matrix<_Scalar, _N + 1, 1>;

  using Vec3 = Eigen::Matrix<_Scalar, 3, 1>;
  using Vec6 = Eigen::Matrix<_Scalar, 6, 1>;
  using Vec9 = Eigen::Matrix<_Scalar, 9, 1>;
  using Vec12 = Eigen::Matrix<_Scalar, 12, 1>;

  using Mat3 = Eigen::Matrix<_Scalar, 3, 3>;
  using Mat6 = Eigen::Matrix<_Scalar, 6, 6>;

  using Mat36 = Eigen::Matrix<_Scalar, 3, 6>;
  using Mat39 = Eigen::Matrix<_Scalar, 3, 9>;
  using Mat312 = Eigen::Matrix<_Scalar, 3, 12>;

  using Matrix3Array = std::array<Mat3, N>;
  using Matrix36Array = std::array<Mat36, N>;
  using Matrix6Array = std::array<Mat6, N>;

  using SO3 = Sophus::SO3<_Scalar>;
  using SE3 = Sophus::SE3<_Scalar>;

  using PosJacobianStruct = typename RdSpline<3, N, _Scalar>::JacobianStruct;
  using SO3JacobianStruct = typename So3Spline<N, _Scalar>::JacobianStruct;

  /// @brief Struct to store the accelerometer residual Jacobian with
  /// respect to knots
  struct AccelPosSO3JacobianStruct {
    size_t start_idx;
    std::array<Mat36, N> d_val_d_knot;
  };

  /// @brief Struct to store the pose Jacobian with respect to knots
  struct PosePosSO3JacobianStruct {
    size_t start_idx;
    std::array<Mat6, N> d_val_d_knot;
  };

  /// @brief Constructor with knot interval and start time
  ///
  /// @param[in] time_interval_ns knot time interval in nanoseconds
  /// @param[in] start_time_ns start time of the spline in nanoseconds
  Se3Spline(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : pos_spline_(time_interval_ns, start_time_ns),
        so3_spline_(time_interval_ns, start_time_ns),
        dt_ns_(time_interval_ns) {}

  /// @brief Gererate random trajectory
  ///
  /// @param[in] n number of knots to generate
  /// @param[in] static_init if true the first N knots will be the same
  /// resulting in static initial condition
  void genRandomTrajectory(int n, bool static_init = false) {
    so3_spline_.genRandomTrajectory(n, static_init);
    pos_spline_.genRandomTrajectory(n, static_init);
  }

  /// @brief Set the knot to particular SE(3) pose
  ///
  /// @param[in] pose SE(3) pose
  /// @param[in] i index of the knot
  void setKnot(const Sophus::SE3d &pose, int i) {
    so3_spline_.getKnot(i) = pose.so3();
    pos_spline_.getKnot(i) = pose.translation();
  }

  /// @brief Reset spline to have num_knots initialized at pose
  ///
  /// @param[in] pose SE(3) pose
  /// @param[in] num_knots number of knots to initialize
  void setKnots(const Sophus::SE3d &pose, int num_knots) {
    so3_spline_.resize(num_knots);
    pos_spline_.resize(num_knots);

    for (int i = 0; i < num_knots; i++) {
      so3_spline_.getKnot(i) = pose.so3();
      pos_spline_.getKnot(i) = pose.translation();
    }
  }

  /// @brief Reset spline to the knots from other spline
  ///
  /// @param[in] other spline to copy knots from
  void setKnots(const Se3Spline<N, _Scalar> &other) {
    BASALT_ASSERT(other.dt_ns_ == dt_ns_);
    BASALT_ASSERT(other.pos_spline_.getKnots().size() == other.pos_spline_.getKnots().size());

    size_t num_knots = other.pos_spline_.getKnots().size();

    so3_spline_.resize(num_knots);
    pos_spline_.resize(num_knots);

    for (size_t i = 0; i < num_knots; i++) {
      so3_spline_.getKnot(i) = other.so3_spline_.getKnot(i);
      pos_spline_.getKnot(i) = other.pos_spline_.getKnot(i);
    }
  }

  /// @brief Add knot to the end of the spline
  ///
  /// @param[in] knot knot to add
  inline void knotsPushBack(const SE3 &knot) {
    so3_spline_.knotsPushBack(knot.so3());
    pos_spline_.knotsPushBack(knot.translation());
  }

  /// @brief Remove knot from the back of the spline
  inline void knotsPopBack() {
    so3_spline_.knotsPopBack();
    pos_spline_.knotsPopBack();
  }

  /// @brief Return the first knot of the spline
  ///
  /// @return first knot of the spline
  inline SE3 knotsFront() const {
    SE3 res(so3_spline_.knots_front(), pos_spline_.knots_front());

    return res;
  }

  /// @brief Remove first knot of the spline and increase the start time
  inline void knotsPopFront() {
    so3_spline_.knots_pop_front();
    pos_spline_.knots_pop_front();

    BASALT_ASSERT(so3_spline_.minTimeNs() == pos_spline_.minTimeNs());
    BASALT_ASSERT(so3_spline_.getKnots().size() == pos_spline_.getKnots().size());
  }

  /// @brief Return the last knot of the spline
  ///
  /// @return last knot of the spline
  SE3 getLastKnot() {
    BASALT_ASSERT(so3_spline_.getKnots().size() == pos_spline_.getKnots().size());

    SE3 res(so3_spline_.getKnots().back(), pos_spline_.getKnots().back());

    return res;
  }

  /// @brief Return knot with index i
  ///
  /// @param i index of the knot
  /// @return knot
  SE3 getKnot(size_t i) const {
    SE3 res(getKnotSO3(i), getKnotPos(i));
    return res;
  }

  /// @brief Return reference to the SO(3) knot with index i
  ///
  /// @param i index of the knot
  /// @return reference to the SO(3) knot
  inline SO3 &getKnotSO3(size_t i) { return so3_spline_.getKnot(i); }

  /// @brief Return const reference to the SO(3) knot with index i
  ///
  /// @param i index of the knot
  /// @return const reference to the SO(3) knot
  inline const SO3 &getKnotSO3(size_t i) const { return so3_spline_.getKnot(i); }

  /// @brief Return reference to the position knot with index i
  ///
  /// @param i index of the knot
  /// @return reference to the position knot
  inline Vec3 &getKnotPos(size_t i) { return pos_spline_.getKnot(i); }

  /// @brief Return const reference to the position knot with index i
  ///
  /// @param i index of the knot
  /// @return const reference to the position knot
  inline const Vec3 &getKnotPos(size_t i) const { return pos_spline_.getKnot(i); }

  /// @brief Set start time for spline
  ///
  /// @param[in] start_time_ns start time of the spline in nanoseconds
  inline void setStartTimeNs(int64_t s) {
    so3_spline_.setStartTimeNs(s);
    pos_spline_.setStartTimeNs(s);
  }

  /// @brief Apply increment to the knot
  ///
  /// The incremernt vector consists of translational and rotational parts \f$
  /// [\upsilon, \omega]^T \f$. Given the current pose of the knot \f$ R \in
  /// SO(3), p \in \mathbb{R}^3\f$ the updated pose is: \f{align}{ R' &=
  /// \exp(\omega) R
  /// \\ p' &= p + \upsilon
  /// \f}
  ///  The increment is consistent with \ref
  /// PoseState::applyInc.
  ///
  /// @param[in] i index of the knot
  /// @param[in] inc 6x1 increment vector
  template <typename Derived>
  void applyInc(int i, const Eigen::MatrixBase<Derived> &inc) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

    pos_spline_.getKnot(i) += inc.template head<3>();
    so3_spline_.getKnot(i) = SO3::exp(inc.template tail<3>()) * so3_spline_.getKnot(i);
  }

  /// @brief Maximum time represented by spline
  ///
  /// @return maximum time represented by spline in nanoseconds
  int64_t maxTimeNs() const {
    BASALT_ASSERT_STREAM(
        so3_spline_.maxTimeNs() == pos_spline_.maxTimeNs(),
        "so3_spline.maxTimeNs() " << so3_spline_.maxTimeNs() << " pos_spline.maxTimeNs() " << pos_spline_.maxTimeNs());
    return pos_spline_.maxTimeNs();
  }

  /// @brief Minimum time represented by spline
  ///
  /// @return minimum time represented by spline in nanoseconds
  int64_t minTimeNs() const {
    BASALT_ASSERT_STREAM(
        so3_spline_.minTimeNs() == pos_spline_.minTimeNs(),
        "so3_spline.minTimeNs() " << so3_spline_.minTimeNs() << " pos_spline.minTimeNs() " << pos_spline_.minTimeNs());
    return pos_spline_.minTimeNs();
  }

  /// @brief Number of knots in the spline
  size_t numKnots() const { return pos_spline_.getKnots().size(); }

  /// @brief Linear acceleration in the world frame.
  ///
  /// @param[in] time_ns time to evaluate linear acceleration in nanoseconds
  inline Vec3 transAccelWorld(int64_t time_ns) const { return pos_spline_.acceleration(time_ns); }

  /// @brief Linear velocity in the world frame.
  ///
  /// @param[in] time_ns time to evaluate linear velocity in nanoseconds
  inline Vec3 transVelWorld(int64_t time_ns) const { return pos_spline_.velocity(time_ns); }

  /// @brief Rotational velocity in the body frame.
  ///
  /// @param[in] time_ns time to evaluate rotational velocity in nanoseconds
  inline Vec3 rotVelBody(int64_t time_ns) const { return so3_spline_.velocityBody(time_ns); }

  /// @brief Evaluate pose.
  ///
  /// @param[in] time_ns time to evaluate pose in nanoseconds
  /// @return SE(3) pose at time_ns
  SE3 pose(int64_t time_ns) const {
    SE3 res;

    res.so3() = so3_spline_.evaluate(time_ns);
    res.translation() = pos_spline_.evaluate(time_ns);

    return res;
  }

  /// @brief Evaluate pose and compute Jacobian.
  ///
  /// @param[in] time_ns time to evaluate pose in nanoseconds
  /// @param[out] J Jacobian of the pose with respect to knots
  /// @return SE(3) pose at time_ns
  Sophus::SE3d pose(int64_t time_ns, PosePosSO3JacobianStruct *J) const {
    Sophus::SE3d res;

    typename So3Spline<_N, _Scalar>::JacobianStruct Jr;
    typename RdSpline<3, N, _Scalar>::JacobianStruct Jp;

    res.so3() = so3_spline_.evaluate(time_ns, &Jr);
    res.translation() = pos_spline_.evaluate(time_ns, &Jp);

    if (J) {
      Eigen::Matrix3d RT = res.so3().inverse().matrix();

      J->start_idx = Jr.start_idx;
      for (int i = 0; i < N; i++) {
        J->d_val_d_knot[i].setZero();
        J->d_val_d_knot[i].template topLeftCorner<3, 3>() = RT * Jp.d_val_d_knot[i];
        J->d_val_d_knot[i].template bottomRightCorner<3, 3>() = RT * Jr.d_val_d_knot[i];
      }
    }

    return res;
  }

  /// @brief Evaluate pose and compute time Jacobian.
  ///
  /// @param[in] time_ns time to evaluate pose in nanoseconds
  /// @param[out] J Jacobian of the pose with time
  void d_pose_d_t(int64_t time_ns, Vec6 &J) const {
    J.template head<3>() = so3_spline_.evaluate(time_ns).inverse() * transVelWorld(time_ns);
    J.template tail<3>() = rotVelBody(time_ns);
  }

  /// @brief Evaluate gyroscope residual.
  ///
  /// @param[in] time_ns time of the measurement
  /// @param[in] measurement gyroscope measurement
  /// @param[in] gyro_bias_full gyroscope calibration
  /// @return gyroscope residual
  Vec3 gyroResidual(int64_t time_ns, const Vec3 &measurement, const CalibGyroBias<_Scalar> &gyro_bias_full) const {
    return so3_spline_.velocityBody(time_ns) - gyro_bias_full.getCalibrated(measurement);
  }

  /// @brief Evaluate gyroscope residual and compute Jacobians.
  ///
  /// @param[in] time_ns time of the measurement
  /// @param[in] measurement gyroscope measurement
  /// @param[in] gyro_bias_full gyroscope calibration
  /// @param[out] J_knots Jacobian with respect to SO(3) spline knots
  /// @param[out] J_bias Jacobian with respect to gyroscope calibration
  /// @return gyroscope residual
  Vec3 gyroResidual(int64_t time_ns, const Vec3 &measurement, const CalibGyroBias<_Scalar> &gyro_bias_full,
                    SO3JacobianStruct *J_knots, Mat312 *J_bias = nullptr) const {
    if (J_bias) {
      J_bias->setZero();
      J_bias->template block<3, 3>(0, 0).diagonal().array() = 1.0;
      J_bias->template block<3, 3>(0, 3).diagonal().array() = -measurement[0];
      J_bias->template block<3, 3>(0, 6).diagonal().array() = -measurement[1];
      J_bias->template block<3, 3>(0, 9).diagonal().array() = -measurement[2];
    }

    return so3_spline_.velocityBody(time_ns, J_knots) - gyro_bias_full.getCalibrated(measurement);
  }

  /// @brief Evaluate accelerometer residual.
  ///
  /// @param[in] time_ns time of the measurement
  /// @param[in] measurement accelerometer measurement
  /// @param[in] accel_bias_full accelerometer calibration
  /// @param[in] g gravity
  /// @return accelerometer residual
  Vec3 accelResidual(int64_t time_ns, const Eigen::Vector3d &measurement,
                     const CalibAccelBias<_Scalar> &accel_bias_full, const Eigen::Vector3d &g) const {
    Sophus::SO3d R = so3_spline_.evaluate(time_ns);
    Eigen::Vector3d accel_world = pos_spline_.acceleration(time_ns);

    return R.inverse() * (accel_world + g) - accel_bias_full.getCalibrated(measurement);
  }

  /// @brief Evaluate accelerometer residual and Jacobians.
  ///
  /// @param[in] time_ns time of the measurement
  /// @param[in] measurement accelerometer measurement
  /// @param[in] accel_bias_full accelerometer calibration
  /// @param[in] g gravity
  /// @param[out] J_knots Jacobian with respect to spline knots
  /// @param[out] J_bias Jacobian with respect to accelerometer calibration
  /// @param[out] J_g Jacobian with respect to gravity
  /// @return accelerometer residual
  Vec3 accelResidual(int64_t time_ns, const Vec3 &measurement, const CalibAccelBias<_Scalar> &accel_bias_full,
                     const Vec3 &g, AccelPosSO3JacobianStruct *J_knots, Mat39 *J_bias = nullptr,
                     Mat3 *J_g = nullptr) const {
    typename So3Spline<_N, _Scalar>::JacobianStruct Jr;
    typename RdSpline<3, N, _Scalar>::JacobianStruct Jp;

    Sophus::SO3d R = so3_spline_.evaluate(time_ns, &Jr);
    Eigen::Vector3d accel_world = pos_spline_.acceleration(time_ns, &Jp);

    Eigen::Matrix3d RT = R.inverse().matrix();
    Eigen::Matrix3d tmp = RT * Sophus::SO3d::hat(accel_world + g);

    BASALT_ASSERT_STREAM(Jr.start_idx == Jp.start_idx,
                         "Jr.start_idx " << Jr.start_idx << " Jp.start_idx " << Jp.start_idx);

    BASALT_ASSERT_STREAM(so3_spline_.getKnots().size() == pos_spline_.getKnots().size(),
                         "so3_spline.getKnots().size() " << so3_spline_.getKnots().size()
                                                         << " pos_spline.getKnots().size() "
                                                         << pos_spline_.getKnots().size());

    J_knots->start_idx = Jp.start_idx;
    for (int i = 0; i < N; i++) {
      J_knots->d_val_d_knot[i].template topLeftCorner<3, 3>() = RT * Jp.d_val_d_knot[i];
      J_knots->d_val_d_knot[i].template bottomRightCorner<3, 3>() = tmp * Jr.d_val_d_knot[i];
    }

    if (J_bias) {
      J_bias->setZero();
      J_bias->template block<3, 3>(0, 0).diagonal().array() = 1.0;
      J_bias->template block<3, 3>(0, 3).diagonal().array() = -measurement[0];
      (*J_bias)(1, 6) = -measurement[1];
      (*J_bias)(2, 7) = -measurement[1];
      (*J_bias)(2, 8) = -measurement[2];
    }
    if (J_g) {
      (*J_g) = RT;
    }

    Vec3 res = RT * (accel_world + g) - accel_bias_full.getCalibrated(measurement);

    return res;
  }

  /// @brief Evaluate position residual.
  ///
  /// @param[in] time_ns time of the measurement
  /// @param[in] measured_position position measurement
  /// @param[out] Jp if not nullptr, Jacobian with respect to knos of the
  /// position spline
  /// @return position residual
  Sophus::Vector3d positionResidual(int64_t time_ns, const Vec3 &measured_position,
                                    PosJacobianStruct *Jp = nullptr) const {
    return pos_spline_.evaluate(time_ns, Jp) - measured_position;
  }

  /// @brief Evaluate orientation residual.
  ///
  /// @param[in] time_ns time of the measurement
  /// @param[in] measured_orientation orientation measurement
  /// @param[out] Jr if not nullptr, Jacobian with respect to knos of the
  /// SO(3) spline
  /// @return orientation residual
  Sophus::Vector3d orientationResidual(int64_t time_ns, const SO3 &measured_orientation,
                                       SO3JacobianStruct *Jr = nullptr) const {
    Sophus::Vector3d res = (so3_spline_.evaluate(time_ns, Jr) * measured_orientation.inverse()).log();

    if (Jr) {
      Eigen::Matrix3d Jrot = Sophus::leftJacobianSO3(res);
      for (int i = 0; i < N; i++) {
        Jr->d_val_d_knot[i] = Jrot * Jr->d_val_d_knot[i];
      }
    }

    return res;
  }

  /// @brief Print knots for debugging.
  inline void printKnots() const {
    for (size_t i = 0; i < pos_spline_.getKnots().size(); i++) {
      std::cout << i << ": p:" << pos_spline_.getKnot(i).transpose()
                << " q: " << so3_spline_.getKnot(i).unit_quaternion().coeffs().transpose() << std::endl;
    }
  }

  /// @brief Print position knots for debugging.
  inline void printPosKnots() const {
    for (size_t i = 0; i < pos_spline_.getKnots().size(); i++) {
      std::cout << pos_spline_.getKnot(i).transpose() << std::endl;
    }
  }

  /// @brief Knot time interval in nanoseconds.
  inline int64_t getDtNs() const { return dt_ns_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  RdSpline<3, _N, _Scalar> pos_spline_;  ///< Position spline
  So3Spline<_N, _Scalar> so3_spline_;    ///< Orientation spline

  int64_t dt_ns_;  ///< Knot interval in nanoseconds
};

}  // namespace basalt
