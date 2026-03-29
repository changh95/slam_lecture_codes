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
@brief Type definitions for IMU preintegration
*/

#pragma once

#include <basalt/utils/sophus_utils.hpp>
#include <memory>

namespace basalt {

constexpr size_t POSE_SIZE = 6;            ///< Dimentionality of the pose state
constexpr size_t POSE_VEL_SIZE = 9;        ///< Dimentionality of the pose-velocity state
constexpr size_t POSE_VEL_BIAS_SIZE = 15;  ///< Dimentionality of the pose-velocity-bias state

/// @brief State that consists of SE(3) pose at a certain time.
template <class Scalar_>
struct PoseState {
  using Scalar = Scalar_;
  using VecN = Eigen::Matrix<Scalar, POSE_SIZE, 1>;
  using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
  using SO3 = Sophus::SO3<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;

  /// @brief Default constructor with Identity pose and zero timestamp.
  PoseState() { t_ns = 0; }

  /// @brief Constructor with timestamp and pose.
  ///
  /// @param t_ns timestamp of the state in nanoseconds
  /// @param T_w_i transformation from the body frame to the world frame
  PoseState(int64_t t_ns, const SE3& T_w_i) : t_ns(t_ns), T_w_i(T_w_i) {}

  /// @brief Create copy with different Scalar type.
  template <class Scalar2>
  PoseState<Scalar2> cast() const {
    PoseState<Scalar2> a;
    a.t_ns = t_ns;
    a.T_w_i = T_w_i.template cast<Scalar2>();
    return a;
  }

  /// @brief Apply increment to the pose
  ///
  /// For details see \ref incPose
  /// @param[in] inc 6x1 increment vector
  void applyInc(const VecN& inc) { incPose(inc, T_w_i); }

  /// @brief Apply increment to the pose
  ///
  /// The incremernt vector consists of translational and rotational parts \f$
  /// [\upsilon, \omega]^T \f$. Given the current pose \f$ R \in
  /// SO(3), p \in \mathbb{R}^3\f$ the updated pose is: \f{align}{ R' &=
  /// \exp(\omega) R
  /// \\ p' &= p + \upsilon
  /// \f}
  ///  The increment is consistent with \ref
  /// Se3Spline::applyInc.
  ///
  /// @param[in] inc 6x1 increment vector
  /// @param[in,out] T the pose to update
  inline static void incPose(const Vec6& inc, SE3& T) {
    T.translation() += inc.template head<3>();
    T.so3() = SO3::exp(inc.template tail<3>()) * T.so3();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int64_t t_ns;  ///< timestamp of the state in nanoseconds
  SE3 T_w_i;     ///< pose of the state
};

/// @brief State that consists of SE(3) pose and linear velocity at a certain
/// time.
template <class Scalar_>
struct PoseVelState : public PoseState<Scalar_> {
  using Scalar = Scalar_;
  using VecN = Eigen::Matrix<Scalar, POSE_VEL_SIZE, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using SE3 = Sophus::SE3<Scalar>;

  /// @brief Default constructor with Identity pose and zero other values.
  PoseVelState() { vel_w_i.setZero(); };

  /// @brief Constructor with timestamp, pose and linear velocity.
  ///
  /// @param t_ns timestamp of the state in nanoseconds
  /// @param T_w_i transformation from the body frame to the world frame
  /// @param vel_w_i linear velocity in world coordinate frame
  PoseVelState(int64_t t_ns, const SE3& T_w_i, const Vec3& vel_w_i)
      : PoseState<Scalar>(t_ns, T_w_i), vel_w_i(vel_w_i) {}

  /// @brief Create copy with different Scalar type.
  template <class Scalar2>
  PoseVelState<Scalar2> cast() const {
    PoseVelState<Scalar2> a;
    static_cast<PoseState<Scalar2>&>(a) = PoseState<Scalar>::template cast<Scalar2>();
    a.vel_w_i = vel_w_i.template cast<Scalar2>();
    return a;
  }

  /// @brief Apply increment to the state.
  ///
  /// For pose see \ref incPose. For velocity simple addition.
  /// @param[in] inc 9x1 increment vector [trans, rot, vel]
  void applyInc(const VecN& inc) {
    PoseState<Scalar>::applyInc(inc.template head<6>());
    vel_w_i += inc.template tail<3>();
  }

  /// @brief Compute difference to other state.
  ///
  /// ```
  ///      PoseVelState::VecN inc;
  ///      PoseVelState p0, p1;
  ///      // Initialize p0 and inc
  ///      p1 = p0;
  ///      p1.applyInc(inc);
  ///      p0.diff(p1) == inc; // Should be true.
  /// ```
  /// @param other state to compute difference.
  VecN diff(const PoseVelState<Scalar>& other) const {
    VecN res;
    res.template segment<3>(0) = other.T_w_i.translation() - this->T_w_i.translation();
    res.template segment<3>(3) = (other.T_w_i.so3() * this->T_w_i.so3().inverse()).log();
    res.template tail<3>() = other.vel_w_i - vel_w_i;
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec3 vel_w_i;  ///< Linear velocity of the state
};

/// @brief State that consists of SE(3) pose, linear velocity, gyroscope and
/// accelerometer biases at a certain time.
template <class Scalar_>
struct PoseVelBiasState : public PoseVelState<Scalar_> {
  using Scalar = Scalar_;
  using Ptr = std::shared_ptr<PoseVelBiasState<Scalar>>;
  using VecN = Eigen::Matrix<Scalar, POSE_VEL_BIAS_SIZE, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using SO3 = Sophus::SO3<Scalar>;
  using SE3 = Sophus::SE3<Scalar>;

  /// @brief Default constructor with Identity pose and zero other values.
  PoseVelBiasState() {
    bias_gyro.setZero();
    bias_accel.setZero();
  };

  /// @brief Constructor with timestamp, pose, linear velocity, gyroscope and
  /// accelerometer biases.
  ///
  /// @param t_ns timestamp of the state in nanoseconds
  /// @param T_w_i transformation from the body frame to the world frame
  /// @param vel_w_i linear velocity in world coordinate frame
  /// @param bias_gyro gyroscope bias
  /// @param bias_accel accelerometer bias
  PoseVelBiasState(int64_t t_ns, const SE3& T_w_i, const Vec3& vel_w_i, const Vec3& bias_gyro, const Vec3& bias_accel)
      : PoseVelState<Scalar>(t_ns, T_w_i, vel_w_i), bias_gyro(bias_gyro), bias_accel(bias_accel) {}

  /// @brief Create copy with different Scalar type.
  template <class Scalar2>
  PoseVelBiasState<Scalar2> cast() const {
    PoseVelBiasState<Scalar2> a;
    static_cast<PoseVelState<Scalar2>&>(a) = PoseVelState<Scalar>::template cast<Scalar2>();
    a.bias_gyro = bias_gyro.template cast<Scalar2>();
    a.bias_accel = bias_accel.template cast<Scalar2>();
    return a;
  }

  /// @brief Apply increment to the state.
  ///
  /// For pose see \ref incPose. For velocity and biases simple addition.
  /// @param[in] inc 15x1 increment vector [trans, rot, vel, bias_gyro,
  /// bias_accel]
  void applyInc(const VecN& inc) {
    PoseVelState<Scalar>::applyInc(inc.template head<9>());
    bias_gyro += inc.template segment<3>(9);
    bias_accel += inc.template segment<3>(12);
  }

  /// @brief Compute difference to other state.
  ///
  /// ```
  ///      PoseVelBiasState::VecN inc;
  ///      PoseVelBiasState p0, p1;
  ///      // Initialize p0 and inc
  ///      p1 = p0;
  ///      p1.applyInc(inc);
  ///      p0.diff(p1) == inc; // Should be true.
  /// ```
  /// @param other state to compute difference.
  VecN diff(const PoseVelBiasState<Scalar>& other) const {
    VecN res;
    res.template segment<9>(0) = PoseVelState<Scalar>::diff(other);
    res.template segment<3>(9) = other.bias_gyro - bias_gyro;
    res.template segment<3>(12) = other.bias_accel - bias_accel;
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec3 bias_gyro;   ///< Gyroscope bias
  Vec3 bias_accel;  ///< Accelerometer bias
  std::shared_ptr<struct OpticalFlowInput> input_images{};
};

/// @brief Timestamped gyroscope and accelerometer measurements.
template <class Scalar_>
struct ImuData {
  using Scalar = Scalar_;
  using Ptr = std::shared_ptr<ImuData>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  int64_t t_ns;  ///< timestamp in nanoseconds
  Vec3 accel;    ///< Accelerometer measurement
  Vec3 gyro;     ///< Gyroscope measurement

  /// @brief Default constructor with zero measurements.
  ImuData() {
    t_ns = 0;
    accel.setZero();
    gyro.setZero();
  }

  /// @brief Create copy with different Scalar type.
  template <class Scalar2>
  ImuData<Scalar2> cast() const {
    ImuData<Scalar2> a;
    a.t_ns = t_ns;
    a.accel = accel.template cast<Scalar2>();
    a.gyro = gyro.template cast<Scalar2>();
    return a;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace basalt
