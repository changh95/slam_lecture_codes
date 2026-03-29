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
@brief Implementation of field-of-view camera model
*/

#pragma once

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

using std::sqrt;

/// @brief Field-of-View camera model
///
/// \image html fov.png
/// This model has N=5 parameters \f$ \mathbf{i} = \left[f_x, f_y, c_x, c_y,
/// w \right]^T \f$. See \ref project and \ref unproject
/// functions for more details.
template <typename Scalar_ = double>
class FovCamera {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 5;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  using Mat44 = Eigen::Matrix<Scalar, 4, 4>;

  /// @brief Default constructor with zero intrinsics
  FovCamera() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, w]
  explicit FovCamera(const VecN& p) { param_ = p; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  FovCamera<Scalar2> cast() const {
    return FovCamera<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "fov"
  static std::string getName() { return "fov"; }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// Projection function is defined as follows:
  /// \f{align}{
  ///  \DeclareMathOperator{\atantwo}{atan2}
  ///  \pi(\mathbf{x}, \mathbf{i}) &=
  ///  \begin{bmatrix} f_x r_d  \frac{x} { r_u }
  ///  \\ f_y r_d \frac{y} { r_u }
  ///  \\ \end{bmatrix}
  ///  +
  ///  \begin{bmatrix}
  ///  c_x
  ///  \\ c_y
  ///  \\ \end{bmatrix},
  ///  \\ r_u &= \sqrt{x^2 + y^2},
  ///  \\ r_d &= \frac{\atantwo(2 r_u \tan{\frac{w}{2}}, z)}{w}.
  /// \f}
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @param[out] d_proj_d_param point if not nullptr computed Jacobian of
  /// projection with respect to intrinsic parameters
  /// @return if projection is valid
  template <class DerivedPoint3D, class DerivedPoint2D, class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D>& p3d, Eigen::MatrixBase<DerivedPoint2D>& proj,
                      DerivedJ3D d_proj_d_p3d = nullptr, DerivedJparam d_proj_d_param = nullptr) const {
    checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D, DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];
    const Scalar& w = param_[4];

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    Scalar r2 = x * x + y * y;
    Scalar r = sqrt(r2);

    Scalar z2 = z * z;

    const Scalar tanwhalf = std::tan(w / 2);
    const Scalar atan_wrd = std::atan2(2 * tanwhalf * r, z);

    Scalar rd = Scalar(1);
    Scalar d_rd_d_w = Scalar(0);
    Scalar d_rd_d_x = Scalar(0);
    Scalar d_rd_d_y = Scalar(0);
    Scalar d_rd_d_z = Scalar(0);

    Scalar tmp1 = Scalar(1) / std::cos(w / 2);
    Scalar d_tanwhalf_d_w = Scalar(0.5) * tmp1 * tmp1;
    Scalar tmp = (z2 + Scalar(4) * tanwhalf * tanwhalf * r2);
    Scalar d_atan_wrd_d_w = Scalar(2) * r * d_tanwhalf_d_w * z / tmp;

    bool is_valid = true;
    if (w > Sophus::Constants<Scalar>::epsilonSqrt()) {
      if (r2 < Sophus::Constants<Scalar>::epsilonSqrt()) {
        if (z < Sophus::Constants<Scalar>::epsilonSqrt()) {
          is_valid = false;
        }

        rd = Scalar(2) * tanwhalf / w;
        d_rd_d_w = Scalar(2) * (d_tanwhalf_d_w * w - tanwhalf) / (w * w);
      } else {
        rd = atan_wrd / (r * w);
        d_rd_d_w = (d_atan_wrd_d_w * w - atan_wrd) / (r * w * w);

        const Scalar d_r_d_x = x / r;
        const Scalar d_r_d_y = y / r;

        const Scalar d_atan_wrd_d_x = Scalar(2) * tanwhalf * d_r_d_x * z / tmp;
        const Scalar d_atan_wrd_d_y = Scalar(2) * tanwhalf * d_r_d_y * z / tmp;
        const Scalar d_atan_wrd_d_z = -Scalar(2) * tanwhalf * r / tmp;

        d_rd_d_x = (d_atan_wrd_d_x * r - d_r_d_x * atan_wrd) / (r * r * w);
        d_rd_d_y = (d_atan_wrd_d_y * r - d_r_d_y * atan_wrd) / (r * r * w);
        d_rd_d_z = d_atan_wrd_d_z / (r * w);
      }
    }

    const Scalar mx = x * rd;
    const Scalar my = y * rd;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_p3d);
      d_proj_d_p3d->setZero();
      (*d_proj_d_p3d)(0, 0) = fx * (d_rd_d_x * x + rd);
      (*d_proj_d_p3d)(0, 1) = fx * d_rd_d_y * x;
      (*d_proj_d_p3d)(0, 2) = fx * d_rd_d_z * x;

      (*d_proj_d_p3d)(1, 0) = fy * d_rd_d_x * y;
      (*d_proj_d_p3d)(1, 1) = fy * (d_rd_d_y * y + rd);
      (*d_proj_d_p3d)(1, 2) = fy * d_rd_d_z * y;
    } else {
      UNUSED(d_proj_d_p3d);
      UNUSED(d_rd_d_x);
      UNUSED(d_rd_d_y);
      UNUSED(d_rd_d_z);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_param);

      d_proj_d_param->setZero();
      (*d_proj_d_param)(0, 0) = mx;
      (*d_proj_d_param)(0, 2) = Scalar(1);
      (*d_proj_d_param)(1, 1) = my;
      (*d_proj_d_param)(1, 3) = Scalar(1);

      (*d_proj_d_param)(0, 4) = fx * x * d_rd_d_w;
      (*d_proj_d_param)(1, 4) = fy * y * d_rd_d_w;
    } else {
      UNUSED(d_proj_d_param);
      UNUSED(d_rd_d_w);
    }

    return is_valid;
  }

  /// @brief Unproject the point and optionally compute Jacobians
  ///
  /// The unprojection function is computed as follows: \f{align}{
  ///     \pi^{-1}(\mathbf{u}, \mathbf{i}) &=
  ///  \begin{bmatrix}
  ///  m_x \frac{\sin(r_d w)}{ 2 r_d \tan{\frac{w}{2}}}
  ///  \\ m_y \frac{\sin(r_d w)}{ 2 r_d \tan{\frac{w}{2}}}
  ///  \\ \cos(r_d w)
  ///  \\ \end{bmatrix},
  ///  \\ m_x &= \frac{u - c_x}{f_x},
  ///  \\ m_y &= \frac{v - c_y}{f_y},
  ///  \\ r_d &= \sqrt{m_x^2 + m_y^2}.
  /// \f}
  ///
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj if not nullptr computed Jacobian of unprojection
  /// with respect to proj
  /// @param[out] d_p3d_d_param point if not nullptr computed Jacobian of
  /// unprojection with respect to intrinsic parameters
  /// @return if unprojection is valid
  template <class DerivedPoint2D, class DerivedPoint3D, class DerivedJ2D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool unproject(const Eigen::MatrixBase<DerivedPoint2D>& proj, Eigen::MatrixBase<DerivedPoint3D>& p3d,
                        DerivedJ2D d_p3d_d_proj = nullptr, DerivedJparam d_p3d_d_param = nullptr) const {
    checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D, DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint2D>::Type proj_eval(proj);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];
    const Scalar& w = param_[4];

    const Scalar tan_w_2 = std::tan(w / Scalar(2));
    const Scalar mul2tanwby2 = tan_w_2 * Scalar(2);

    const Scalar mx = (proj_eval[0] - cx) / fx;
    const Scalar my = (proj_eval[1] - cy) / fy;

    const Scalar rd = sqrt(mx * mx + my * my);

    Scalar ru = Scalar(1);
    Scalar sin_rd_w = Scalar(0);
    Scalar cos_rd_w = Scalar(1);

    Scalar d_ru_d_rd = Scalar(0);

    Scalar rd_inv = Scalar(1);

    if (mul2tanwby2 > Sophus::Constants<Scalar>::epsilonSqrt() && rd > Sophus::Constants<Scalar>::epsilonSqrt()) {
      sin_rd_w = std::sin(rd * w);
      cos_rd_w = std::cos(rd * w);
      ru = sin_rd_w / (rd * mul2tanwby2);

      rd_inv = Scalar(1) / rd;

      d_ru_d_rd = (w * cos_rd_w * rd - sin_rd_w) * rd_inv * rd_inv / mul2tanwby2;
    }

    p3d.setZero();
    p3d[0] = mx * ru;
    p3d[1] = my * ru;
    p3d[2] = cos_rd_w;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
      constexpr int SIZE_3D = DerivedPoint3D::SizeAtCompileTime;
      Eigen::Matrix<Scalar, SIZE_3D, 1> c0, c1;

      c0.setZero();
      c0(0) = (ru + mx * d_ru_d_rd * mx * rd_inv) / fx;
      c0(1) = my * d_ru_d_rd * mx * rd_inv / fx;
      c0(2) = -sin_rd_w * w * mx * rd_inv / fx;

      c1.setZero();
      c1(0) = my * d_ru_d_rd * mx * rd_inv / fy;
      c1(1) = (ru + my * d_ru_d_rd * my * rd_inv) / fy;
      c1(2) = -sin_rd_w * w * my * rd_inv / fy;

      if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
        BASALT_ASSERT(d_p3d_d_proj);
        d_p3d_d_proj->col(0) = c0;
        d_p3d_d_proj->col(1) = c1;
      } else {
        UNUSED(d_p3d_d_proj);
      }

      if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
        BASALT_ASSERT(d_p3d_d_param);
        d_p3d_d_param->setZero();

        d_p3d_d_param->col(2) = -c0;
        d_p3d_d_param->col(3) = -c1;

        d_p3d_d_param->col(0) = -c0 * mx;
        d_p3d_d_param->col(1) = -c1 * my;

        Scalar tmp = (cos_rd_w - (tan_w_2 * tan_w_2 + Scalar(1)) * sin_rd_w * rd_inv / (2 * tan_w_2)) / mul2tanwby2;

        (*d_p3d_d_param)(0, 4) = mx * tmp;
        (*d_p3d_d_param)(1, 4) = my * tmp;
        (*d_p3d_d_param)(2, 4) = -sin_rd_w * rd;
      } else {
        UNUSED(d_p3d_d_param);
        UNUSED(d_ru_d_rd);
      }

      Scalar norm = p3d.norm();
      Scalar norm_inv = Scalar(1) / norm;
      Scalar norm_inv2 = norm_inv * norm_inv;
      Scalar norm_inv3 = norm_inv2 * norm_inv;

      Eigen::Matrix<Scalar, SIZE_3D, SIZE_3D> d_p_norm_d_p;
      d_p_norm_d_p.setZero();

      d_p_norm_d_p(0, 0) = norm_inv * (Scalar(1) - p3d[0] * p3d[0] * norm_inv2);
      d_p_norm_d_p(1, 0) = -p3d[1] * p3d[0] * norm_inv3;
      d_p_norm_d_p(2, 0) = -p3d[2] * p3d[0] * norm_inv3;

      d_p_norm_d_p(0, 1) = -p3d[1] * p3d[0] * norm_inv3;
      d_p_norm_d_p(1, 1) = norm_inv * (Scalar(1) - p3d[1] * p3d[1] * norm_inv2);
      d_p_norm_d_p(2, 1) = -p3d[1] * p3d[2] * norm_inv3;

      d_p_norm_d_p(0, 2) = -p3d[2] * p3d[0] * norm_inv3;
      d_p_norm_d_p(1, 2) = -p3d[2] * p3d[1] * norm_inv3;
      d_p_norm_d_p(2, 2) = norm_inv * (Scalar(1) - p3d[2] * p3d[2] * norm_inv2);

      if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
        (*d_p3d_d_proj) = d_p_norm_d_p * (*d_p3d_d_proj);
      }
      if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
        (*d_p3d_d_param) = d_p_norm_d_p * (*d_p3d_d_param);
      }
    } else {
      UNUSED(d_p3d_d_proj);
      UNUSED(d_p3d_d_param);
      UNUSED(d_ru_d_rd);
    }

    p3d /= p3d.norm();

    return true;
  }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[f_x, f_y, c_x, c_y, 1
  /// \right]^T \f$
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init) {
    param_[0] = init[0];
    param_[1] = init[1];
    param_[2] = init[2];
    param_[3] = init[3];
    param_[4] = 1;
  }

  /// @brief Increment intrinsic parameters by inc
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) { param_ += inc; }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f_x, f_y, c_x, c_y, k1, k2, k3, k4
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<FovCamera> getTestProjections() {
    Eigen::aligned_vector<FovCamera> res;

    VecN vec1;

    // Euroc
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.9259487501905697;
    res.emplace_back(vec1);

    return res;
  }

  /// @brief Resolutions used for unit-tests
  static Eigen::aligned_vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::aligned_vector<Eigen::Vector2i> res;

    res.emplace_back(752, 480);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;
};

}  // namespace basalt
