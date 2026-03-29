/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2021-2022, Collabora Ltd.
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
@brief Implementation of pinhole camera model with radial-tangential distortion
@author Mateo de Mayo <mateo.demayo@collabora.com>
*/

#pragma once

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

// clang-format off
// Scalar might not be a literal type (e.g., when Scalar = ceres::Jet), so we
// cannot use constexpr. These are undefined at the end of the file.
#define SN1 Scalar{-1}
#define S0 Scalar{0}
#define S1 Scalar{1}
#define S2 Scalar{2}
#define S3 Scalar{3}
// clang-format on

namespace basalt {

using std::abs;
using std::max;
using std::sqrt;

/// @brief Pinhole camera model with radial-tangential distortion
///
/// This model has N=12 parameters with \f$\mathbf{i} = \left[
/// f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6
/// \right]^T \f$ and an optional $r'_{max}$ to limit the valid projection
/// domain. See \ref project and \ref unproject functions for more details.
template <typename Scalar_ = double>
class PinholeRadtan8Camera {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 12;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  PinholeRadtan8Camera() {
    param_.setZero();
    rpmax_ = S0;
  }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, k1, k2, p1,
  /// p2, k3, k4, k5, k6]
  /// @param[in] rpmax Optional. Radius of a circle that approximates the valid
  /// projection domain area.
  /// If -1, one will be estimated with @ref computeRpmax().
  explicit PinholeRadtan8Camera(const VecN& p, Scalar rpmax = SN1) {
    param_ = p;
    rpmax_ = rpmax == SN1 ? computeRpmax() : rpmax;
  }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  PinholeRadtan8Camera<Scalar2> cast() const {
    return PinholeRadtan8Camera<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "pinhole-radtan8"
  static std::string getName() { return "pinhole-radtan8"; }

  /// @brief Computes an estimate for the \f$ r'_{max} \f$ value for this camera
  /// if not provided.
  ///
  /// Some radtan8 calibrations are not injective; in particular, once you
  /// start to go too far away in 3D space, the projection will fold back into
  /// the image center. To avoid this issue we can approximate an area in Z=1
  /// for which the calibration is injective and discard points outside of this
  /// area. Our approximated area will be a circle with radius `rpmax`.
  /// This phenomena is better explained in
  /// "On the Maximum Radius of Polynomial Lens Distortion"
  /// https://doi.org/10.1109/WACV51458.2022.00243
  /// This function generalizes the core ideas of that paper to estimate rpmax.
  /// Note that we are making some assumptions, see comments and asserts.
  ///
  /// @return 0 if the model is injective in all its domain, >0 otherwise.
  Scalar computeRpmax() {
    // We want project/unproject to succeed in this scope so we set rpmax_ = 0
    Scalar rpmax_backup = rpmax_;
    rpmax_ = S0;

    // Good enough constants for the tested calibrations
    const int MAX_ITERS{1000};             // Gradient ascent (GA) max iterations
    const Scalar STEP_SIZE{0.1};           // GA fixed "learning rate"
    const Scalar MIN_REL_STEP{0.0001};     // GA minimum step (relative) size
    const Scalar NUDGE{0.1};               // Image center offset for the GA first guess
    const Scalar CORNER_BOUND_SCALE{1.5};  // Divergence bounds scaler
    const Scalar RPMAX_SCALE{0.85};        // Shrink the resulting circle to be safe

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    // Construct our initial guess near the image center
    Vec2 almost_central_pixel{NUDGE * fx + cx, NUDGE * fy + cy};
    Vec3 acp_uproj;
    bool unproject_success = unproject(almost_central_pixel, acp_uproj);
    BASALT_ASSERT(unproject_success);
    acp_uproj /= acp_uproj.z();
    Vec2 guess{acp_uproj.x(), acp_uproj.y()};

    // rpp2(x, y) = How far from the image center does (x, y, 1) project into?
    // If we get far from (0, 0, 1) we, initially, also get far from the image
    // center. Once that's not true then we have surpassed the injective area.
    // This is the local maximum to optimize with gradient ascent.
    auto rpp2 = [this](Vec2 xy) {
      Vec2 xypp;
      distort(xy, xypp);
      return xypp.squaredNorm();
    };

    // Numeric derivative of rpp2 based on scipy's `approx_derivative`
    auto numeric_rpp2_jacobian = [&rpp2](Vec2 xy) {
      const Scalar eps = Sophus::Constants<Scalar>::epsilonSqrt();
      const auto sign = [](Scalar n) { return n < S0 ? SN1 : S1; };
      const Scalar hx = eps * sign(xy.x()) * max(S1, xy.x());
      const Scalar hy = eps * sign(xy.y()) * max(S1, xy.y());

      const auto& f = rpp2;
      const Scalar f_xy = f(xy);
      const Scalar df_dx = (f(xy + Vec2{hx, S0}) - f_xy) / hx;
      const Scalar df_dy = (f(xy + Vec2{S0, hy}) - f_xy) / hy;
      return Vec2{df_dx, df_dy};
    };

    // Compute a bound for rpp2 domain: the max rp2 from unprojected corners

    // rp2(u, v) = Squared norm in Z=1 of unprojected point uv
    auto rp2 = [this](Vec2 uv) {
      Vec3 xyz;
      unproject(uv, xyz);
      xyz /= xyz.z();  // To z=1
      return xyz.x() * xyz.x() + xyz.y() * xyz.y();
    };

    // Compute optimization bound.
    // We are assuming that the valid projection area bounds are inside of where
    // corners unproject, in particular close by at most CORNER_BOUND_SCALE
    // times that distance.

    // Approximations for width/height. We don't have access to the actual image
    // size at this point
    const Scalar w = S2 * cx;
    const Scalar h = S2 * cy;
    const std::vector<Vec2> corners = {{S0, S0}, {w, S0}, {S0, h}, {w, h}};
    Scalar corners_maxrp2{-1};
    for (const Vec2& uv : corners) {
      const Scalar corner_rp2 = rp2(uv);
      if (corner_rp2 > corners_maxrp2) {
        corners_maxrp2 = corner_rp2;
      }
    }
    const Scalar domain_bound = corners_maxrp2 * CORNER_BOUND_SCALE;

    // Gradient ascent.
    // Consider that in reality we would like to find all the critical points of
    // rpp2(x, y, z=1) starting our optimization from (0, 0, 1). And from those,
    // our solution would be the point (x, y) with the smallest length (the
    // length being the selected rpmax). As a rough approximation we find
    // instead a local maximum with GA and shrink it with RPMAX_SCALE. This
    // tends to be enough because tangential distortion is usually small.

    //! @todo Implement a better numerical algorithm in which we sample 8 or 16
    //! directions around (0, 0) using Newton's method to find critical points
    //! and we keep with the one with the lowest norm.

    Vec2 x = guess;
    Scalar rpp2_x = rpp2(x);
    bool diverged = false;
    for (int i = 1; i < MAX_ITERS; i++) {
      x += STEP_SIZE * numeric_rpp2_jacobian(x);

      const Scalar rp2_x = x.squaredNorm();
      if (rp2_x > domain_bound) {
        diverged = true;
        break;
      }

      const Scalar old_rpp2_x = rpp2_x;
      rpp2_x = rpp2(x);
      if (abs(rpp2_x - old_rpp2_x) < MIN_REL_STEP * old_rpp2_x) {
        break;
      }
    }

    // Finally, this is our rpmax estimate
    const Scalar rpmax = diverged ? S0 : RPMAX_SCALE * x.norm();

    rpmax_ = rpmax_backup;
    return rpmax;
  }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// Projection function is defined as follows:
  /// \f{align}{
  ///   \pi(\mathbf{x}, \mathbf{i}) &=
  ///   \begin{bmatrix}
  ///     f_x x'' + c_x
  /// \\  f_y y'' + c_y
  /// \\\end{bmatrix}
  /// \newline
  ///
  /// \\\begin{bmatrix}
  ///     x''
  /// \\  y''
  ///   \end{bmatrix} &=
  ///   \begin{bmatrix}
  ///     x' d + 2 p_1 x' y' + p_2 (r'^2 + 2 x'^2)
  /// \\  y' d + 2 p_2 x' y' + p_1 (r'^2 + 2 y'^2)
  /// \\\end{bmatrix}
  /// \newline
  ///
  /// \\d &= \frac{
  ///     1 + k_1 r'^2 + k_2 r'^4 + k_3 r'^6
  ///   }{
  ///     1 + k_4 r'^2 + k_5 r'^4 + k_6 r'^6
  ///   }
  /// \newline
  ///
  /// \\r'^2 &= x'^2 + y'^2
  /// \newline
  ///
  /// \\\begin{bmatrix}
  ///     x'
  /// \\  y'
  /// \\\end{bmatrix} &=
  ///   \begin{bmatrix}
  ///     x / z
  /// \\  y / z
  /// \\\end{bmatrix}
  /// \newline
  /// \f}
  ///
  /// A set of 3D points that results in valid projection is expressed as
  /// follows: \f{align}{
  ///    \Omega &= \{\mathbf{x} \in \mathbb{R}^3 ~|~
  ///    z > 0 \land r'^2 < {r'_{max}}^2 \}
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
    const Scalar& k1 = param_[4];
    const Scalar& k2 = param_[5];
    const Scalar& p1 = param_[6];
    const Scalar& p2 = param_[7];
    const Scalar& k3 = param_[8];
    const Scalar& k4 = param_[9];
    const Scalar& k5 = param_[10];
    const Scalar& k6 = param_[11];

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    const Scalar xp = x / z;
    const Scalar yp = y / z;
    const Scalar rp2 = xp * xp + yp * yp;
    const Scalar cdist = (S1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (S1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
    const Scalar deltaX = S2 * p1 * xp * yp + p2 * (rp2 + S2 * xp * xp);
    const Scalar deltaY = S2 * p2 * xp * yp + p1 * (rp2 + S2 * yp * yp);
    const Scalar xpp = xp * cdist + deltaX;
    const Scalar ypp = yp * cdist + deltaY;
    const Scalar u = fx * xpp + cx;
    const Scalar v = fy * ypp + cy;

    proj[0] = u;
    proj[1] = v;

    bool positive_z = z >= Sophus::Constants<Scalar>::epsilonSqrt();
    bool in_injective_area = rpmax_ == S0 ? true : rp2 <= rpmax_ * rpmax_;
    bool is_valid = positive_z && in_injective_area;

    // The following derivative expressions were computed automatically with
    // simpy, see radtan8/main_jacobians.py.

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_p3d);

      d_proj_d_p3d->setZero();

      const Scalar v0 = p1 * y;
      const Scalar v1 = p2 * x;
      const Scalar v2 = z * z * z * z * z * z;
      const Scalar v3 = x * x;
      const Scalar v4 = y * y;
      const Scalar v5 = v3 + v4;
      const Scalar v6 = z * z * z * z;
      const Scalar v7 = z * z;
      const Scalar v8 = k5 * v7;
      const Scalar v9 = k6 * v5;
      const Scalar v10 = k4 * v6 + v5 * (v8 + v9);
      const Scalar v11 = v10 * v5 + v2;
      const Scalar v12 = v11 * v11;
      const Scalar v13 = S2 * v12;
      const Scalar v14 = k2 * v7;
      const Scalar v15 = k3 * v5;
      const Scalar v16 = k1 * v6 + v5 * (v14 + v15);
      const Scalar v17 = v16 * v5 + v2;
      const Scalar v18 = v17 * z * (v10 + v5 * (v8 + S2 * v9));
      const Scalar v19 = S2 * v18;
      const Scalar v20 = v16 + v5 * (v14 + S2 * v15);
      const Scalar v21 = S2 * v20;
      const Scalar v22 = v11 * z;
      const Scalar v23 = S1 / v7;
      const Scalar v24 = S1 / v12;
      const Scalar v25 = fx * v24;
      const Scalar v26 = v23 * v25;
      const Scalar v27 = p2 * y;
      const Scalar v28 = x * y;
      const Scalar v29 = S2 * v12 * (p1 * x + v27) - S2 * v18 * v28 + S2 * v20 * v22 * v28;
      const Scalar v30 = S1 / (z * z * z);
      const Scalar v31 = S2 * x;
      const Scalar v32 = v22 * (v17 + v21 * v5);
      const Scalar v33 = fy * v24;
      const Scalar v34 = v23 * v33;

      const Scalar du_dx = v26 * (v13 * (v0 + S3 * v1) - v19 * v3 + v22 * (v17 + v21 * v3));
      const Scalar du_dy = v26 * v29;
      const Scalar du_dz = -v25 * v30 * (v13 * (p2 * (S3 * v3 + v4) + v0 * v31) - v18 * v31 * v5 + v32 * x);
      const Scalar dv_dx = v29 * v34;
      const Scalar dv_dy = v34 * (v13 * (S3 * v0 + v1) - v19 * v4 + v22 * (v17 + v21 * v4));
      const Scalar dv_dz = -v30 * v33 * (v13 * (p1 * (v3 + S3 * v4) + v27 * v31) - v19 * v5 * y + v32 * y);

      (*d_proj_d_p3d)(0, 0) = du_dx;
      (*d_proj_d_p3d)(0, 1) = du_dy;
      (*d_proj_d_p3d)(0, 2) = du_dz;
      (*d_proj_d_p3d)(1, 0) = dv_dx;
      (*d_proj_d_p3d)(1, 1) = dv_dy;
      (*d_proj_d_p3d)(1, 2) = dv_dz;
    } else {
      UNUSED(d_proj_d_p3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_param);
      d_proj_d_param->setZero();

      const Scalar w0 = z * z * z * z * z * z;
      const Scalar w1 = x * x;
      const Scalar w2 = y * y;
      const Scalar w3 = w1 + w2;
      const Scalar w4 = z * z * z * z;
      const Scalar w5 = z * z;
      const Scalar w6 = w0 + w3 * (k1 * w4 + w3 * (k2 * w5 + k3 * w3));
      const Scalar w7 = w6 * z;
      const Scalar w8 = w7 * x;
      const Scalar w9 = S2 * x * y;
      const Scalar w10 = S3 * w1 + w2;
      const Scalar w11 = w0 + w3 * (k4 * w4 + w3 * (k5 * w5 + k6 * w3));
      const Scalar w12 = S1 / w5;
      const Scalar w13 = S1 / w11;
      const Scalar w14 = w12 * w13;
      const Scalar w15 = w3 * (z * z * z);
      const Scalar w16 = fx * x;
      const Scalar w17 = w13 * w16;
      const Scalar w18 = w3 * w3;
      const Scalar w19 = w18 * z;
      const Scalar w20 = fx * w12;
      const Scalar w21 = (w3 * w3 * w3) * S1 / z;
      const Scalar w22 = S1 / (w11 * w11);
      const Scalar w23 = w22 * w6;
      const Scalar w24 = w16 * w23;
      const Scalar w25 = w18 * w22;
      const Scalar w26 = w7 * y;
      const Scalar w27 = w1 + S3 * w2;
      const Scalar w28 = fy * y;
      const Scalar w29 = w13 * w28;
      const Scalar w30 = fy * w12;
      const Scalar w31 = w23 * w28;

      const Scalar du_fx = w14 * (w11 * (p1 * w9 + p2 * w10) + w8);
      const Scalar du_fy = S0;
      const Scalar du_cx = S1;
      const Scalar du_cy = S0;
      const Scalar du_k1 = w15 * w17;
      const Scalar du_k2 = w17 * w19;
      const Scalar du_p1 = w20 * w9;
      const Scalar du_p2 = w10 * w20;
      const Scalar du_k3 = w17 * w21;
      const Scalar du_k4 = -w15 * w24;
      const Scalar du_k5 = -fx * w25 * w8;
      const Scalar du_k6 = -w21 * w24;
      const Scalar dv_fx = S0;
      const Scalar dv_fy = w14 * (w11 * (p1 * w27 + p2 * w9) + w26);
      const Scalar dv_cx = S0;
      const Scalar dv_cy = S1;
      const Scalar dv_k1 = w15 * w29;
      const Scalar dv_k2 = w19 * w29;
      const Scalar dv_p1 = w27 * w30;
      const Scalar dv_p2 = w30 * w9;
      const Scalar dv_k3 = w21 * w29;
      const Scalar dv_k4 = -w15 * w31;
      const Scalar dv_k5 = -fy * w25 * w26;
      const Scalar dv_k6 = -w21 * w31;

      (*d_proj_d_param)(0, 0) = du_fx;
      (*d_proj_d_param)(0, 1) = du_fy;
      (*d_proj_d_param)(0, 2) = du_cx;
      (*d_proj_d_param)(0, 3) = du_cy;
      (*d_proj_d_param)(0, 4) = du_k1;
      (*d_proj_d_param)(0, 5) = du_k2;
      (*d_proj_d_param)(0, 6) = du_p1;
      (*d_proj_d_param)(0, 7) = du_p2;
      (*d_proj_d_param)(0, 8) = du_k3;
      (*d_proj_d_param)(0, 9) = du_k4;
      (*d_proj_d_param)(0, 10) = du_k5;
      (*d_proj_d_param)(0, 11) = du_k6;
      (*d_proj_d_param)(1, 0) = dv_fx;
      (*d_proj_d_param)(1, 1) = dv_fy;
      (*d_proj_d_param)(1, 2) = dv_cx;
      (*d_proj_d_param)(1, 3) = dv_cy;
      (*d_proj_d_param)(1, 4) = dv_k1;
      (*d_proj_d_param)(1, 5) = dv_k2;
      (*d_proj_d_param)(1, 6) = dv_p1;
      (*d_proj_d_param)(1, 7) = dv_p2;
      (*d_proj_d_param)(1, 8) = dv_k3;
      (*d_proj_d_param)(1, 9) = dv_k4;
      (*d_proj_d_param)(1, 10) = dv_k5;
      (*d_proj_d_param)(1, 11) = dv_k6;
    } else {
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  /// @brief Distorts a normalized 2D point
  ///
  /// Given \f$ (x', y') \f$ computes \f$ (x'', y'') \f$ as defined in
  /// @ref project. It can also optionally compute its jacobian.
  /// @param[in] undist Undistorted normalized 2D point \f$ (x', y') \f$
  /// @param[out] dist Result of distortion \f$ (x'', y'') \f$
  /// @param[out] d_dist_d_undist if not nullptr, computed Jacobian of @p dist
  /// w.r.t @p undist
  template <class DerivedJundist = std::nullptr_t>
  inline void distort(const Vec2& undist, Vec2& dist, DerivedJundist d_dist_d_undist = nullptr) const {
    const Scalar& k1 = param_[4];
    const Scalar& k2 = param_[5];
    const Scalar& p1 = param_[6];
    const Scalar& p2 = param_[7];
    const Scalar& k3 = param_[8];
    const Scalar& k4 = param_[9];
    const Scalar& k5 = param_[10];
    const Scalar& k6 = param_[11];

    const Scalar xp = undist.x();
    const Scalar yp = undist.y();
    const Scalar rp2 = xp * xp + yp * yp;
    const Scalar cdist = (S1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (S1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)));
    const Scalar deltaX = S2 * p1 * xp * yp + p2 * (rp2 + S2 * xp * xp);
    const Scalar deltaY = S2 * p2 * xp * yp + p1 * (rp2 + S2 * yp * yp);
    const Scalar xpp = xp * cdist + deltaX;
    const Scalar ypp = yp * cdist + deltaY;
    dist.x() = xpp;
    dist.y() = ypp;

    if constexpr (!std::is_same_v<DerivedJundist, std::nullptr_t>) {
      BASALT_ASSERT(d_dist_d_undist);

      // Expressions derived with sympy
      const Scalar v0 = xp * xp;
      const Scalar v1 = yp * yp;
      const Scalar v2 = v0 + v1;
      const Scalar v3 = k6 * v2;
      const Scalar v4 = k4 + v2 * (k5 + v3);
      const Scalar v5 = v2 * v4 + S1;
      const Scalar v6 = v5 * v5;
      const Scalar v7 = S1 / v6;
      const Scalar v8 = p1 * yp;
      const Scalar v9 = p2 * xp;
      const Scalar v10 = S2 * v6;
      const Scalar v11 = k3 * v2;
      const Scalar v12 = k1 + v2 * (k2 + v11);
      const Scalar v13 = v12 * v2 + S1;
      const Scalar v14 = v13 * (v2 * (k5 + S2 * v3) + v4);
      const Scalar v15 = S2 * v14;
      const Scalar v16 = v12 + v2 * (k2 + S2 * v11);
      const Scalar v17 = S2 * v16;
      const Scalar v18 = xp * yp;
      const Scalar v19 = S2 * v7 * (-v14 * v18 + v16 * v18 * v5 + v6 * (p1 * xp + p2 * yp));

      const Scalar dxpp_dxp = v7 * (-v0 * v15 + v10 * (v8 + S3 * v9) + v5 * (v0 * v17 + v13));
      const Scalar dxpp_dyp = v19;
      const Scalar dypp_dxp = v19;
      const Scalar dypp_dyp = v7 * (-v1 * v15 + v10 * (S3 * v8 + v9) + v5 * (v1 * v17 + v13));

      (*d_dist_d_undist)(0, 0) = dxpp_dxp;
      (*d_dist_d_undist)(0, 1) = dxpp_dyp;
      (*d_dist_d_undist)(1, 0) = dypp_dxp;
      (*d_dist_d_undist)(1, 1) = dypp_dyp;
    } else {
      UNUSED(d_dist_d_undist);
    }
  }

  /// @brief Unproject the point
  /// @note Computing the jacobians is not implemented
  ///
  /// The unprojection function is computed as follows:
  /// \f{align}{
  ///   \pi^{-1}(\mathbf{u}, \mathbf{i}) &= \frac{1}{\sqrt{x'^2 + y'^2 + 1}}
  ///   \begin{bmatrix} x'^2 \\ y'^2 \\ 1 \end{bmatrix}
  ///   \newline
  ///
  /// \\\begin{bmatrix} x' \\ y' \end{bmatrix} &=
  ///   distort^{-1}\left( \begin{bmatrix} x'' \\ y'' \end{bmatrix} \right)
  ///   \newline
  ///
  /// \\\begin{bmatrix} x'' \\ y'' \end{bmatrix} &=
  ///   \begin{bmatrix} (u - c_x) / f_x \\ (v - c_y) / f_y \end{bmatrix}
  ///   \newline
  ///
  /// \f}
  ///
  /// In which \f$ distort^{-1} \f$ is the inverse of @ref distort computed
  /// iteratively with [Newton's
  /// method](https://en.wikipedia.org/wiki/Newton%27s_method).
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj \b UNIMPLEMENTED if not nullptr, computed
  /// Jacobian of unprojection with respect to proj
  /// @param[out] d_p3d_d_param \b UNIMPLEMENTED if not nullptr, computed
  /// Jacobian of unprojection with respect to intrinsic parameters
  /// @return whether the unprojection is valid
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

    const Scalar& u = proj_eval[0];
    const Scalar& v = proj_eval[1];

    const Scalar x0 = (u - cx) / fx;
    const Scalar y0 = (v - cy) / fy;

    //! @todo Decide if besides rpmax, it could be useful to have an rppmax
    //! field. A good starting point to having this would be using the sqrt of
    //! the max rpp2 value computed in the optimization of `computeRpmax()`.

#if 1
    // Newton solver
    Vec2 dist{x0, y0};
    Vec2 undist{dist};
    const Scalar EPS = Sophus::Constants<Scalar>::epsilonSqrt();
    constexpr int N = 5;  // Max iterations
    for (int i = 0; i < N; i++) {
      Mat22 J{};
      Vec2 fundist{};
      distort(undist, fundist, &J);
      Vec2 residual = fundist - dist;
      undist -= J.inverse() * residual;
      if (residual.norm() < EPS) {
        break;
      }
    }
    const Scalar xp = undist.x();
    const Scalar yp = undist.y();
#else
    // Jacobi solver, same as OpenCV undistortPoints. Less precise.
    constexpr int N = 100;  // Number of iterations
    Scalar xp = x0;
    Scalar yp = y0;
    for (int i = 0; i < N; i++) {
      const Scalar rp2 = xp * xp + yp * yp;
      const Scalar icdist = (S1 + rp2 * (k4 + rp2 * (k5 + rp2 * k6))) / (S1 + rp2 * (k1 + rp2 * (k2 + rp2 * k3)));
      if (icdist <= S0) {
        return false;  // OpenCV just sets xp=x0, yp=y0 instead
      }
      const Scalar deltaX = S2 * p1 * xp * yp + p2 * (rp2 + S2 * xp * xp);
      const Scalar deltaY = S2 * p2 * xp * yp + p1 * (rp2 + S2 * yp * yp);
      xp = (x0 - deltaX) * icdist;
      yp = (y0 - deltaY) * icdist;
    }
#endif

    const Scalar norm_inv = S1 / sqrt(xp * xp + yp * yp + S1);
    p3d.setZero();
    p3d[0] = xp * norm_inv;
    p3d[1] = yp * norm_inv;
    p3d[2] = norm_inv;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> || !std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(false);  // Not implemented
      // If this gets implemented update: docs, benchmarks and tests
    }
    UNUSED(d_p3d_d_proj);
    UNUSED(d_p3d_d_param);

    const Scalar rp2 = xp * xp + yp * yp;
    bool in_injective_area = rpmax_ == S0 ? true : rp2 <= rpmax_ * rpmax_;
    bool is_valid = in_injective_area;

    return is_valid;
  }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[
  /// f_x, f_y, c_x, c_y, 0, 0, 0, 0, 0, 0, 0, 0 \right]^T \f$
  /// @param[in] init vector [f_x, f_y, c_x, c_y]
  inline void setFromInit(const Vec4& init) {
    param_.setZero();
    param_[0] = init[0];
    param_[1] = init[1];
    param_[2] = init[2];
    param_[3] = init[3];
    rpmax_ = S0;  // No distortion, so biyective
  }

  /// @brief Increment intrinsic parameters by inc
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) {
    param_ += inc;
    rpmax_ = computeRpmax();
  }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is as follows: \f$ \left[
  /// f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Returns @ref rpmax_.
  Scalar getRpmax() const { return rpmax_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<PinholeRadtan8Camera> getTestProjections() {
    Eigen::aligned_vector<PinholeRadtan8Camera> res;

    VecN vec1{};

    // Odyssey+, original rpmax: 2.7941114902496338 (see bit.ly/monado-datasets)
    vec1 << 269.0600776672363, 269.1679859161377, 324.3333053588867, 245.22674560546875, 0.6257319450378418,
        0.46612036228179932, -0.00018502399325370789, -4.2882973502855748e-5, 0.0041795829311013222,
        0.89431935548782349, 0.54253977537155151, 0.0662121474742889;

    res.emplace_back(vec1);

    return res;
  }

  /// @brief Resolutions used for unit-tests
  static Eigen::aligned_vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::aligned_vector<Eigen::Vector2i> res;
    res.emplace_back(640, 480);
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;

  /// Specifies the radius of a circle that approximates the valid projection
  /// domain. 0 means unbounded.
  Scalar rpmax_;
};

}  // namespace basalt

#undef SN1
#undef S0
#undef S1
#undef S2
#undef S3
