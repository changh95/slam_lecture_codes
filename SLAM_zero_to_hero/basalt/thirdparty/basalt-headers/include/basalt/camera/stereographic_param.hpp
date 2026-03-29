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
@brief Implementation of stereographic parametrization for unit vectors
*/

#pragma once

#include <Eigen/Dense>

namespace basalt {

/// @brief Stereographic projection for minimal representation of unit vectors
///
/// \image html stereographic.png
/// The projection is defined on the entire
/// sphere, except at one point: the projection point. Where it is defined, the
/// mapping is smooth and bijective. The illustrations shows 3 examples of unit
/// vectors parametrized with a point in the 2D plane. See
/// \ref project and \ref unproject functions for more details.
template <typename Scalar_ = double>
class StereographicParam {
 public:
  using Scalar = Scalar_;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;

  /// @brief Project the point and optionally compute Jacobian
  ///
  /// Projection function is defined as follows:
  /// \f{align}{
  ///    \pi(\mathbf{x}) &=
  ///    \begin{bmatrix}
  ///   {\frac{x}{d + z}}
  ///    \\ {\frac{y}{d + z}}
  ///    \\ \end{bmatrix},
  ///    \\ d &= \sqrt{x^2 + y^2 + z^2}.
  /// \f}
  /// @param[in] p3d point to project [x,y,z]
  /// @param[out] d_r_d_p if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @return 2D projection of the point which parametrizes the corresponding
  /// direction vector
  static inline Vec2 project(const Vec4& p3d, Mat24* d_r_d_p = nullptr) {
    const Scalar sqrt = p3d.template head<3>().norm();
    const Scalar norm = p3d[2] + sqrt;
    const Scalar norm_inv = Scalar(1) / norm;

    Vec2 res(p3d[0] * norm_inv, p3d[1] * norm_inv);

    if (d_r_d_p) {
      Scalar norm_inv2 = norm_inv * norm_inv;
      Scalar tmp = -norm_inv2 / sqrt;

      (*d_r_d_p).setZero();

      (*d_r_d_p)(0, 0) = norm_inv + p3d[0] * p3d[0] * tmp;
      (*d_r_d_p)(1, 0) = p3d[0] * p3d[1] * tmp;

      (*d_r_d_p)(1, 1) = norm_inv + p3d[1] * p3d[1] * tmp;
      (*d_r_d_p)(0, 1) = p3d[0] * p3d[1] * tmp;

      (*d_r_d_p)(0, 2) = p3d[0] * norm * tmp;
      (*d_r_d_p)(1, 2) = p3d[1] * norm * tmp;

      (*d_r_d_p)(0, 3) = Scalar(0);
      (*d_r_d_p)(1, 3) = Scalar(0);
    }

    return res;
  }

  /// @brief Unproject the point and optionally compute Jacobian
  ///
  /// The unprojection function is computed as follows: \f{align}{
  ///    \pi^{-1}(\mathbf{u}) &=
  ///    \frac{2}{1 + x^2 + y^2}
  ///    \begin{bmatrix}
  ///    u \\ v \\ 1
  ///    \\ \end{bmatrix}-\begin{bmatrix}
  ///    0 \\ 0 \\ 1
  ///    \\ \end{bmatrix}.
  /// \f}
  ///
  /// @param[in] proj point to unproject [u,v]
  /// @param[out] d_r_d_p if not nullptr computed Jacobian of unprojection
  /// with respect to proj
  /// @return unprojected 3D unit vector
  static inline Vec4 unproject(const Vec2& proj, Mat42* d_r_d_p = nullptr) {
    const Scalar x2 = proj[0] * proj[0];
    const Scalar y2 = proj[1] * proj[1];
    const Scalar r2 = x2 + y2;

    const Scalar norm_inv = Scalar(2) / (Scalar(1) + r2);

    Vec4 res(proj[0] * norm_inv, proj[1] * norm_inv, norm_inv - Scalar(1), Scalar(0));

    if (d_r_d_p) {
      const Scalar norm_inv2 = norm_inv * norm_inv;
      const Scalar xy = proj[0] * proj[1];

      (*d_r_d_p)(0, 0) = (norm_inv - x2 * norm_inv2);
      (*d_r_d_p)(0, 1) = -xy * norm_inv2;

      (*d_r_d_p)(1, 0) = -xy * norm_inv2;
      (*d_r_d_p)(1, 1) = (norm_inv - y2 * norm_inv2);

      (*d_r_d_p)(2, 0) = -proj[0] * norm_inv2;
      (*d_r_d_p)(2, 1) = -proj[1] * norm_inv2;

      (*d_r_d_p)(3, 0) = Scalar(0);
      (*d_r_d_p)(3, 1) = Scalar(0);
    }

    return res;
  }
};

}  // namespace basalt
