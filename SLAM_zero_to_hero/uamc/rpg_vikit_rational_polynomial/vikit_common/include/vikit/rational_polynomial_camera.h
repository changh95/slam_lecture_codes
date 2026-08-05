/*
 * rational_polynomial_camera.h
 *
 *  Created on: March 21, 2026
 *  OpenCV rational polynomial distortion model (CALIB_RATIONAL_MODEL)
 *  with 8 distortion parameters (k1-k6, p1, p2) and 4 intrinsics (fx, fy, cx, cy).
 */

#ifndef RATIONAL_POLYNOMIAL_CAMERA_H_
#define RATIONAL_POLYNOMIAL_CAMERA_H_

#include <stdlib.h>
#include <string>
#include <Eigen/Eigen>
#include <vikit/abstract_camera.h>

namespace vk {

using namespace std;
using namespace Eigen;

class RationalPolynomialCamera : public AbstractCamera {

private:
  const double fx_, fy_;
  const double cx_, cy_;
  bool distortion_;
  double k1_, k2_, k3_, k4_, k5_, k6_;
  double p1_, p2_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RationalPolynomialCamera(double width, double height,
                           double fx, double fy, double cx, double cy,
                           double k1=0.0, double k2=0.0, double k3=0.0,
                           double k4=0.0, double k5=0.0, double k6=0.0,
                           double p1=0.0, double p2=0.0);

  ~RationalPolynomialCamera();

  virtual Vector3d
  cam2world(const double& x, const double& y) const;

  virtual Vector3d
  cam2world(const Vector2d& px) const;

  virtual Vector2d
  world2cam(const Vector3d& xyz_c) const;

  virtual Vector2d
  world2cam(const Vector2d& uv) const;

  const Vector2d focal_length() const
  {
    return Vector2d(fx_, fy_);
  }

  /// Compute distorted coordinates from undistorted normalized coords
  inline Vector2d distort(const Vector2d& uv) const
  {
    const double x = uv[0];
    const double y = uv[1];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    double denom = 1.0 + k4_ * r2 + k5_ * r4 + k6_ * r6;
    if(fabs(denom) < 1e-8)
      denom = (denom > 0) ? 1e-8 : -1e-8;

    const double r_coeff = (1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6) / denom;

    const double dx = 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    const double dy = p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;

    return Vector2d(x * r_coeff + dx, y * r_coeff + dy);
  }

  /// Jacobian of the distortion function w.r.t. undistorted coords (x, y)
  inline Matrix2d distortJacobian(const Vector2d& uv) const
  {
    const double x = uv[0];
    const double y = uv[1];
    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;

    const double num = 1.0 + k1_ * r2 + k2_ * r4 + k3_ * r6;
    double denom = 1.0 + k4_ * r2 + k5_ * r4 + k6_ * r6;
    if(fabs(denom) < 1e-8)
      denom = (denom > 0) ? 1e-8 : -1e-8;

    const double r_coeff = num / denom;

    // dr2/dx = 2x, dr2/dy = 2y
    const double dnum_dr2 = k1_ + 2.0 * k2_ * r2 + 3.0 * k3_ * r4;
    const double ddenom_dr2 = k4_ + 2.0 * k5_ * r2 + 3.0 * k6_ * r4;
    const double dr_coeff_dr2 = (dnum_dr2 * denom - num * ddenom_dr2) / (denom * denom);

    // d(r_coeff)/dx = dr_coeff_dr2 * 2x
    const double dr_coeff_dx = dr_coeff_dr2 * 2.0 * x;
    const double dr_coeff_dy = dr_coeff_dr2 * 2.0 * y;

    // xd = x * r_coeff + dx_tang
    // dxd/dx = r_coeff + x * dr_coeff_dx + 2*p1*y + p2*(2x + 4x) = r_coeff + x*dr_coeff_dx + 2*p1*y + 6*p2*x
    // dxd/dy = x * dr_coeff_dy + 2*p1*x + 2*p2*y
    // dyd/dx = y * dr_coeff_dx + 2*p1*x + 2*p2*y
    // dyd/dy = r_coeff + y * dr_coeff_dy + p1*(2y + 4y) + 2*p2*x = r_coeff + y*dr_coeff_dy + 6*p1*y + 2*p2*x

    Matrix2d J;
    J(0, 0) = r_coeff + x * dr_coeff_dx + 2.0 * p1_ * y + 6.0 * p2_ * x;
    J(0, 1) = x * dr_coeff_dy + 2.0 * p1_ * x + 2.0 * p2_ * y;
    J(1, 0) = y * dr_coeff_dx + 2.0 * p1_ * x + 2.0 * p2_ * y;
    J(1, 1) = r_coeff + y * dr_coeff_dy + 6.0 * p1_ * y + 2.0 * p2_ * x;
    return J;
  }

  inline Eigen::Matrix<double, 2, 3> jacobian_2x3(const Eigen::Vector3d& p) const
  {
    Eigen::Matrix<double, 2, 3> jac;
    const double z_inv = 1.0 / p[2];
    const double z_inv_2 = z_inv * z_inv;
    jac(0, 0) = fx_ * z_inv;
    jac(0, 1) = 0.0;
    jac(0, 2) = -fx_ * p[0] * z_inv_2;
    jac(1, 0) = 0.0;
    jac(1, 1) = fy_ * z_inv;
    jac(1, 2) = -fy_ * p[1] * z_inv_2;
    return jac;
  }

  virtual double errorMultiplier2() const
  {
    return fabs(fx_);
  }

  virtual double errorMultiplier() const
  {
    return fabs(4.0 * fx_ * fy_);
  }

  virtual double fx() const { return fx_; };
  virtual double fy() const { return fy_; };
  virtual double cx() const { return cx_; };
  virtual double cy() const { return cy_; };
};

} // end namespace vk

#endif /* RATIONAL_POLYNOMIAL_CAMERA_H_ */
