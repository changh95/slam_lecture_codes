/*
 * rational_polynomial_camera.cpp
 *
 *  Created on: March 21, 2026
 *  OpenCV rational polynomial distortion model (CALIB_RATIONAL_MODEL).
 */

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vikit/rational_polynomial_camera.h>
#include <vikit/math_utils.h>

namespace vk {

RationalPolynomialCamera::
RationalPolynomialCamera(double width, double height,
                         double fx, double fy, double cx, double cy,
                         double k1, double k2, double k3,
                         double k4, double k5, double k6,
                         double p1, double p2) :
    AbstractCamera(width, height, 1.0),
    fx_(fx), fy_(fy), cx_(cx), cy_(cy),
    distortion_(fabs(k1) > 1e-7 || fabs(k2) > 1e-7 || fabs(k3) > 1e-7 ||
                fabs(k4) > 1e-7 || fabs(k5) > 1e-7 || fabs(k6) > 1e-7 ||
                fabs(p1) > 1e-7 || fabs(p2) > 1e-7),
    k1_(k1), k2_(k2), k3_(k3), k4_(k4), k5_(k5), k6_(k6),
    p1_(p1), p2_(p2)
{}

RationalPolynomialCamera::
~RationalPolynomialCamera()
{}

Vector3d RationalPolynomialCamera::
cam2world(const double& u, const double& v) const
{
  Vector3d xyz;
  if(!distortion_)
  {
    xyz[0] = (u - cx_) / fx_;
    xyz[1] = (v - cy_) / fy_;
    xyz[2] = 1.0;
  }
  else
  {
    // Remove intrinsics to get distorted normalized coords
    double xd = (u - cx_) / fx_;
    double yd = (v - cy_) / fy_;

    // Iterative undistortion via Newton's method
    double x = xd;
    double y = yd;
    for(int i = 0; i < 10; ++i)
    {
      Vector2d uv_cur(x, y);
      Vector2d distorted = distort(uv_cur);
      Vector2d residual(distorted[0] - xd, distorted[1] - yd);
      Matrix2d J = distortJacobian(uv_cur);
      Vector2d update = J.inverse() * residual;
      x -= update[0];
      y -= update[1];
    }

    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = 1.0;
  }
  return xyz.normalized();
}

Vector3d RationalPolynomialCamera::
cam2world(const Vector2d& px) const
{
  return cam2world(px[0], px[1]);
}

Vector2d RationalPolynomialCamera::
world2cam(const Vector3d& xyz_c) const
{
  Vector2d px;
  if(!distortion_)
  {
    px[0] = fx_ * xyz_c[0] / xyz_c[2] + cx_;
    px[1] = fy_ * xyz_c[1] / xyz_c[2] + cy_;
  }
  else
  {
    const double x = xyz_c[0] / xyz_c[2];
    const double y = xyz_c[1] / xyz_c[2];

    Vector2d d = distort(Vector2d(x, y));
    px[0] = fx_ * d[0] + cx_;
    px[1] = fy_ * d[1] + cy_;
  }
  return px;
}

Vector2d RationalPolynomialCamera::
world2cam(const Vector2d& uv) const
{
  Vector2d px;
  if(!distortion_)
  {
    px[0] = fx_ * uv[0] + cx_;
    px[1] = fy_ * uv[1] + cy_;
  }
  else
  {
    Vector2d d = distort(uv);
    px[0] = fx_ * d[0] + cx_;
    px[1] = fy_ * d[1] + cy_;
  }
  return px;
}

} // end namespace vk
