
#include <Eigen/Dense>

#include <basalt/image/image.h>

#include "gtest/gtest.h"
#include "test_utils.h"

void setImageData(uint16_t* image_array, int size) {
  double norm = RAND_MAX;
  norm /= (double)std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < size; i++) {
    image_array[i] = (uint16_t)(rand() / norm);
  }
}

TEST(Image, ImageInterpolate) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  double eps = 1e-12;
  double threshold = 1e-6;

  {
    const Eigen::Vector2i& pi = offset;
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    const Eigen::Vector2i& pi = offset;
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(1, 0);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(-eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(0, 1);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, -eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(1, 1);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(-eps, -eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }
}

TEST(Image, ImageInterpolateGrad) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGrad<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) { return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x)); },
      Eigen::Vector2d::Zero(),
      1);  // only works with eps=1 with this gradient interpolation
}

TEST(Image, ImageInterpolateGradBilinearExact) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGradBilinearExact<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) { return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x)); },
      Eigen::Vector2d::Zero(), 1e-4);
}

TEST(Image, ImageInterpolateGradCubicSplines) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGradCubicSplines<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) { return Eigen::Matrix<double, 1, 1>(img.interpCubicSplines<double>(pd + x)); },
      Eigen::Vector2d::Zero(), 1e-4);
}
