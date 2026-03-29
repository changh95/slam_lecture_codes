/**
BSD 3-Clause License

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

#include <basalt/camera/generic_camera.hpp>
#include <basalt/camera/stereographic_param.hpp>

#include "gtest/gtest.h"
#include "test_utils.h"

template <typename CamT>
void testProjectJacobian() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using VecN = typename CamT::VecN;
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat24 = typename CamT::Mat24;
  using Mat2N = typename CamT::Mat2N;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = -1; z <= 5; z++) {
          Vec4 p(x, y, z, 1);

          Mat24 J_p;
          Mat2N J_param;

          Vec2 res1;

          bool success = cam.project(p, res1, &J_p, &J_param);

          if (success) {
            test_jacobian(
                "d_r_d_p", J_p,
                [&](const Vec4 &x) {
                  Vec2 res;
                  cam.project(p + x, res);
                  return res;
                },
                Vec4::Zero());

            test_jacobian(
                "d_r_d_param", J_param,
                [&](const VecN &x) {
                  Vec2 res;

                  CamT cam1 = cam;
                  cam1 += x;

                  cam1.project(p, res);
                  return res;
                },
                VecN::Zero());
          }
        }
      }
    }
  }
}

template <typename CamT>
void testProjectJacobian3() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  constexpr int N = CamT::N;

  using Scalar = typename CamT::Scalar;
  using VecN = typename CamT::VecN;
  using Vec2 = typename CamT::Vec2;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  using Mat23 = Eigen::Matrix<Scalar, 2, 3, Eigen::RowMajor>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N, Eigen::RowMajor>;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = -1; z <= 5; z++) {
          Vec3 p_init(x, y, z);

          Scalar p_raw_array[3];
          Scalar res_raw_array[2];
          Scalar J_p_raw_array[2 * 3];
          Scalar J_param_raw_array[2 * N];

          Eigen::Map<Vec3> p(p_raw_array);
          p = p_init;

          Eigen::Map<Mat23> J_p(J_p_raw_array);
          Eigen::Map<Mat2N> J_param(J_param_raw_array);

          Eigen::Map<Vec2> res1(res_raw_array);

          bool success = cam.project(p, res1, &J_p, nullptr);
          success = cam.project(p, res1, nullptr, &J_param);

          if (success) {
            test_jacobian(
                "d_r_d_p", J_p,
                [&](const Vec3 &x) {
                  Vec2 res;
                  cam.project(p + x, res);
                  return res;
                },
                Vec3::Zero());

            test_jacobian(
                "d_r_d_param", J_param,
                [&](const VecN &x) {
                  Vec2 res;

                  CamT cam1 = cam;
                  cam1 += x;

                  cam1.project(p, res);
                  return res;
                },
                VecN::Zero());
          }
        }
      }
    }
  }
}

template <typename CamT>
void testProjectUnproject() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using Scalar = typename CamT::Vec2::Scalar;
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = 0; z <= 5; z++) {
          Vec4 p(x, y, z, 0.23424);

          Vec4 p_normalized = Vec4::Zero();
          p_normalized.template head<3>() = p.template head<3>().normalized();
          Vec2 res;
          bool success = cam.project(p, res);

          if (success) {
            Vec4 p_uproj;
            cam.unproject(res, p_uproj);

            EXPECT_TRUE(p_normalized.isApprox(p_uproj, Sophus::Constants<Scalar>::epsilonSqrt()))
                << "p_normalized " << p_normalized.transpose() << " p_uproj " << p_uproj.transpose();
          }
        }
      }
    }
  }
}

template <typename CamT>
void testUnprojectJacobians() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using VecN = typename CamT::VecN;
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat42 = typename CamT::Mat42;
  using Mat4N = typename CamT::Mat4N;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = 0; z <= 5; z++) {
          Vec4 p_3d(x, y, z, 0);

          Vec2 p;
          bool success = cam.project(p_3d, p);

          if (success) {
            Mat42 J_p;
            Mat4N J_param;

            Vec4 res1;
            cam.unproject(p, res1, &J_p, &J_param);

            test_jacobian(
                "d_r_d_p", J_p,
                [&](const Vec2 &x) {
                  Vec4 res = Vec4::Zero();
                  cam.unproject(p + x, res);
                  return res;
                },
                Vec2::Zero());

            test_jacobian(
                "d_r_d_param", J_param,
                [&](const VecN &x) {
                  Vec4 res = Vec4::Zero();
                  CamT cam1 = cam;
                  cam1 += x;

                  cam1.unproject(p, res);
                  return res;
                },
                VecN::Zero());
          }
        }
      }
    }
  }
}

template <typename CamT>
void testUnprojectJacobians3() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using Scalar = typename CamT::Scalar;
  constexpr int N = CamT::N;

  using VecN = typename CamT::VecN;
  using Vec2 = typename CamT::Vec2;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  using Mat32 = Eigen::Matrix<Scalar, 3, 2, Eigen::RowMajor>;
  using Mat3N = Eigen::Matrix<Scalar, 3, N, Eigen::RowMajor>;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = 0; z <= 5; z++) {
          Vec3 p_3d(x, y, z);

          Vec2 p;
          bool success = cam.project(p_3d, p);

          if (success) {
            Mat32 J_p;
            Mat3N J_param;

            Vec3 res1;
            cam.unproject(p, res1, &J_p, nullptr);
            cam.unproject(p, res1, nullptr, &J_param);

            test_jacobian(
                "d_r_d_p", J_p,
                [&](const Vec2 &x) {
                  Vec3 res = Vec3::Zero();
                  cam.unproject(p + x, res);
                  return res;
                },
                Vec2::Zero());

            test_jacobian(
                "d_r_d_param", J_param,
                [&](const VecN &x) {
                  Vec3 res = Vec3::Zero();
                  CamT cam1 = cam;
                  cam1 += x;

                  cam1.unproject(p, res);
                  return res;
                },
                VecN::Zero());
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeProjectJacobians) { testProjectJacobian<basalt::PinholeCamera<double>>(); }
TEST(CameraTestCase, PinholeProjectJacobiansFloat) { testProjectJacobian<basalt::PinholeCamera<float>>(); }

TEST(CameraTestCase, PinholeRadtan8ProjectJacobians) { testProjectJacobian<basalt::PinholeRadtan8Camera<double>>(); }
TEST(CameraTestCase, PinholeRadtan8ProjectJacobiansFloat) {
  testProjectJacobian<basalt::PinholeRadtan8Camera<float>>();
}

TEST(CameraTestCase, UnifiedProjectJacobians) { testProjectJacobian<basalt::UnifiedCamera<double>>(); }
TEST(CameraTestCase, UnifiedProjectJacobiansFloat) { testProjectJacobian<basalt::UnifiedCamera<float>>(); }

TEST(CameraTestCase, ExtendedUnifiedProjectJacobians) { testProjectJacobian<basalt::ExtendedUnifiedCamera<double>>(); }
TEST(CameraTestCase, ExtendedUnifiedProjectJacobiansFloat) {
  testProjectJacobian<basalt::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtProjectJacobians) { testProjectJacobian<basalt::KannalaBrandtCamera4<double>>(); }
TEST(CameraTestCase, KannalaBrandtProjectJacobiansFloat) { testProjectJacobian<basalt::KannalaBrandtCamera4<float>>(); }

TEST(CameraTestCase, DoubleSphereJacobians) { testProjectJacobian<basalt::DoubleSphereCamera<double>>(); }
TEST(CameraTestCase, FovCameraJacobians) { testProjectJacobian<basalt::FovCamera<double>>(); }

TEST(CameraTestCase, BalCameraJacobians) { testProjectJacobian<basalt::BalCamera<double>>(); }

TEST(CameraTestCase, BalCameraJacobiansFloat) { testProjectJacobian<basalt::BalCamera<float>>(); }

////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeProjectUnproject) { testProjectUnproject<basalt::PinholeCamera<double>>(); }
TEST(CameraTestCase, PinholeProjectUnprojectFloat) { testProjectUnproject<basalt::PinholeCamera<float>>(); }

TEST(CameraTestCase, PinholeRadtan8ProjectUnproject) { testProjectUnproject<basalt::PinholeRadtan8Camera<double>>(); }
TEST(CameraTestCase, PinholeRadtan8ProjectUnprojectFloat) {
  testProjectUnproject<basalt::PinholeRadtan8Camera<float>>();
}

TEST(CameraTestCase, UnifiedProjectUnproject) { testProjectUnproject<basalt::UnifiedCamera<double>>(); }
TEST(CameraTestCase, UnifiedProjectUnprojectFloat) { testProjectUnproject<basalt::UnifiedCamera<float>>(); }

TEST(CameraTestCase, ExtendedUnifiedProjectUnproject) { testProjectUnproject<basalt::ExtendedUnifiedCamera<double>>(); }
TEST(CameraTestCase, ExtendedUnifiedProjectUnprojectFloat) {
  testProjectUnproject<basalt::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtProjectUnproject) { testProjectUnproject<basalt::KannalaBrandtCamera4<double>>(); }
TEST(CameraTestCase, KannalaBrandtProjectUnprojectFloat) {
  testProjectUnproject<basalt::KannalaBrandtCamera4<float>>();
}

TEST(CameraTestCase, DoubleSphereProjectUnproject) { testProjectUnproject<basalt::DoubleSphereCamera<double>>(); }
TEST(CameraTestCase, DoubleSphereProjectUnprojectFloat) { testProjectUnproject<basalt::DoubleSphereCamera<float>>(); }

TEST(CameraTestCase, FovProjectUnproject) { testProjectUnproject<basalt::FovCamera<double>>(); }

TEST(CameraTestCase, FovProjectUnprojectFloat) { testProjectUnproject<basalt::FovCamera<float>>(); }

TEST(CameraTestCase, BalProjectUnproject) { testProjectUnproject<basalt::BalCamera<double>>(); }

TEST(CameraTestCase, BalProjectUnprojectFloat) { testProjectUnproject<basalt::BalCamera<float>>(); }

/////////////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeUnprojectJacobians) { testUnprojectJacobians<basalt::PinholeCamera<double>>(); }
TEST(CameraTestCase, PinholeUnprojectJacobiansFloat) { testUnprojectJacobians<basalt::PinholeCamera<float>>(); }

// TEST(CameraTestCase, PinholeRadtan8UnprojectJacobians) {
//   testUnprojectJacobians<basalt::PinholeRadtan8Camera<double>>();
// }
// TEST(CameraTestCase, PinholeRadtan8UnprojectJacobiansFloat) {
//   testUnprojectJacobians<basalt::PinholeRadtan8Camera<float>>();
// }

TEST(CameraTestCase, UnifiedUnprojectJacobians) { testUnprojectJacobians<basalt::UnifiedCamera<double>>(); }
TEST(CameraTestCase, UnifiedUnprojectJacobiansFloat) { testUnprojectJacobians<basalt::UnifiedCamera<float>>(); }

TEST(CameraTestCase, ExtendedUnifiedUnprojectJacobians) {
  testUnprojectJacobians<basalt::ExtendedUnifiedCamera<double>>();
}
TEST(CameraTestCase, ExtendedUnifiedUnprojectJacobiansFloat) {
  testUnprojectJacobians<basalt::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtUnprojectJacobians) {
  testUnprojectJacobians<basalt::KannalaBrandtCamera4<double>>();
}
// TEST(CameraTestCase, KannalaBrandtUnprojectJacobiansFloat) {
//  test_unproject_jacobians<basalt::KannalaBrandtCamera4<float>>();
//}

TEST(CameraTestCase, DoubleSphereUnprojectJacobians) { testUnprojectJacobians<basalt::DoubleSphereCamera<double>>(); }
// TEST(CameraTestCase, DoubleSphereUnprojectJacobiansFloat) {
//  test_unproject_jacobians<basalt::DoubleSphereCamera<float>>();
//}

TEST(CameraTestCase, FovUnprojectJacobians) { testUnprojectJacobians<basalt::FovCamera<double>>(); }
TEST(CameraTestCase, FovUnprojectJacobiansFloat) { testUnprojectJacobians<basalt::FovCamera<float>>(); }

////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeProjectJacobians3) { testProjectJacobian3<basalt::PinholeCamera<double>>(); }
TEST(CameraTestCase, PinholeProjectJacobiansFloat3) { testProjectJacobian3<basalt::PinholeCamera<float>>(); }

TEST(CameraTestCase, PinholeRadtan8ProjectJacobians3) { testProjectJacobian3<basalt::PinholeRadtan8Camera<double>>(); }
TEST(CameraTestCase, PinholeRadtan8ProjectJacobiansFloat3) {
  testProjectJacobian3<basalt::PinholeRadtan8Camera<float>>();
}

TEST(CameraTestCase, UnifiedProjectJacobians3) { testProjectJacobian3<basalt::UnifiedCamera<double>>(); }
TEST(CameraTestCase, UnifiedProjectJacobiansFloat3) { testProjectJacobian3<basalt::UnifiedCamera<float>>(); }

TEST(CameraTestCase, ExtendedUnifiedProjectJacobians3) {
  testProjectJacobian3<basalt::ExtendedUnifiedCamera<double>>();
}
TEST(CameraTestCase, ExtendedUnifiedProjectJacobiansFloat3) {
  testProjectJacobian3<basalt::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtProjectJacobians3) { testProjectJacobian3<basalt::KannalaBrandtCamera4<double>>(); }
TEST(CameraTestCase, KannalaBrandtProjectJacobiansFloat3) {
  testProjectJacobian3<basalt::KannalaBrandtCamera4<float>>();
}

TEST(CameraTestCase, DoubleSphereJacobians3) { testProjectJacobian3<basalt::DoubleSphereCamera<double>>(); }
TEST(CameraTestCase, FovCameraJacobians3) { testProjectJacobian3<basalt::FovCamera<double>>(); }

TEST(CameraTestCase, BalCameraJacobians3) { testProjectJacobian3<basalt::BalCamera<double>>(); }

TEST(CameraTestCase, BalCameraJacobiansFloat3) { testProjectJacobian3<basalt::BalCamera<float>>(); }

////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeUnprojectJacobians3) { testUnprojectJacobians3<basalt::PinholeCamera<double>>(); }
TEST(CameraTestCase, PinholeUnprojectJacobiansFloat3) { testUnprojectJacobians3<basalt::PinholeCamera<float>>(); }

// TEST(CameraTestCase, PinholeRadtan8UnprojectJacobians3) {
//   testUnprojectJacobians3<basalt::PinholeRadtan8Camera<double>>();
// }
// TEST(CameraTestCase, PinholeRadtan8UnprojectJacobiansFloat3) {
//   testUnprojectJacobians3<basalt::PinholeRadtan8Camera<float>>();
// }

TEST(CameraTestCase, UnifiedUnprojectJacobians3) { testUnprojectJacobians3<basalt::UnifiedCamera<double>>(); }
TEST(CameraTestCase, UnifiedUnprojectJacobiansFloat3) { testUnprojectJacobians3<basalt::UnifiedCamera<float>>(); }

TEST(CameraTestCase, ExtendedUnifiedUnprojectJacobians3) {
  testUnprojectJacobians3<basalt::ExtendedUnifiedCamera<double>>();
}
TEST(CameraTestCase, ExtendedUnifiedUnprojectJacobiansFloat3) {
  testUnprojectJacobians3<basalt::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtUnprojectJacobians3) {
  testUnprojectJacobians3<basalt::KannalaBrandtCamera4<double>>();
}
// TEST(CameraTestCase, KannalaBrandtUnprojectJacobiansFloat3) {
//  test_unproject_jacobians3<basalt::KannalaBrandtCamera4<float>>();
//}

TEST(CameraTestCase, DoubleSphereUnprojectJacobians3) { testUnprojectJacobians3<basalt::DoubleSphereCamera<double>>(); }
// TEST(CameraTestCase, DoubleSphereUnprojectJacobiansFloat3) {
//  test_unproject_jacobians3<basalt::DoubleSphereCamera<float>>();
//}

TEST(CameraTestCase, FovUnprojectJacobians3) { testUnprojectJacobians3<basalt::FovCamera<double>>(); }
TEST(CameraTestCase, FovUnprojectJacobiansFloat3) { testUnprojectJacobians3<basalt::FovCamera<float>>(); }

////////////////////////////////////////////////////////////////

template <typename CamT>
void testStereographicProjectJacobian() {
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;
  using Mat24 = typename CamT::Mat24;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec4 p(x, y, 5, 0.23424);

      Mat24 J_p;

      Vec2 res1 = CamT::project(p, &J_p);
      Vec2 res2 = CamT::project(p);

      ASSERT_TRUE(res1.isApprox(res2)) << "res1 " << res1.transpose() << " res2 " << res2.transpose();

      test_jacobian(
          "d_r_d_p", J_p, [&](const Vec4 &x) { return CamT::project(p + x); }, Vec4::Zero());
    }
  }
}

template <typename CamT>
void testStereographicProjectUnproject() {
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec4 p(x, y, 5, 0.23424);

      Vec4 p_normalized = Vec4::Zero();
      p_normalized.template head<3>() = p.template head<3>().normalized();
      Vec2 res = CamT::project(p);
      Vec4 p_uproj = CamT::unproject(res);

      ASSERT_TRUE(p_normalized.isApprox(p_uproj))
          << "p_normalized " << p_normalized.transpose() << " p_uproj " << p_uproj.transpose();
    }
  }
}

template <typename CamT>
void testStereographicUnprojectJacobian() {
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat42 = typename CamT::Mat42;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec4 p_3d(x, y, 5, 0.23424);

      Vec2 p = CamT::project(p_3d);

      Mat42 J_p;

      Vec4 res1 = CamT::unproject(p, &J_p);
      Vec4 res2 = CamT::unproject(p);

      ASSERT_TRUE(res1.isApprox(res2)) << "res1 " << res1.transpose() << " res2 " << res2.transpose();

      test_jacobian(
          "d_r_d_p", J_p, [&](const Vec2 &x) { return CamT::unproject(p + x); }, Vec2::Zero());
    }
  }
}

TEST(CameraTestCase, StereographicParamProjectJacobians) {
  testStereographicProjectJacobian<basalt::StereographicParam<double>>();
}
TEST(CameraTestCase, StereographicParamProjectJacobiansFloat) {
  testStereographicProjectJacobian<basalt::StereographicParam<float>>();
}

TEST(CameraTestCase, StereographicParamProjectUnproject) {
  testStereographicProjectUnproject<basalt::StereographicParam<double>>();
}
TEST(CameraTestCase, StereographicParamProjectUnprojectFloat) {
  testStereographicProjectUnproject<basalt::StereographicParam<float>>();
}

TEST(CameraTestCase, StereographicParamUnprojectJacobians) {
  testStereographicUnprojectJacobian<basalt::StereographicParam<double>>();
}
TEST(CameraTestCase, StereographicParamUnprojectJacobiansFloat) {
  testStereographicUnprojectJacobian<basalt::StereographicParam<float>>();
}

template <class Scalar, int N>
void testEvalOrReference() {
  using VecType = Eigen::Matrix<Scalar, N, 1>;
  using MapType = Eigen::Map<Eigen::Matrix<Scalar, N, 1>>;
  using MatType = Eigen::Matrix<Scalar, N, N>;

  Scalar raw_array[N];
  MapType mapped_p(raw_array);
  VecType p1, p2;
  MatType m1;
  p1.setRandom();
  p2.setRandom();
  m1.setRandom();
  mapped_p.setRandom();

  // Non-evaluated sum
  auto sum = p1 + p2;
  typename basalt::EvalOrReference<decltype(sum)>::Type res1(sum);
  static_assert(std::is_same_v<VecType, decltype(res1)>);

  // Non-evaluated operations with matrix
  auto affine = m1 * p1 + p2;
  typename basalt::EvalOrReference<decltype(affine)>::Type res2(affine);
  static_assert(std::is_same_v<VecType, decltype(res2)>);

  // Vector: Should be reference. No data copy.
  typename basalt::EvalOrReference<decltype(p1)>::Type res3(p1);
  static_assert(std::is_same_v<const VecType &, decltype(res3)>);
  ASSERT_EQ(&res3[0], &p1[0]);

  // Map: Should be reference. No data copy.
  typename basalt::EvalOrReference<decltype(mapped_p)>::Type res4(mapped_p);
  static_assert(std::is_same_v<const Eigen::MatrixBase<MapType> &, decltype(res4)>);
  ASSERT_EQ(&res4[0], &mapped_p[0]);

  // Map with standard Eigen eval: Copies data. Should not be used in the code.
  typename Eigen::internal::eval<MapType>::type res5(mapped_p);
  static_assert(std::is_same_v<VecType, decltype(res5)>);
  ASSERT_NE(&res5[0], &mapped_p[0]);
}

TEST(CameraTestCase, EvalOrReferenceTypeCast) {
  testEvalOrReference<double, 3>();
  testEvalOrReference<double, 4>();
  testEvalOrReference<float, 3>();
  testEvalOrReference<float, 4>();
}
