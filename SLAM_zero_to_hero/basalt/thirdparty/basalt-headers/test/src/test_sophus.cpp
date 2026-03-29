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

#include <basalt/utils/sophus_utils.hpp>

#include <sophus/se2.hpp>
#include <sophus/sim2.hpp>

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(SophusUtilsCase, RightJacobianSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d J_a = Sophus::rightJacobianSO3(phi);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianSO3", J_a,
      [&](const Eigen::Vector3d &x) { return (Sophus::SO3d::exp(phi).inverse() * Sophus::SO3d::exp(phi + x)).log(); },
      x0);
}

TEST(SophusUtilsCase, RightJacobianInvSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d J_a = Sophus::rightJacobianInvSO3(phi);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianInvSO3", J_a,
      [&](const Eigen::Vector3d &x) { return (Sophus::SO3d::exp(phi) * Sophus::SO3d::exp(x)).log(); }, x0);
}

TEST(SophusUtilsCase, LeftJacobianSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d J_a = Sophus::leftJacobianSO3(phi);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "leftJacobianSO3", J_a,
      [&](const Eigen::Vector3d &x) { return (Sophus::SO3d::exp(phi + x) * Sophus::SO3d::exp(phi).inverse()).log(); },
      x0);
}

TEST(SophusUtilsCase, LeftJacobianInvSO3) {
  Eigen::Vector3d phi;
  phi.setRandom();

  Eigen::Matrix3d J_a = Sophus::leftJacobianInvSO3(phi);

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "leftJacobianInvSO3", J_a,
      [&](const Eigen::Vector3d &x) { return (Sophus::SO3d::exp(x) * Sophus::SO3d::exp(phi)).log(); }, x0);
}

TEST(SophusUtilsCase, RightJacobianSE3Decoupled) {
  Sophus::Vector6d phi;
  phi.setRandom();

  Sophus::Matrix6d J_a = Sophus::rightJacobianSE3Decoupled(phi);

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianSE3Decoupled", J_a,
      [&](const Sophus::Vector6d &x) {
        return Sophus::se3_logd(Sophus::se3_expd(phi).inverse() * Sophus::se3_expd(phi + x));
      },
      x0);
}

TEST(SophusUtilsCase, RightJacobianInvSE3Decoupled) {
  Sophus::Vector6d phi;
  phi.setRandom();

  Sophus::Matrix6d J_a = Sophus::rightJacobianInvSE3Decoupled(phi);

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianInvSE3Decoupled", J_a,
      [&](const Sophus::Vector6d &x) { return Sophus::se3_logd(Sophus::se3_expd(phi) * Sophus::se3_expd(x)); }, x0);
}

// Verify that the adjoint definition works equally for expd and logd.
TEST(SophusUtilsCase, Adjoint) {
  Sophus::SE3d pose = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  Sophus::Matrix6d J_a = pose.inverse().Adj();

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "Adj", J_a,
      [&](const Sophus::Vector6d &x) { return Sophus::se3_logd(pose.inverse() * Sophus::se3_expd(x) * pose); }, x0);
}

TEST(SophusUtilsCase, RotTestSO3) {
  Eigen::Vector3d t1 = Eigen::Vector3d::Random();
  Eigen::Vector3d t2 = Eigen::Vector3d::Random();

  double k = 0.6234234;

  Eigen::Matrix3d J_a = Sophus::rightJacobianSO3(k * t1);
  J_a = -k * Sophus::SO3d::exp(k * t1).matrix() * Sophus::SO3d::hat(t2) * J_a;

  Eigen::Vector3d x0;
  x0.setZero();

  test_jacobian(
      "Rot Test", J_a, [&](const Eigen::Vector3d &x) { return Sophus::SO3d::exp(k * (t1 + x)) * t2; }, x0);
}

// Jacobian of the SE3 right-increment w.r.t. the decoupled left-increment.
TEST(SophusUtilsCase, incTest) {
  Sophus::SE3d T_w_i = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  Sophus::Matrix6d J_a;
  J_a.setZero();

  Eigen::Matrix3d R_w_i = T_w_i.so3().inverse().matrix();
  J_a.topLeftCorner<3, 3>() = R_w_i;
  J_a.bottomRightCorner<3, 3>() = R_w_i;

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "inc test", J_a,
      [&](const Sophus::Vector6d &x) {
        Sophus::SE3d pose1;
        pose1.so3() = Sophus::SO3d::exp(x.tail<3>()) * T_w_i.so3();
        pose1.translation() = T_w_i.translation() + x.head<3>();

        return Sophus::se3_logd(T_w_i.inverse() * pose1);
      },
      x0);
}

// Jacobian of the SE3 left-increment w.r.t. the decoupled left-increment.
TEST(SophusUtilsCase, incTest2) {
  Sophus::SE3d pose = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  Sophus::Matrix6d J_a;
  J_a.setIdentity();

  J_a.topRightCorner<3, 3>() = Sophus::SO3d::hat(pose.translation());

  Sophus::Vector6d x0;
  x0.setZero();

  test_jacobian(
      "inc test", J_a,
      [&](const Sophus::Vector6d &x) {
        Sophus::SE3d pose1;
        pose1.so3() = Sophus::SO3d::exp(x.tail<3>()) * pose.so3();
        pose1.translation() = pose.translation() + x.head<3>();

        return Sophus::se3_logd(pose1 * pose.inverse());
      },
      x0);
}

TEST(SophusUtilsCase, SO2Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 1> J_a;
  J_a[0] = -phi[1];
  J_a[1] = phi[0];

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", J_a, [&](const Eigen::Matrix<double, 1, 1> &x) { return Sophus::SO2d::exp(x[0]) * phi; }, x0);
}

TEST(SophusUtilsCase, Se3Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 3> J_a;
  J_a.topLeftCorner<2, 2>().setIdentity();
  J_a(0, 2) = -phi[1];
  J_a(1, 2) = phi[0];

  Eigen::Matrix<double, 3, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", J_a, [&](const Eigen::Matrix<double, 3, 1> &x) { return Sophus::SE2d::exp(x) * phi; }, x0);
}

TEST(SophusUtilsCase, Sim2Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 4> J_a;
  J_a.topLeftCorner<2, 2>().setIdentity();
  J_a(0, 2) = -phi[1];
  J_a(1, 2) = phi[0];
  J_a(0, 3) = phi[0];
  J_a(1, 3) = phi[1];

  Eigen::Matrix<double, 4, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", J_a, [&](const Eigen::Matrix<double, 4, 1> &x) { return Sophus::Sim2d::exp(x) * phi; }, x0);
}

TEST(SophusUtilsCase, RxSO2Test) {
  Sophus::Vector2d phi;
  phi.setRandom();

  // std::cout << "phi " << phi.transpose() << std::endl;

  Sophus::Matrix<double, 2, 2> J_a;
  J_a(0, 0) = -phi[1];
  J_a(1, 0) = phi[0];
  J_a(0, 1) = phi[0];
  J_a(1, 1) = phi[1];

  Eigen::Matrix<double, 2, 1> x0;
  x0.setZero();

  test_jacobian(
      "inc test", J_a, [&](const Eigen::Matrix<double, 2, 1> &x) { return Sophus::RxSO2d::exp(x) * phi; }, x0);
}

TEST(SophusUtilsCase, RightJacobianSim3Decoupled) {
  Sophus::Vector7d phi;
  phi.setRandom();

  Sophus::Matrix7d J_a = Sophus::rightJacobianSim3Decoupled(phi);
  Sophus::Matrix7d J_n;

  Sophus::Vector7d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianSim3Decoupled", J_a,
      [&](const Sophus::Vector7d &x) {
        return Sophus::sim3_logd(Sophus::sim3_expd(phi).inverse() * Sophus::sim3_expd(phi + x));
      },
      x0);
}

TEST(SophusUtilsCase, RightJacobianInvSim3Decoupled) {
  Sophus::Vector7d phi;
  phi.setRandom();

  Sophus::Matrix7d J_a = Sophus::rightJacobianInvSim3Decoupled(phi);
  Sophus::Matrix7d J_n;

  Sophus::Vector7d x0;
  x0.setZero();

  test_jacobian(
      "rightJacobianInvSim3Decoupled", J_a,
      [&](const Sophus::Vector7d &x) { return Sophus::sim3_logd(Sophus::sim3_expd(phi) * Sophus::sim3_expd(x)); }, x0);
}

// Verify adjoint definition holds for decoupled log/exp.
TEST(SophusUtilsCase, AdjointSim3) {
  Sophus::Sim3d pose = Sophus::Sim3d::exp(Sophus::Vector7d::Random());

  Sophus::Matrix7d J_a = pose.inverse().Adj();

  Sophus::Vector7d x0;
  x0.setZero();

  test_jacobian(
      "AdjSim3", J_a,
      [&](const Sophus::Vector7d &x) { return Sophus::sim3_logd(pose.inverse() * Sophus::sim3_expd(x) * pose); }, x0);
}

// Note: For the relative pose tests we test different measurement error
// magnitudes. It is not sufficient to only test small errors. For example,
// with 1e-4, the tests do also pass when approximating the rightJacobian with
// identity.

TEST(SophusUtilsCase, RelPoseTestRightIncSE3) {
  Sophus::SE3d T_w_i = Sophus::SE3d::exp(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_j = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  for (const double meas_error : {1e0, 1e-1, 1e-2, 1e-4}) {
    Sophus::SE3d T_ij_meas = T_w_i.inverse() * T_w_j * Sophus::SE3d::exp(Sophus::Vector6d::Random() * meas_error);

    Sophus::SE3d T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res = Sophus::se3_logd(T_ij_meas * T_j_i);

    Sophus::Matrix6d J_T_w_i = Sophus::rightJacobianInvSE3Decoupled(res);
    Sophus::Matrix6d J_T_w_j = -J_T_w_i * T_j_i.inverse().Adj();

    Sophus::Vector6d x0;
    x0.setZero();

    test_jacobian(
        "d_res_d_T_w_i", J_T_w_i,
        [&](const Sophus::Vector6d &x) {
          Sophus::SE3d T_w_i_new = T_w_i * Sophus::se3_expd(x);

          return Sophus::se3_logd(T_ij_meas * T_w_j.inverse() * T_w_i_new);
        },
        x0);

    test_jacobian(
        "d_res_d_T_w_j", J_T_w_j,
        [&](const Sophus::Vector6d &x) {
          Sophus::SE3d T_w_j_new = T_w_j * Sophus::se3_expd(x);

          return Sophus::se3_logd(T_ij_meas * T_w_j_new.inverse() * T_w_i);
        },
        x0);
  }
}

TEST(SophusUtilsCase, RelPoseTestLeftIncSE3) {
  Sophus::SE3d T_w_i = Sophus::SE3d::exp(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_j = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  for (const double meas_error : {1e0, 1e-1, 1e-2, 1e-4}) {
    Sophus::SE3d T_ij_meas = T_w_i.inverse() * T_w_j * Sophus::SE3d::exp(Sophus::Vector6d::Random() * meas_error);

    Sophus::SE3d T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res = Sophus::se3_logd(T_ij_meas * T_j_i);

    Sophus::Matrix6d J_T_w_i = Sophus::rightJacobianInvSE3Decoupled(res);
    J_T_w_i = J_T_w_i * T_w_i.inverse().Adj();
    Sophus::Matrix6d J_T_w_j = -J_T_w_i;

    Sophus::Vector6d x0;
    x0.setZero();

    test_jacobian(
        "d_res_d_T_w_i", J_T_w_i,
        [&](const Sophus::Vector6d &x) {
          Sophus::SE3d T_w_i_new = Sophus::se3_expd(x) * T_w_i;

          return Sophus::se3_logd(T_ij_meas * T_w_j.inverse() * T_w_i_new);
        },
        x0);

    test_jacobian(
        "d_res_d_T_w_j", J_T_w_j,
        [&](const Sophus::Vector6d &x) {
          Sophus::SE3d T_w_j_new = Sophus::se3_expd(x) * T_w_j;

          return Sophus::se3_logd(T_ij_meas * T_w_j_new.inverse() * T_w_i);
        },
        x0);
  }
}

TEST(SophusUtilsCase, RelPoseTestDecoupledLeftIncSE3) {
  Sophus::SE3d T_w_i = Sophus::SE3d::exp(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_j = Sophus::SE3d::exp(Sophus::Vector6d::Random());

  for (const double meas_error : {1e0, 1e-1, 1e-2, 1e-4}) {
    Sophus::SE3d T_ij_meas = T_w_i.inverse() * T_w_j * Sophus::SE3d::exp(Sophus::Vector6d::Random() * meas_error);

    Sophus::SE3d T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector6d res = Sophus::se3_logd(T_ij_meas * T_j_i);

    Sophus::Matrix6d J_T_w_i = Sophus::rightJacobianInvSE3Decoupled(res);
    Sophus::Matrix6d J_T_w_j = -J_T_w_i * T_j_i.inverse().Adj();
    Sophus::Matrix6d rr_i = Sophus::Matrix6d::Zero();
    Sophus::Matrix6d rr_j = Sophus::Matrix6d::Zero();

    rr_i.topLeftCorner<3, 3>() = rr_i.bottomRightCorner<3, 3>() = T_w_i.so3().inverse().matrix();
    rr_j.topLeftCorner<3, 3>() = rr_j.bottomRightCorner<3, 3>() = T_w_j.so3().inverse().matrix();

    Sophus::Vector6d x0;
    x0.setZero();

    test_jacobian(
        "d_res_d_T_w_i", J_T_w_i * rr_i,
        [&](const Sophus::Vector6d &x) {
          Sophus::SE3d T_w_i_new;
          T_w_i_new.so3() = Sophus::SO3d::exp(x.tail<3>()) * T_w_i.so3();
          T_w_i_new.translation() = T_w_i.translation() + x.head<3>();

          return Sophus::se3_logd(T_ij_meas * T_w_j.inverse() * T_w_i_new);
        },
        x0);

    test_jacobian(
        "d_res_d_T_w_j", J_T_w_j * rr_j,
        [&](const Sophus::Vector6d &x) {
          Sophus::SE3d T_w_j_new;
          T_w_j_new.so3() = Sophus::SO3d::exp(x.tail<3>()) * T_w_j.so3();
          T_w_j_new.translation() = T_w_j.translation() + x.head<3>();

          return Sophus::se3_logd(T_ij_meas * T_w_j_new.inverse() * T_w_i);
        },
        x0);
  }
}

TEST(SophusUtilsCase, RelPoseTestRightIncSim3) {
  Sophus::Sim3d T_w_i = Sophus::Sim3d::exp(Sophus::Vector7d::Random());
  Sophus::Sim3d T_w_j = Sophus::Sim3d::exp(Sophus::Vector7d::Random());

  for (const double meas_error : {1e0, 1e-1, 1e-2, 1e-4}) {
    Sophus::Sim3d T_ij_meas = T_w_i.inverse() * T_w_j * Sophus::Sim3d::exp(Sophus::Vector7d::Random() * meas_error);

    Sophus::Sim3d T_j_i = T_w_j.inverse() * T_w_i;
    Sophus::Vector7d res = Sophus::sim3_logd(T_ij_meas * T_j_i);

    Sophus::Matrix7d J_T_w_i = Sophus::rightJacobianInvSim3Decoupled(res);
    Sophus::Matrix7d J_T_w_j = -J_T_w_i * T_j_i.inverse().Adj();
    Sophus::Vector7d x0 = Sophus::Vector7d::Zero();

    test_jacobian(
        "d_res_d_T_w_i", J_T_w_i,
        [&](const Sophus::Vector7d &x) {
          Sophus::Sim3d T_w_i_new = T_w_i * Sophus::sim3_expd(x);

          return Sophus::sim3_logd(T_ij_meas * T_w_j.inverse() * T_w_i_new);
        },
        x0);

    test_jacobian(
        "d_res_d_T_w_j", J_T_w_j,
        [&](const Sophus::Vector7d &x) {
          Sophus::Sim3d T_w_j_new = T_w_j * Sophus::sim3_expd(x);

          return Sophus::sim3_logd(T_ij_meas * T_w_j_new.inverse() * T_w_i);
        },
        x0);
  }
}
