/**
 * Sophus SE(3): 3D Rigid Body Transformation Lie Group
 *
 * This example demonstrates:
 * 1. Creating SE(3) elements
 * 2. Exponential and log maps (twist vectors)
 * 3. Transformation composition
 * 4. Inverse transformation
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

int main() {
    cout << "=== Sophus SE(3): 3D Rigid Body Transformation ===\n" << endl;
    cout << fixed << setprecision(4);

    // =========================================================================
    // 1. Creating SE(3) Elements
    // =========================================================================
    cout << "1. Creating SE(3) Elements\n" << string(50, '-') << endl;

    // From identity
    Sophus::SE3d T_identity;
    cout << "Identity:\n" << T_identity.matrix() << endl;

    // From SO(3) and translation
    Sophus::SO3d R = Sophus::SO3d::exp(Vector3d(0, 0, M_PI / 4));  // 45° Z
    Vector3d t(1.0, 2.0, 3.0);
    Sophus::SE3d T_from_Rt(R, t);
    cout << "\nFrom R and t:\n" << T_from_Rt.matrix() << endl;

    // From 4x4 matrix
    Matrix4d mat = Matrix4d::Identity();
    mat.block<3, 3>(0, 0) = R.matrix();
    mat.block<3, 1>(0, 3) = t;
    Sophus::SE3d T_from_mat(mat);
    cout << "\nFrom 4x4 matrix:\n" << T_from_mat.matrix() << endl;

    // From quaternion and translation
    Quaterniond q = R.unit_quaternion();
    Sophus::SE3d T_from_qt(q, t);
    cout << "\nFrom quaternion and t:\n" << T_from_qt.matrix() << endl;

    // =========================================================================
    // 2. Accessing Components
    // =========================================================================
    cout << "\n2. Accessing Components\n" << string(50, '-') << endl;

    cout << "Rotation (SO3):\n" << T_from_Rt.so3().matrix() << endl;
    cout << "\nTranslation: " << T_from_Rt.translation().transpose() << endl;
    cout << "\nQuaternion (w,x,y,z): "
         << T_from_Rt.unit_quaternion().w() << ", "
         << T_from_Rt.unit_quaternion().x() << ", "
         << T_from_Rt.unit_quaternion().y() << ", "
         << T_from_Rt.unit_quaternion().z() << endl;

    // =========================================================================
    // 3. Exponential Map: se(3) → SE(3)
    // =========================================================================
    cout << "\n3. Exponential Map: se(3) → SE(3)\n" << string(50, '-') << endl;

    // Twist vector: [translation_part, rotation_part] (6D)
    // Note: Sophus uses [rotation, translation] order internally
    Matrix<double, 6, 1> xi;
    xi << 0.1, 0.2, 0.3,   // rotation (omega)
          0.4, 0.5, 0.6;   // translation (v)
    cout << "Twist vector xi: " << xi.transpose() << endl;

    // Exponential map
    Sophus::SE3d T_exp = Sophus::SE3d::exp(xi);
    cout << "\nexp(xi) = SE(3):\n" << T_exp.matrix() << endl;

    // =========================================================================
    // 4. Log Map: SE(3) → se(3)
    // =========================================================================
    cout << "\n4. Log Map: SE(3) → se(3)\n" << string(50, '-') << endl;

    Matrix<double, 6, 1> xi_log = T_exp.log();
    cout << "log(T) = xi: " << xi_log.transpose() << endl;

    // Verify round-trip
    cout << "\nRound-trip verification:" << endl;
    cout << "  Original xi:  " << xi.transpose() << endl;
    cout << "  log(exp(xi)): " << xi_log.transpose() << endl;
    cout << "  Difference: " << (xi - xi_log).norm() << endl;

    // =========================================================================
    // 5. Transformation Composition
    // =========================================================================
    cout << "\n5. Transformation Composition\n" << string(50, '-') << endl;

    Sophus::SE3d T_a(Sophus::SO3d::exp(Vector3d(0.1, 0, 0)), Vector3d(1, 0, 0));
    Sophus::SE3d T_b(Sophus::SO3d::exp(Vector3d(0, 0.1, 0)), Vector3d(0, 1, 0));

    cout << "T_a:\n" << T_a.matrix() << endl;
    cout << "\nT_b:\n" << T_b.matrix() << endl;

    // Composition: T_ab = T_a * T_b (first T_b, then T_a in world frame)
    Sophus::SE3d T_ab = T_a * T_b;
    cout << "\nT_a * T_b:\n" << T_ab.matrix() << endl;

    // Note: Non-commutative
    Sophus::SE3d T_ba = T_b * T_a;
    cout << "\nT_b * T_a:\n" << T_ba.matrix() << endl;

    // =========================================================================
    // 6. Inverse Transformation
    // =========================================================================
    cout << "\n6. Inverse Transformation\n" << string(50, '-') << endl;

    Sophus::SE3d T_inv = T_from_Rt.inverse();
    cout << "T:\n" << T_from_Rt.matrix() << endl;
    cout << "\nT^(-1):\n" << T_inv.matrix() << endl;

    // Verify: T * T^(-1) = I
    Sophus::SE3d T_check = T_from_Rt * T_inv;
    cout << "\nT * T^(-1) (should be identity):\n" << T_check.matrix() << endl;

    // Note: T^(-1) = [R^T, -R^T * t]
    cout << "\nManual inverse:" << endl;
    cout << "  R^T:\n" << T_from_Rt.so3().inverse().matrix() << endl;
    cout << "  -R^T * t: " << (-T_from_Rt.so3().inverse() * T_from_Rt.translation()).transpose() << endl;

    // =========================================================================
    // 7. Transform a Point
    // =========================================================================
    cout << "\n7. Transform a Point\n" << string(50, '-') << endl;

    Vector3d p_local(1.0, 0.0, 0.0);
    cout << "Point in local frame: " << p_local.transpose() << endl;

    // Transform: p_world = R * p_local + t
    Vector3d p_world = T_from_Rt * p_local;
    cout << "Point in world frame: " << p_world.transpose() << endl;

    // Transform back
    Vector3d p_back = T_from_Rt.inverse() * p_world;
    cout << "Transformed back: " << p_back.transpose() << endl;

    // =========================================================================
    // 8. Adjoint Representation
    // =========================================================================
    cout << "\n8. Adjoint Representation\n" << string(50, '-') << endl;

    // Adjoint matrix (6x6)
    Matrix<double, 6, 6> Adj = T_from_Rt.Adj();
    cout << "Adjoint matrix (6x6):\n" << Adj << endl;

    // Property: T * exp(xi) = exp(Adj * xi) * T
    Matrix<double, 6, 1> xi_small;
    xi_small << 0.01, 0.02, 0.03, 0.01, 0.02, 0.03;

    Sophus::SE3d lhs = T_from_Rt * Sophus::SE3d::exp(xi_small);
    Sophus::SE3d rhs = Sophus::SE3d::exp(Adj * xi_small) * T_from_Rt;

    cout << "\nVerify T * exp(xi) = exp(Adj*xi) * T:" << endl;
    cout << "  LHS translation: " << lhs.translation().transpose() << endl;
    cout << "  RHS translation: " << rhs.translation().transpose() << endl;
    cout << "  Difference: " << (lhs.matrix() - rhs.matrix()).norm() << endl;

    // =========================================================================
    // 9. Practical Example: Camera Poses
    // =========================================================================
    cout << "\n9. Practical Example: Camera Poses\n" << string(50, '-') << endl;

    // Camera 0 at origin, looking along Z
    Sophus::SE3d T_w_c0 = Sophus::SE3d();

    // Camera 1: moved 1m forward (Z) and rotated 30° around Y
    Sophus::SE3d T_c0_c1(
        Sophus::SO3d::exp(Vector3d(0, M_PI / 6, 0)),
        Vector3d(0, 0, 1)
    );
    Sophus::SE3d T_w_c1 = T_w_c0 * T_c0_c1;

    cout << "Camera 0 pose (world):\n" << T_w_c0.matrix() << endl;
    cout << "\nCamera 1 pose (world):\n" << T_w_c1.matrix() << endl;

    // Relative pose between cameras
    Sophus::SE3d T_c1_c0 = T_w_c1.inverse() * T_w_c0;
    cout << "\nRelative pose C1 -> C0:\n" << T_c1_c0.matrix() << endl;

    // Point in world frame
    Vector3d p_world_pt(0, 0, 2);
    cout << "\nPoint in world: " << p_world_pt.transpose() << endl;

    // Point in each camera frame
    Vector3d p_c0 = T_w_c0.inverse() * p_world_pt;
    Vector3d p_c1 = T_w_c1.inverse() * p_world_pt;
    cout << "Point in C0 frame: " << p_c0.transpose() << endl;
    cout << "Point in C1 frame: " << p_c1.transpose() << endl;

    cout << "\n=== Complete ===" << endl;

    return 0;
}
