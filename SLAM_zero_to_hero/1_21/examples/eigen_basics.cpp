/**
 * Eigen Basics: Rotation and Transformation
 *
 * This example demonstrates:
 * 1. Rotation matrices
 * 2. Quaternions
 * 3. Euler angles
 * 4. Transformation matrices
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main() {
    cout << "=== Eigen Basics: Rotation and Transformation ===\n" << endl;
    cout << fixed << setprecision(4);

    // =========================================================================
    // 1. Rotation Matrix
    // =========================================================================
    cout << "1. Rotation Matrix\n" << string(40, '-') << endl;

    // Create rotation matrix from angle-axis
    double angle = M_PI / 4;  // 45 degrees
    Vector3d axis(0, 0, 1);   // Z-axis
    Matrix3d R_z = AngleAxisd(angle, axis).toRotationMatrix();

    cout << "Rotation around Z-axis by 45 degrees:\n" << R_z << endl;

    // Verify orthogonality: R * R^T = I
    cout << "\nR * R^T (should be identity):\n" << R_z * R_z.transpose() << endl;

    // Verify determinant = 1
    cout << "det(R) = " << R_z.determinant() << " (should be 1)" << endl;

    // =========================================================================
    // 2. Rotation from Euler Angles (ZYX convention)
    // =========================================================================
    cout << "\n2. Euler Angles (ZYX/yaw-pitch-roll)\n" << string(40, '-') << endl;

    double yaw = M_PI / 6;    // 30 degrees
    double pitch = M_PI / 12; // 15 degrees
    double roll = M_PI / 18;  // 10 degrees

    // ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Matrix3d R_euler = (AngleAxisd(yaw, Vector3d::UnitZ()) *
                        AngleAxisd(pitch, Vector3d::UnitY()) *
                        AngleAxisd(roll, Vector3d::UnitX())).toRotationMatrix();

    cout << "Rotation from yaw=" << yaw * 180 / M_PI
         << ", pitch=" << pitch * 180 / M_PI
         << ", roll=" << roll * 180 / M_PI << " degrees:\n" << R_euler << endl;

    // Extract Euler angles back
    Vector3d euler = R_euler.eulerAngles(2, 1, 0);  // ZYX order
    cout << "\nExtracted Euler angles (rad): " << euler.transpose() << endl;
    cout << "Extracted Euler angles (deg): " << (euler * 180 / M_PI).transpose() << endl;

    // =========================================================================
    // 3. Quaternion
    // =========================================================================
    cout << "\n3. Quaternion\n" << string(40, '-') << endl;

    // Create quaternion from rotation matrix
    Quaterniond q(R_euler);
    cout << "Quaternion (w, x, y, z): " << q.w() << ", "
         << q.x() << ", " << q.y() << ", " << q.z() << endl;

    // Verify unit quaternion
    cout << "Quaternion norm: " << q.norm() << " (should be 1)" << endl;

    // Convert back to rotation matrix
    Matrix3d R_from_q = q.toRotationMatrix();
    cout << "\nRotation matrix from quaternion:\n" << R_from_q << endl;

    // Quaternion multiplication (composition)
    Quaterniond q2(AngleAxisd(M_PI / 2, Vector3d::UnitZ()));
    Quaterniond q_composed = q * q2;
    cout << "\nComposed quaternion: " << q_composed.w() << ", "
         << q_composed.x() << ", " << q_composed.y() << ", " << q_composed.z() << endl;

    // =========================================================================
    // 4. Angle-Axis
    // =========================================================================
    cout << "\n4. Angle-Axis\n" << string(40, '-') << endl;

    AngleAxisd aa(R_euler);
    cout << "Angle: " << aa.angle() * 180 / M_PI << " degrees" << endl;
    cout << "Axis: " << aa.axis().transpose() << endl;

    // Rotation vector (angle * axis)
    Vector3d rot_vec = aa.angle() * aa.axis();
    cout << "Rotation vector: " << rot_vec.transpose() << endl;

    // =========================================================================
    // 5. Transformation Matrix (4x4)
    // =========================================================================
    cout << "\n5. Transformation Matrix\n" << string(40, '-') << endl;

    Vector3d translation(1.0, 2.0, 3.0);

    // Create 4x4 transformation matrix
    Matrix4d T = Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R_euler;
    T.block<3, 1>(0, 3) = translation;

    cout << "Transformation matrix:\n" << T << endl;

    // Using Isometry3d (more efficient)
    Isometry3d T_iso = Isometry3d::Identity();
    T_iso.rotate(R_euler);
    T_iso.pretranslate(translation);

    cout << "\nIsometry3d matrix:\n" << T_iso.matrix() << endl;

    // =========================================================================
    // 6. Transform a Point
    // =========================================================================
    cout << "\n6. Transform a Point\n" << string(40, '-') << endl;

    Vector3d p_local(1.0, 0.0, 0.0);
    cout << "Point in local frame: " << p_local.transpose() << endl;

    // Transform using rotation only
    Vector3d p_rotated = R_euler * p_local;
    cout << "After rotation: " << p_rotated.transpose() << endl;

    // Transform using full transformation
    Vector3d p_transformed = R_euler * p_local + translation;
    cout << "After rotation + translation: " << p_transformed.transpose() << endl;

    // Using Isometry3d
    Vector3d p_iso = T_iso * p_local;
    cout << "Using Isometry3d: " << p_iso.transpose() << endl;

    // =========================================================================
    // 7. Inverse Transformation
    // =========================================================================
    cout << "\n7. Inverse Transformation\n" << string(40, '-') << endl;

    Isometry3d T_inv = T_iso.inverse();
    cout << "Inverse transformation:\n" << T_inv.matrix() << endl;

    // Verify: T * T^(-1) = I
    cout << "\nT * T^(-1):\n" << (T_iso * T_inv).matrix() << endl;

    // Transform back
    Vector3d p_back = T_inv * p_transformed;
    cout << "\nPoint transformed back: " << p_back.transpose()
         << " (should be " << p_local.transpose() << ")" << endl;

    cout << "\n=== Complete ===" << endl;

    return 0;
}
