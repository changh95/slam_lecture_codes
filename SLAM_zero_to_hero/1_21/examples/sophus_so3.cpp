/**
 * Sophus SO(3): 3D Rotation Lie Group
 *
 * This example demonstrates:
 * 1. Creating SO(3) elements
 * 2. Exponential and log maps
 * 3. Rotation composition
 * 4. Interpolation (slerp)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

using namespace std;
using namespace Eigen;

int main() {
    cout << "=== Sophus SO(3): 3D Rotation Lie Group ===\n" << endl;
    cout << fixed << setprecision(4);

    // =========================================================================
    // 1. Creating SO(3) Elements
    // =========================================================================
    cout << "1. Creating SO(3) Elements\n" << string(50, '-') << endl;

    // From identity
    Sophus::SO3d R_identity;
    cout << "Identity:\n" << R_identity.matrix() << endl;

    // From rotation matrix
    Matrix3d rot_mat = AngleAxisd(M_PI / 4, Vector3d::UnitZ()).toRotationMatrix();
    Sophus::SO3d R_from_mat(rot_mat);
    cout << "\nFrom rotation matrix (45° around Z):\n" << R_from_mat.matrix() << endl;

    // From quaternion
    Quaterniond quat(AngleAxisd(M_PI / 3, Vector3d::UnitX()));
    Sophus::SO3d R_from_quat(quat);
    cout << "\nFrom quaternion (60° around X):\n" << R_from_quat.matrix() << endl;

    // =========================================================================
    // 2. Exponential Map: so(3) → SO(3)
    // =========================================================================
    cout << "\n2. Exponential Map: so(3) → SO(3)\n" << string(50, '-') << endl;

    // Rotation vector (axis * angle) in tangent space
    Vector3d omega(0.1, 0.2, 0.3);  // Small rotation
    cout << "Rotation vector (omega): " << omega.transpose() << endl;
    cout << "  |omega| = " << omega.norm() << " rad = "
         << omega.norm() * 180 / M_PI << " deg" << endl;

    // Exponential map: tangent space → manifold
    Sophus::SO3d R_exp = Sophus::SO3d::exp(omega);
    cout << "\nexp(omega) = SO(3):\n" << R_exp.matrix() << endl;

    // Get quaternion representation
    cout << "\nQuaternion (w, x, y, z): "
         << R_exp.unit_quaternion().w() << ", "
         << R_exp.unit_quaternion().x() << ", "
         << R_exp.unit_quaternion().y() << ", "
         << R_exp.unit_quaternion().z() << endl;

    // =========================================================================
    // 3. Log Map: SO(3) → so(3)
    // =========================================================================
    cout << "\n3. Log Map: SO(3) → so(3)\n" << string(50, '-') << endl;

    // Log map: manifold → tangent space
    Vector3d omega_log = R_exp.log();
    cout << "log(R) = omega: " << omega_log.transpose() << endl;

    // Verify round-trip
    cout << "\nRound-trip verification:" << endl;
    cout << "  Original omega:  " << omega.transpose() << endl;
    cout << "  log(exp(omega)): " << omega_log.transpose() << endl;
    cout << "  Difference: " << (omega - omega_log).norm() << endl;

    // =========================================================================
    // 4. Rotation Composition
    // =========================================================================
    cout << "\n4. Rotation Composition\n" << string(50, '-') << endl;

    Sophus::SO3d R_a = Sophus::SO3d::exp(Vector3d(0.1, 0, 0));  // Small X rotation
    Sophus::SO3d R_b = Sophus::SO3d::exp(Vector3d(0, 0.2, 0));  // Small Y rotation

    cout << "R_a (rotation around X):\n" << R_a.matrix() << endl;
    cout << "\nR_b (rotation around Y):\n" << R_b.matrix() << endl;

    // Composition: R_ab = R_a * R_b
    Sophus::SO3d R_ab = R_a * R_b;
    cout << "\nR_a * R_b:\n" << R_ab.matrix() << endl;

    // Note: Rotation is NOT commutative
    Sophus::SO3d R_ba = R_b * R_a;
    cout << "\nR_b * R_a:\n" << R_ba.matrix() << endl;
    cout << "\nR_a * R_b == R_b * R_a? "
         << ((R_ab.matrix() - R_ba.matrix()).norm() < 1e-10 ? "Yes" : "No") << endl;

    // =========================================================================
    // 5. Inverse
    // =========================================================================
    cout << "\n5. Inverse\n" << string(50, '-') << endl;

    Sophus::SO3d R_inv = R_ab.inverse();
    cout << "R_ab inverse:\n" << R_inv.matrix() << endl;

    // Verify: R * R^(-1) = I
    Sophus::SO3d R_check = R_ab * R_inv;
    cout << "\nR * R^(-1) (should be identity):\n" << R_check.matrix() << endl;

    // =========================================================================
    // 6. Rotate a Point
    // =========================================================================
    cout << "\n6. Rotate a Point\n" << string(50, '-') << endl;

    Vector3d point(1.0, 0.0, 0.0);
    cout << "Original point: " << point.transpose() << endl;

    // Rotate using SO(3)
    Vector3d rotated = R_from_mat * point;
    cout << "After 45° Z rotation: " << rotated.transpose() << endl;

    // Equivalent to matrix multiplication
    Vector3d rotated_mat = R_from_mat.matrix() * point;
    cout << "Using matrix: " << rotated_mat.transpose() << endl;

    // =========================================================================
    // 7. Interpolation (Slerp)
    // =========================================================================
    cout << "\n7. Interpolation (Slerp)\n" << string(50, '-') << endl;

    Sophus::SO3d R_start = Sophus::SO3d::exp(Vector3d(0, 0, 0));
    Sophus::SO3d R_end = Sophus::SO3d::exp(Vector3d(0, 0, M_PI / 2));  // 90° Z

    cout << "Start rotation (identity):\n" << R_start.matrix() << endl;
    cout << "\nEnd rotation (90° Z):\n" << R_end.matrix() << endl;

    // Interpolate at different t values
    cout << "\nInterpolation:" << endl;
    for (double t = 0.0; t <= 1.0; t += 0.25) {
        // Slerp: R(t) = R_start * exp(t * log(R_start^(-1) * R_end))
        Vector3d delta = (R_start.inverse() * R_end).log();
        Sophus::SO3d R_t = R_start * Sophus::SO3d::exp(t * delta);

        // Extract angle
        double angle = R_t.log().norm() * 180 / M_PI;
        cout << "  t=" << t << ": angle = " << angle << " degrees" << endl;
    }

    // =========================================================================
    // 8. Hat and Vee Operators
    // =========================================================================
    cout << "\n8. Hat and Vee Operators\n" << string(50, '-') << endl;

    Vector3d v(1, 2, 3);
    cout << "Vector v: " << v.transpose() << endl;

    // Hat operator: vector → skew-symmetric matrix
    Matrix3d v_hat = Sophus::SO3d::hat(v);
    cout << "\nhat(v) (skew-symmetric matrix):\n" << v_hat << endl;

    // Vee operator: skew-symmetric matrix → vector
    Vector3d v_vee = Sophus::SO3d::vee(v_hat);
    cout << "\nvee(hat(v)): " << v_vee.transpose() << endl;

    // Property: hat(v) * p = v × p (cross product)
    Vector3d p(4, 5, 6);
    cout << "\nhat(v) * p: " << (v_hat * p).transpose() << endl;
    cout << "v × p:      " << v.cross(p).transpose() << endl;

    cout << "\n=== Complete ===" << endl;

    return 0;
}
