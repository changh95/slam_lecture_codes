/**
 * Perturbation Model for Optimization
 *
 * This example demonstrates:
 * 1. Left vs Right perturbation
 * 2. Why perturbation is used in optimization
 * 3. Numerical Jacobian computation
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Core>
#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

// A simple residual function: distance to a target point
Vector3d residual(const Sophus::SE3d& T, const Vector3d& p_world, const Vector3d& p_observed) {
    // Transform world point to camera frame and compare with observation
    Vector3d p_camera = T.inverse() * p_world;
    return p_camera - p_observed;
}

int main() {
    cout << "=== Perturbation Model for Optimization ===\n" << endl;
    cout << fixed << setprecision(6);

    // =========================================================================
    // 1. Why Perturbation?
    // =========================================================================
    cout << "1. Why Perturbation?\n" << string(50, '-') << endl;

    cout << R"(
In optimization, we update parameters iteratively:
    x_new = x + delta

For Lie Groups (SO3, SE3), simple addition doesn't work because:
- Rotation matrices must stay orthogonal
- Adding arbitrary values breaks the group structure

Instead, we use perturbation via exponential map:
    T_new = exp(delta) * T     (left perturbation)
    T_new = T * exp(delta)     (right perturbation)

This keeps T_new on the manifold!
)" << endl;

    // =========================================================================
    // 2. Left Perturbation
    // =========================================================================
    cout << "\n2. Left Perturbation: T_new = exp(delta) * T\n" << string(50, '-') << endl;

    // Current pose
    Sophus::SE3d T(
        Sophus::SO3d::exp(Vector3d(0.1, 0.2, 0.3)),
        Vector3d(1, 2, 3)
    );
    cout << "Current pose T:\n" << T.matrix() << endl;

    // Small perturbation in tangent space
    Matrix<double, 6, 1> delta_left;
    delta_left << 0.01, 0.02, 0.03,  // rotation
                  0.1, 0.2, 0.3;      // translation
    cout << "\nPerturbation delta: " << delta_left.transpose() << endl;

    // Left perturbation: perturbs in world frame
    Sophus::SE3d T_left = Sophus::SE3d::exp(delta_left) * T;
    cout << "\nLeft perturbed T_new = exp(delta) * T:\n" << T_left.matrix() << endl;

    // =========================================================================
    // 3. Right Perturbation
    // =========================================================================
    cout << "\n3. Right Perturbation: T_new = T * exp(delta)\n" << string(50, '-') << endl;

    // Right perturbation: perturbs in local frame
    Sophus::SE3d T_right = T * Sophus::SE3d::exp(delta_left);
    cout << "Right perturbed T_new = T * exp(delta):\n" << T_right.matrix() << endl;

    // Note the difference
    cout << "\nDifference between left and right:" << endl;
    cout << "  Translation diff: "
         << (T_left.translation() - T_right.translation()).norm() << endl;
    cout << "  Rotation diff: "
         << (T_left.so3().inverse() * T_right.so3()).log().norm() << endl;

    // =========================================================================
    // 4. Numerical Jacobian (Left Perturbation)
    // =========================================================================
    cout << "\n4. Numerical Jacobian (Left Perturbation)\n" << string(50, '-') << endl;

    // Setup: point observation problem
    Vector3d p_world(4, 5, 6);
    Vector3d p_observed(1.0, 1.5, 2.0);  // Simulated observation

    cout << "World point: " << p_world.transpose() << endl;
    cout << "Observed point: " << p_observed.transpose() << endl;

    // Current residual
    Vector3d r0 = residual(T, p_world, p_observed);
    cout << "\nCurrent residual: " << r0.transpose() << endl;

    // Compute numerical Jacobian: J = dr/d(delta)
    // J is 3x6 (residual dim x tangent space dim)
    Matrix<double, 3, 6> J_numerical;
    double eps = 1e-8;

    for (int i = 0; i < 6; ++i) {
        Matrix<double, 6, 1> delta_eps = Matrix<double, 6, 1>::Zero();
        delta_eps(i) = eps;

        // Left perturbation
        Sophus::SE3d T_plus = Sophus::SE3d::exp(delta_eps) * T;
        Vector3d r_plus = residual(T_plus, p_world, p_observed);

        // Finite difference
        J_numerical.col(i) = (r_plus - r0) / eps;
    }

    cout << "\nNumerical Jacobian (3x6):\n" << J_numerical << endl;

    // =========================================================================
    // 5. Gauss-Newton Update
    // =========================================================================
    cout << "\n5. Gauss-Newton Update\n" << string(50, '-') << endl;

    cout << R"(
In Gauss-Newton optimization:
    delta = -(J^T * J)^(-1) * J^T * r

Then update:
    T_new = exp(delta) * T
)" << endl;

    // Compute update
    Matrix<double, 6, 6> H = J_numerical.transpose() * J_numerical;
    Matrix<double, 6, 1> b = -J_numerical.transpose() * r0;

    // Solve H * delta = b
    Matrix<double, 6, 1> delta_gn = H.ldlt().solve(b);
    cout << "Computed delta: " << delta_gn.transpose() << endl;

    // Apply update
    Sophus::SE3d T_updated = Sophus::SE3d::exp(delta_gn) * T;

    // New residual
    Vector3d r_new = residual(T_updated, p_world, p_observed);
    cout << "\nResidual before: " << r0.norm() << endl;
    cout << "Residual after:  " << r_new.norm() << endl;
    cout << "Improvement: " << (1 - r_new.norm() / r0.norm()) * 100 << "%" << endl;

    // =========================================================================
    // 6. Iterative Optimization
    // =========================================================================
    cout << "\n6. Iterative Optimization\n" << string(50, '-') << endl;

    // Reset to initial pose
    Sophus::SE3d T_opt = T;
    int max_iters = 10;

    cout << "Iteration | Residual Norm" << endl;
    cout << string(30, '-') << endl;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Current residual
        Vector3d r = residual(T_opt, p_world, p_observed);
        cout << "    " << iter << "     |   " << r.norm() << endl;

        if (r.norm() < 1e-10) {
            cout << "Converged!" << endl;
            break;
        }

        // Compute numerical Jacobian
        Matrix<double, 3, 6> J;
        for (int i = 0; i < 6; ++i) {
            Matrix<double, 6, 1> d = Matrix<double, 6, 1>::Zero();
            d(i) = eps;
            Sophus::SE3d T_p = Sophus::SE3d::exp(d) * T_opt;
            J.col(i) = (residual(T_p, p_world, p_observed) - r) / eps;
        }

        // Gauss-Newton step
        Matrix<double, 6, 6> H = J.transpose() * J;
        Matrix<double, 6, 1> b = -J.transpose() * r;
        Matrix<double, 6, 1> delta = H.ldlt().solve(b);

        // Update
        T_opt = Sophus::SE3d::exp(delta) * T_opt;
    }

    cout << "\nOptimized pose:\n" << T_opt.matrix() << endl;

    // Verify: T_opt^(-1) * p_world should equal p_observed
    Vector3d p_check = T_opt.inverse() * p_world;
    cout << "\nT_opt^(-1) * p_world: " << p_check.transpose() << endl;
    cout << "Expected (p_observed): " << p_observed.transpose() << endl;

    // =========================================================================
    // 7. SO(3) Perturbation
    // =========================================================================
    cout << "\n7. SO(3) Perturbation\n" << string(50, '-') << endl;

    Sophus::SO3d R = Sophus::SO3d::exp(Vector3d(0.5, 0.3, 0.2));
    cout << "Original rotation R:\n" << R.matrix() << endl;

    Vector3d delta_rot(0.01, 0.02, 0.03);
    cout << "\nSmall rotation delta: " << delta_rot.transpose() << endl;

    // Left perturbation
    Sophus::SO3d R_left = Sophus::SO3d::exp(delta_rot) * R;
    cout << "\nLeft: exp(delta) * R:\n" << R_left.matrix() << endl;

    // Right perturbation
    Sophus::SO3d R_right = R * Sophus::SO3d::exp(delta_rot);
    cout << "\nRight: R * exp(delta):\n" << R_right.matrix() << endl;

    cout << "\n=== Complete ===" << endl;

    return 0;
}
