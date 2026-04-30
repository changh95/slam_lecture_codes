/**
 * @file rpgo_outlier_rejection.cpp
 * @brief Kimera-RPGO의 outlier rejection 기능 데모
 *
 * 이 예제에서는:
 * 1. 정상적인 loop closure와 outlier loop closure 추가
 * 2. RPGO가 outlier를 어떻게 검출하고 제거하는지 확인
 * 3. PCM과 GNC 방법 비교
 */

#include <iostream>
#include <vector>
#include <random>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <KimeraRPGO/RobustSolver.h>
#include <KimeraRPGO/SolverParams.h>

using gtsam::symbol_shorthand::X;

/**
 * @brief 간단한 직선 궤적 생성
 */
void buildStraightTrajectory(
    gtsam::NonlinearFactorGraph& factors,
    gtsam::Values& values,
    int num_poses,
    double step_size,
    gtsam::SharedNoiseModel odom_noise) {

    // 첫 pose
    gtsam::Pose3 origin = gtsam::Pose3::Identity();
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());
    factors.addPrior(X(0), origin, prior_noise);
    values.insert(X(0), origin);

    // 직선 odometry
    gtsam::Pose3 odom(gtsam::Rot3::Identity(), gtsam::Point3(step_size, 0, 0));

    gtsam::Pose3 current = origin;
    for (int i = 0; i < num_poses - 1; ++i) {
        factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(i), X(i+1), odom, odom_noise);
        current = current * odom;
        values.insert(X(i+1), current);
    }
}

/**
 * @brief Outlier loop closure 생성 (완전히 잘못된 상대 pose)
 */
gtsam::Pose3 generateOutlierMeasurement() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> trans_dist(-10.0, 10.0);
    std::uniform_real_distribution<> rot_dist(-M_PI, M_PI);

    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(rot_dist(gen), rot_dist(gen), rot_dist(gen)),
        gtsam::Point3(trans_dist(gen), trans_dist(gen), trans_dist(gen))
    );
}

int main() {
    std::cout << "=== Kimera-RPGO Outlier Rejection Demo ===" << std::endl;

    // =========================================
    // 1. 파라미터 및 Solver 설정
    // =========================================
    std::cout << "\n1. Setting up solvers..." << std::endl;

    // PCM 기반 solver
    KimeraRPGO::RobustSolverParams pcm_params;
    pcm_params.setPcmSimple3DParams(0.5, 0.1, KimeraRPGO::Verbosity::QUIET);
    KimeraRPGO::RobustSolver pcm_solver(pcm_params);

    // GNC 기반 solver (비교용)
    KimeraRPGO::RobustSolverParams gnc_params;
    gnc_params.setGncInlierCostThresholdsAtProbability(0.99);
    gnc_params.verbosity = KimeraRPGO::Verbosity::QUIET;
    KimeraRPGO::RobustSolver gnc_solver(gnc_params);

    std::cout << "   PCM solver created." << std::endl;
    std::cout << "   GNC solver created." << std::endl;

    // =========================================
    // 2. 기본 궤적 생성
    // =========================================
    std::cout << "\n2. Building base trajectory..." << std::endl;

    int num_poses = 20;
    double step_size = 1.0;

    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01).finished());
    auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.02, 0.02, 0.02).finished());

    gtsam::NonlinearFactorGraph base_factors;
    gtsam::Values base_values;
    buildStraightTrajectory(base_factors, base_values, num_poses, step_size, odom_noise);

    std::cout << "   Created trajectory with " << num_poses << " poses." << std::endl;

    // =========================================
    // 3. 정상적인 Loop Closure 추가
    // =========================================
    std::cout << "\n3. Adding valid loop closures..." << std::endl;

    gtsam::NonlinearFactorGraph valid_loops;

    // 인접한 pose들 사이의 "loop closure" (실제로는 중복 확인)
    // X(5)에서 X(3)으로의 loop (실제 거리: 2m)
    gtsam::Pose3 valid_loop1(gtsam::Rot3::Identity(), gtsam::Point3(-2.0, 0, 0));
    valid_loops.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(5), X(3), valid_loop1, loop_noise);

    // X(10)에서 X(7)으로의 loop (실제 거리: 3m)
    gtsam::Pose3 valid_loop2(gtsam::Rot3::Identity(), gtsam::Point3(-3.0, 0, 0));
    valid_loops.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(10), X(7), valid_loop2, loop_noise);

    std::cout << "   Added 2 valid loop closures." << std::endl;

    // =========================================
    // 4. Outlier Loop Closure 추가
    // =========================================
    std::cout << "\n4. Adding OUTLIER loop closures..." << std::endl;

    gtsam::NonlinearFactorGraph outlier_loops;

    // 완전히 잘못된 loop closure들
    for (int i = 0; i < 3; ++i) {
        int from_idx = 15;
        int to_idx = 2 + i;  // X(15) -> X(2), X(3), X(4)

        gtsam::Pose3 outlier_measurement = generateOutlierMeasurement();
        outlier_loops.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(from_idx), X(to_idx), outlier_measurement, loop_noise);

        std::cout << "   Added outlier loop X(" << from_idx << ") -> X(" << to_idx << ")" << std::endl;
    }

    std::cout << "   Total outliers added: 3" << std::endl;

    // =========================================
    // 5. PCM Solver 테스트
    // =========================================
    std::cout << "\n5. Testing PCM solver..." << std::endl;

    // Base factors 추가
    pcm_solver.update(base_factors, base_values);

    // Valid loops 추가
    gtsam::Values empty_values;
    pcm_solver.update(valid_loops, empty_values);
    std::cout << "   Valid loops processed." << std::endl;

    // Outlier loops 추가
    pcm_solver.update(outlier_loops, empty_values);
    std::cout << "   Outlier loops processed." << std::endl;

    // 결과 확인
    gtsam::Values pcm_result = pcm_solver.calculateEstimate();

    // Outlier로 판정된 factor 수 확인
    std::cout << "\n   PCM Results:" << std::endl;
    std::cout << "   - Poses optimized: " << pcm_result.size() << std::endl;

    // =========================================
    // 6. GNC Solver 테스트
    // =========================================
    std::cout << "\n6. Testing GNC solver..." << std::endl;

    gnc_solver.update(base_factors, base_values);
    gnc_solver.update(valid_loops, empty_values);
    gnc_solver.update(outlier_loops, empty_values);

    gtsam::Values gnc_result = gnc_solver.calculateEstimate();

    std::cout << "\n   GNC Results:" << std::endl;
    std::cout << "   - Poses optimized: " << gnc_result.size() << std::endl;

    // =========================================
    // 7. 결과 비교
    // =========================================
    std::cout << "\n7. Comparing results..." << std::endl;

    // 마지막 pose 비교 (X(19))
    gtsam::Pose3 expected_final = gtsam::Pose3(
        gtsam::Rot3::Identity(),
        gtsam::Point3((num_poses - 1) * step_size, 0, 0)
    );

    gtsam::Pose3 pcm_final = pcm_result.at<gtsam::Pose3>(X(num_poses - 1));
    gtsam::Pose3 gnc_final = gnc_result.at<gtsam::Pose3>(X(num_poses - 1));

    double pcm_error = (pcm_final.translation() - expected_final.translation()).norm();
    double gnc_error = (gnc_final.translation() - expected_final.translation()).norm();

    std::cout << "\n   Expected final position: ["
              << expected_final.translation().x() << ", "
              << expected_final.translation().y() << ", "
              << expected_final.translation().z() << "]" << std::endl;

    std::cout << "\n   PCM final position: ["
              << pcm_final.translation().x() << ", "
              << pcm_final.translation().y() << ", "
              << pcm_final.translation().z() << "]" << std::endl;
    std::cout << "   PCM position error: " << pcm_error << " m" << std::endl;

    std::cout << "\n   GNC final position: ["
              << gnc_final.translation().x() << ", "
              << gnc_final.translation().y() << ", "
              << gnc_final.translation().z() << "]" << std::endl;
    std::cout << "   GNC position error: " << gnc_error << " m" << std::endl;

    // =========================================
    // 8. 결론
    // =========================================
    std::cout << "\n8. Conclusions:" << std::endl;

    if (pcm_error < 0.5 && gnc_error < 0.5) {
        std::cout << "   SUCCESS: Both solvers rejected outliers effectively!" << std::endl;
    } else if (pcm_error < 0.5) {
        std::cout << "   PCM performed better in this case." << std::endl;
    } else if (gnc_error < 0.5) {
        std::cout << "   GNC performed better in this case." << std::endl;
    } else {
        std::cout << "   WARNING: Both solvers were affected by outliers." << std::endl;
        std::cout << "   Consider adjusting thresholds." << std::endl;
    }

    std::cout << "\n=== Outlier Rejection Demo Complete ===" << std::endl;

    return 0;
}
