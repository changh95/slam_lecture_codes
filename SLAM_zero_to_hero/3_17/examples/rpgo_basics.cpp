/**
 * @file rpgo_basics.cpp
 * @brief Kimera-RPGO 기본 사용법 예제
 *
 * 이 예제에서는:
 * 1. RobustSolver 초기화
 * 2. Prior 및 Odometry factor 추가
 * 3. Loop closure factor 추가
 * 4. Robust 최적화 수행
 */

#include <iostream>
#include <vector>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <KimeraRPGO/RobustSolver.h>
#include <KimeraRPGO/SolverParams.h>

using gtsam::symbol_shorthand::X;  // Pose keys

int main() {
    std::cout << "=== Kimera-RPGO Basics ===" << std::endl;

    // =========================================
    // 1. RobustSolver 파라미터 설정
    // =========================================
    std::cout << "\n1. Setting up RobustSolver parameters..." << std::endl;

    KimeraRPGO::RobustSolverParams params;

    // PCM 파라미터 (3D)
    // - translation_threshold: 위치 일관성 임계값 (미터)
    // - rotation_threshold: 회전 일관성 임계값 (라디안)
    double translation_threshold = 0.5;  // 50cm
    double rotation_threshold = 0.1;     // ~6도

    params.setPcmSimple3DParams(
        translation_threshold,
        rotation_threshold,
        KimeraRPGO::Verbosity::UPDATE  // 업데이트 시 출력
    );

    // Solver 생성
    KimeraRPGO::RobustSolver solver(params);

    std::cout << "   Translation threshold: " << translation_threshold << " m" << std::endl;
    std::cout << "   Rotation threshold: " << rotation_threshold << " rad" << std::endl;

    // =========================================
    // 2. Noise model 정의
    // =========================================
    std::cout << "\n2. Defining noise models..." << std::endl;

    // Prior noise (매우 작은 불확실성)
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());

    // Odometry noise
    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01).finished());

    // Loop closure noise
    auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.02, 0.02, 0.02).finished());

    // =========================================
    // 3. 초기 Pose Graph 구성
    // =========================================
    std::cout << "\n3. Building initial pose graph..." << std::endl;

    gtsam::NonlinearFactorGraph factors;
    gtsam::Values values;

    // 첫 번째 pose (원점에 prior)
    gtsam::Pose3 origin = gtsam::Pose3::Identity();
    factors.addPrior(X(0), origin, prior_noise);
    values.insert(X(0), origin);

    // Odometry로 trajectory 생성 (사각형 경로)
    std::vector<gtsam::Pose3> odometry_measurements = {
        gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1.0, 0.0, 0.0)),  // 앞으로
        gtsam::Pose3(gtsam::Rot3::Rz(M_PI/2), gtsam::Point3(0.0, 0.0, 0.0)),  // 왼쪽 회전
        gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1.0, 0.0, 0.0)),  // 앞으로
        gtsam::Pose3(gtsam::Rot3::Rz(M_PI/2), gtsam::Point3(0.0, 0.0, 0.0)),  // 왼쪽 회전
        gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1.0, 0.0, 0.0)),  // 앞으로
        gtsam::Pose3(gtsam::Rot3::Rz(M_PI/2), gtsam::Point3(0.0, 0.0, 0.0)),  // 왼쪽 회전
        gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1.0, 0.0, 0.0)),  // 앞으로 (원점으로)
    };

    // Odometry factor 추가
    gtsam::Pose3 current_pose = origin;
    for (size_t i = 0; i < odometry_measurements.size(); ++i) {
        factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
            X(i), X(i+1), odometry_measurements[i], odom_noise);

        current_pose = current_pose * odometry_measurements[i];
        values.insert(X(i+1), current_pose);

        std::cout << "   Added odometry factor X(" << i << ") -> X(" << i+1 << ")" << std::endl;
    }

    // =========================================
    // 4. Solver에 factor 추가
    // =========================================
    std::cout << "\n4. Adding factors to RobustSolver..." << std::endl;

    solver.update(factors, values);

    std::cout << "   Initial factors added." << std::endl;

    // =========================================
    // 5. Loop closure 추가
    // =========================================
    std::cout << "\n5. Adding loop closure factor..." << std::endl;

    // 정상적인 loop closure: X(7)이 X(0) 근처로 돌아옴
    gtsam::Pose3 loop_measurement = gtsam::Pose3(
        gtsam::Rot3::Rz(M_PI/2),  // 마지막 회전
        gtsam::Point3(0.0, 0.0, 0.0)
    );

    gtsam::NonlinearFactorGraph loop_factors;
    loop_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        X(7), X(0), loop_measurement, loop_noise);

    gtsam::Values empty_values;
    solver.update(loop_factors, empty_values);

    std::cout << "   Loop closure X(7) -> X(0) added." << std::endl;

    // =========================================
    // 6. 최적화 결과 확인
    // =========================================
    std::cout << "\n6. Getting optimization results..." << std::endl;

    gtsam::Values result = solver.calculateEstimate();

    std::cout << "\n   Optimized poses:" << std::endl;
    for (size_t i = 0; i <= 7; ++i) {
        gtsam::Pose3 pose = result.at<gtsam::Pose3>(X(i));
        gtsam::Point3 t = pose.translation();
        std::cout << "   X(" << i << "): t = [" << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;
    }

    // =========================================
    // 7. Graph 통계
    // =========================================
    std::cout << "\n7. Graph statistics:" << std::endl;
    std::cout << "   Number of poses: 8" << std::endl;
    std::cout << "   Number of odometry factors: " << odometry_measurements.size() << std::endl;
    std::cout << "   Number of loop closures: 1" << std::endl;

    std::cout << "\n=== RPGO Basics Complete ===" << std::endl;

    return 0;
}
