/**
 * @file run_vo_kitti.cpp
 * @brief Run monocular VO on KITTI dataset with ground truth scale
 *
 * This example demonstrates the complete monocular VO pipeline on
 * KITTI Visual Odometry dataset, using ground truth poses for scale.
 *
 * Usage:
 *   ./run_vo_kitti <image_directory> <poses_file> [max_frames]
 *
 * Example:
 *   ./run_vo_kitti /data/kitti/sequences/00/image_0 /data/kitti/poses/00.txt 500
 *
 * KITTI camera parameters for sequence 00-02:
 *   - Focal length: 718.856
 *   - Principal point: (607.1928, 185.2157)
 */

#include "monocular_vo.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Monocular Visual Odometry - KITTI Demo" << std::endl;
    std::cout << "============================================" << std::endl;

    if (argc < 3) {
        std::cout << "\nUsage: " << argv[0]
                  << " <image_directory> <poses_file> [max_frames]" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0]
                  << " /data/kitti/sequences/00/image_0 /data/kitti/poses/00.txt"
                  << std::endl;
        std::cout << "\nKITTI dataset structure:" << std::endl;
        std::cout << "  sequences/00/image_0/  - Left grayscale images" << std::endl;
        std::cout << "  sequences/00/image_1/  - Right grayscale images" << std::endl;
        std::cout << "  poses/00.txt           - Ground truth poses" << std::endl;
        std::cout << "\nDownload from: https://www.cvlibs.net/datasets/kitti/eval_odometry.php"
                  << std::endl;
        return 1;
    }

    std::string image_dir = argv[1];
    std::string poses_file = argv[2];
    int max_frames = (argc > 3) ? std::stoi(argv[3]) : -1;

    // KITTI camera parameters (sequence 00-02)
    double focal = 718.856;
    cv::Point2d pp(607.1928, 185.2157);

    std::cout << "\nCamera parameters:" << std::endl;
    std::cout << "  Focal length: " << focal << std::endl;
    std::cout << "  Principal point: (" << pp.x << ", " << pp.y << ")" << std::endl;

    // Load images
    std::cout << "\nLoading images from: " << image_dir << std::endl;
    std::vector<std::string> image_paths = vo_utils::loadImageSequence(image_dir);

    if (image_paths.empty()) {
        std::cerr << "Error: No images found in " << image_dir << std::endl;
        return 1;
    }

    if (max_frames > 0 && image_paths.size() > static_cast<size_t>(max_frames)) {
        image_paths.resize(max_frames);
    }
    std::cout << "Processing " << image_paths.size() << " images" << std::endl;

    // Initialize VO with KITTI ground truth scale
    std::cout << "Loading ground truth from: " << poses_file << std::endl;
    MonocularVO_KITTI vo(focal, pp, poses_file);
    vo.setVisualization(true);

    // Get ground truth for comparison
    const auto& ground_truth = vo.getGroundTruth();
    std::cout << "Ground truth poses: " << ground_truth.size() << std::endl;

    // Create windows
    cv::namedWindow("Features", cv::WINDOW_NORMAL);
    cv::namedWindow("Trajectory", cv::WINDOW_NORMAL);
    cv::resizeWindow("Features", 1200, 400);
    cv::resizeWindow("Trajectory", 800, 800);

    // Statistics
    double total_time = 0;
    int successful_frames = 0;

    // Main processing loop
    std::cout << "\nProcessing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < image_paths.size(); i++) {
        cv::Mat frame = cv::imread(image_paths[i]);
        if (frame.empty()) {
            std::cerr << "Warning: Failed to load " << image_paths[i] << std::endl;
            continue;
        }

        // Process frame
        auto t1 = std::chrono::high_resolution_clock::now();
        bool success = vo.processFrame(frame);
        auto t2 = std::chrono::high_resolution_clock::now();

        double frame_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_time += frame_time;

        if (success) successful_frames++;

        // Visualize
        cv::Mat vis_features = vo.getVisualizationFrame();
        cv::Mat vis_trajectory = vo_utils::drawTrajectory(
            vo.getTrajectory(), ground_truth, -1  // Auto scale
        );

        // Add timing info
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1)
           << "Time: " << frame_time << " ms | FPS: " << (1000.0 / frame_time);
        cv::putText(vis_features, ss.str(), cv::Point(10, vis_features.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

        // Add ATE info to trajectory
        double ate = vo.computeATE();
        if (ate >= 0) {
            ss.str("");
            ss << std::fixed << std::setprecision(2) << "ATE: " << ate << " m";
            cv::putText(vis_trajectory, ss.str(), cv::Point(10, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Features", vis_features);
        cv::imshow("Trajectory", vis_trajectory);

        // Print progress
        if (i % 100 == 0 || i == image_paths.size() - 1) {
            cv::Point3d pos = vo.getPositionPoint();
            std::cout << "Frame " << i << "/" << image_paths.size()
                      << " | Features: " << vo.getFeatureCount()
                      << " | Position: (" << std::fixed << std::setprecision(1)
                      << pos.x << ", " << pos.z << ")"
                      << " | ATE: " << ate << " m" << std::endl;
        }

        // Handle key press
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            std::cout << "Interrupted by user" << std::endl;
            break;
        }
        if (key == ' ') {
            std::cout << "Paused. Press any key to continue..." << std::endl;
            cv::waitKey(0);
        }
        if (key == 's') {
            // Save current trajectory
            cv::imwrite("trajectory_current.png", vis_trajectory);
            std::cout << "Saved trajectory_current.png" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Final statistics
    std::cout << "\n============================================" << std::endl;
    std::cout << "              Final Results" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Processed frames: " << vo.getFrameId() << std::endl;
    std::cout << "Successful motion estimates: " << successful_frames << std::endl;
    std::cout << "Total processing time: " << elapsed << " s" << std::endl;
    std::cout << "Average frame time: " << (total_time / vo.getFrameId()) << " ms" << std::endl;
    std::cout << "Average FPS: " << (vo.getFrameId() / elapsed) << std::endl;

    cv::Point3d final_pos = vo.getPositionPoint();
    std::cout << "\nFinal position: ("
              << final_pos.x << ", " << final_pos.y << ", " << final_pos.z << ")" << std::endl;

    double final_ate = vo.computeATE();
    if (final_ate >= 0) {
        std::cout << "Absolute Trajectory Error (RMSE): " << final_ate << " m" << std::endl;
    }

    // Save final results
    cv::Mat final_trajectory = vo_utils::drawTrajectory(vo.getTrajectory(), ground_truth, -1);

    // Add final stats to image
    std::stringstream ss;
    ss << "Frames: " << vo.getFrameId() << " | ATE: "
       << std::fixed << std::setprecision(2) << final_ate << " m";
    cv::putText(final_trajectory, ss.str(), cv::Point(10, final_trajectory.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    cv::imwrite("trajectory_final.png", final_trajectory);
    std::cout << "\nTrajectory saved to trajectory_final.png" << std::endl;

    // Wait for key press before closing
    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
