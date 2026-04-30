/**
 * @file feature_matching.cpp
 * @brief Demonstrates feature matching techniques using OpenCV
 *
 * This example shows:
 * 1. Brute-Force matching with distance threshold
 * 2. Brute-Force matching with Lowe's ratio test
 * 3. FLANN-based matching for ORB (LSH) and SIFT (KD-Tree)
 * 4. Cross-check matching
 *
 * Relevant for SLAM: Feature matching establishes correspondences between
 * frames for motion estimation and loop closure.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

/**
 * @brief Measure execution time
 */
template <typename Func>
double measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Filter matches using distance threshold
 *
 * Simple approach: reject matches above a threshold.
 * For Hamming distance (ORB), typical threshold: 30-50
 * For L2 distance (SIFT), typical threshold: 100-200
 */
std::vector<cv::DMatch> filterByDistance(const std::vector<cv::DMatch>& matches,
                                          float threshold) {
    std::vector<cv::DMatch> good_matches;
    for (const auto& m : matches) {
        if (m.distance < threshold) {
            good_matches.push_back(m);
        }
    }
    return good_matches;
}

/**
 * @brief Apply Lowe's ratio test
 *
 * Compares the distance of best match to second-best match.
 * If best match is significantly better, it's likely correct.
 * Typical ratio: 0.7-0.8
 */
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches,
                                        float ratio = 0.75f) {
    std::vector<cv::DMatch> good_matches;
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2) {
            if (match_pair[0].distance < ratio * match_pair[1].distance) {
                good_matches.push_back(match_pair[0]);
            }
        }
    }
    return good_matches;
}

/**
 * @brief Print matching statistics
 */
void printMatchStats(const std::string& method, size_t total_matches, size_t good_matches,
                     double time_ms) {
    float ratio = total_matches > 0 ? (100.0f * good_matches / total_matches) : 0;
    std::cout << std::left << std::setw(25) << method << " | "
              << std::right << std::setw(5) << good_matches << " / " << std::setw(5) << total_matches
              << " matches (" << std::fixed << std::setprecision(1) << ratio << "%) | "
              << std::setprecision(2) << time_ms << " ms" << std::endl;
}

/**
 * @brief Demo: ORB feature matching
 *
 * ORB uses binary descriptors, matched with Hamming distance.
 */
void demoORBMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n===== ORB Feature Matching =====" << std::endl;
    std::cout << "(Binary descriptor, Hamming distance)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Detect ORB features
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    orb->detectAndCompute(img1, cv::Mat(), kp1, desc1);
    orb->detectAndCompute(img2, cv::Mat(), kp2, desc2);

    std::cout << "Detected: " << kp1.size() << " and " << kp2.size() << " keypoints" << std::endl;

    if (desc1.empty() || desc2.empty()) {
        std::cout << "Error: No descriptors computed" << std::endl;
        return;
    }

    // ===== Method 1: BFMatcher with distance threshold =====
    std::vector<cv::DMatch> bf_matches;
    double time_bf = measureTime([&]() {
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING);
        bf->match(desc1, desc2, bf_matches);
    });
    auto good_bf = filterByDistance(bf_matches, 50.0f);
    printMatchStats("BF (threshold=50)", bf_matches.size(), good_bf.size(), time_bf);

    // ===== Method 2: BFMatcher with ratio test =====
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_ratio;
    double time_ratio = measureTime([&]() {
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING);
        bf->knnMatch(desc1, desc2, knn_matches, 2);
        good_ratio = applyRatioTest(knn_matches, 0.75f);
    });
    printMatchStats("BF + Ratio Test (0.75)", knn_matches.size(), good_ratio.size(), time_ratio);

    // ===== Method 3: BFMatcher with cross-check =====
    std::vector<cv::DMatch> crosscheck_matches;
    double time_cross = measureTime([&]() {
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);  // crossCheck=true
        bf->match(desc1, desc2, crosscheck_matches);
    });
    auto good_cross = filterByDistance(crosscheck_matches, 50.0f);
    printMatchStats("BF + CrossCheck", crosscheck_matches.size(), good_cross.size(), time_cross);

    // ===== Method 4: FLANN with LSH (for binary descriptors) =====
    std::vector<std::vector<cv::DMatch>> flann_knn;
    std::vector<cv::DMatch> good_flann;
    double time_flann = measureTime([&]() {
        // LSH parameters for binary descriptors
        // table_number: number of hash tables (12-20)
        // key_size: hash key size in bits (10-20)
        // multi_probe_level: number of probes (0-2)
        cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        flann.knnMatch(desc1, desc2, flann_knn, 2);
        good_flann = applyRatioTest(flann_knn, 0.75f);
    });
    printMatchStats("FLANN-LSH + Ratio", flann_knn.size(), good_flann.size(), time_flann);

    // Visualize best method (ratio test usually best)
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, good_ratio, img_matches,
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::putText(img_matches, "ORB + BF + Ratio Test", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::imshow("ORB Matching", img_matches);
    cv::imwrite("orb_matching_result.png", img_matches);
}

/**
 * @brief Demo: SIFT feature matching
 *
 * SIFT uses float descriptors, matched with L2 (Euclidean) distance.
 */
void demoSIFTMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n===== SIFT Feature Matching =====" << std::endl;
    std::cout << "(Float descriptor, L2 distance)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Detect SIFT features
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(500);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    sift->detectAndCompute(img1, cv::Mat(), kp1, desc1);
    sift->detectAndCompute(img2, cv::Mat(), kp2, desc2);

    std::cout << "Detected: " << kp1.size() << " and " << kp2.size() << " keypoints" << std::endl;

    if (desc1.empty() || desc2.empty()) {
        std::cout << "Error: No descriptors computed" << std::endl;
        return;
    }

    // ===== Method 1: BFMatcher with distance threshold =====
    std::vector<cv::DMatch> bf_matches;
    double time_bf = measureTime([&]() {
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_L2);
        bf->match(desc1, desc2, bf_matches);
    });
    auto good_bf = filterByDistance(bf_matches, 200.0f);
    printMatchStats("BF (threshold=200)", bf_matches.size(), good_bf.size(), time_bf);

    // ===== Method 2: BFMatcher with ratio test =====
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_ratio;
    double time_ratio = measureTime([&]() {
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_L2);
        bf->knnMatch(desc1, desc2, knn_matches, 2);
        good_ratio = applyRatioTest(knn_matches, 0.75f);
    });
    printMatchStats("BF + Ratio Test (0.75)", knn_matches.size(), good_ratio.size(), time_ratio);

    // ===== Method 3: FLANN with KD-Tree (for float descriptors) =====
    std::vector<std::vector<cv::DMatch>> flann_knn;
    std::vector<cv::DMatch> good_flann;
    double time_flann = measureTime([&]() {
        // KD-Tree parameters for float descriptors
        // trees: number of parallel kd-trees (5-10 typical)
        cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::KDTreeIndexParams>(5),
                                     cv::makePtr<cv::flann::SearchParams>(50));
        flann.knnMatch(desc1, desc2, flann_knn, 2);
        good_flann = applyRatioTest(flann_knn, 0.75f);
    });
    printMatchStats("FLANN-KDTree + Ratio", flann_knn.size(), good_flann.size(), time_flann);

    // Visualize
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, good_ratio, img_matches,
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::putText(img_matches, "SIFT + BF + Ratio Test", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    cv::imshow("SIFT Matching", img_matches);
    cv::imwrite("sift_matching_result.png", img_matches);
}

/**
 * @brief Demo: Matching with stricter ratio test (loop closure scenario)
 *
 * Uses a tighter ratio threshold for more reliable matches,
 * as would be needed for loop closure detection.
 */
void demoLoopClosureMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n===== Loop Closure Scenario =====" << std::endl;
    std::cout << "(Stricter matching for reliability)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // Use SIFT with more features for loop closure
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(1000);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    sift->detectAndCompute(img1, cv::Mat(), kp1, desc1);
    sift->detectAndCompute(img2, cv::Mat(), kp2, desc2);

    std::cout << "Detected: " << kp1.size() << " and " << kp2.size() << " keypoints" << std::endl;

    if (desc1.empty() || desc2.empty()) {
        std::cout << "Error: No descriptors computed" << std::endl;
        return;
    }

    // Match with stricter ratio test (0.7 instead of 0.75)
    cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf->knnMatch(desc1, desc2, knn_matches, 2);
    auto good_matches = applyRatioTest(knn_matches, 0.7f);

    printMatchStats("SIFT + Ratio (0.7)", knn_matches.size(), good_matches.size(), 0);

    // Visualize
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches,
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::putText(img_matches, "Loop Closure Matching (Strict Ratio)", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    cv::imshow("Loop Closure Matching", img_matches);
    cv::imwrite("loop_closure_matching_result.png", img_matches);
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Feature Matching Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << std::endl;

    cv::Mat img1, img2;

    // Default to data folder images
    std::string img1_path = (argc >= 3) ? argv[1] : "../data/1.jpg";
    std::string img2_path = (argc >= 3) ? argv[2] : "../data/2.jpg";

    img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images from " << img1_path << " and " << img2_path << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image1] [image2]" << std::endl;
        return 1;
    }
    std::cout << "Loaded: " << img1_path << " and " << img2_path << std::endl;

    std::cout << "Image size: " << img1.cols << "x" << img1.rows << std::endl;

    // Run demos
    demoORBMatching(img1, img2);
    demoSIFTMatching(img1, img2);
    demoLoopClosureMatching(img1, img2);

    // Summary
    std::cout << "\n===== Summary =====" << std::endl;
    std::cout << "For SLAM applications:" << std::endl;
    std::cout << "- Frame-to-frame tracking: ORB + BF + Ratio Test (fast)" << std::endl;
    std::cout << "- Loop closure detection: SIFT + FLANN + Ratio Test (accurate)" << std::endl;
    std::cout << "- Real-time systems: ORB with cross-check or ratio test" << std::endl;
    std::cout << "- Always apply geometric verification (RANSAC) after matching!" << std::endl;
    std::cout << std::endl;

    std::cout << "Results saved to:" << std::endl;
    std::cout << "  - orb_matching_result.png" << std::endl;
    std::cout << "  - sift_matching_result.png" << std::endl;
    std::cout << "  - loop_closure_matching_result.png" << std::endl;
    std::cout << std::endl;
    std::cout << "Press any key to exit..." << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
