/**
 * @file feature_profiling.cpp
 * @brief Feature detection/matching profiling with FAST+TEBLID, ORB, SIFT
 *
 * This example demonstrates:
 * 1. FAST keypoint detection + TEBLID descriptor
 * 2. Performance comparison: ORB vs SIFT vs FAST+TEBLID
 * 3. Matching quality comparison
 * 4. Profiling with easy_profiler (view results in profiler_gui)
 *
 * TEBLID: Boosted Binary Local Image Descriptor
 * - Binary descriptor like ORB (efficient matching with Hamming distance)
 * - Better accuracy than ORB, approaching SIFT-like performance
 * - Fast computation, suitable for real-time SLAM
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <easy/profiler.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

/**
 * @brief Apply Lowe's ratio test
 */
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>>& knn_matches,
                                        float ratio = 0.75f) {
    std::vector<cv::DMatch> good_matches;
    for (const auto& match_pair : knn_matches) {
        if (match_pair.size() >= 2 && match_pair[0].distance < ratio * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
        }
    }
    return good_matches;
}

/**
 * @brief ORB detection, description, and matching
 */
void profileORB(const cv::Mat& img1, const cv::Mat& img2, int nfeatures = 500) {
    EASY_FUNCTION(profiler::colors::Red);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    // Create detector
    cv::Ptr<cv::ORB> orb;
    {
        EASY_BLOCK("ORB_Create", profiler::colors::Red100);
        orb = cv::ORB::create(nfeatures);
    }

    // Detection
    {
        EASY_BLOCK("ORB_Detect", profiler::colors::Red200);
        orb->detect(img1, kp1);
        orb->detect(img2, kp2);
    }

    // Description
    {
        EASY_BLOCK("ORB_Compute", profiler::colors::Red300);
        orb->compute(img1, kp1, desc1);
        orb->compute(img2, kp2, desc2);
    }

    // Matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_matches;
    {
        EASY_BLOCK("ORB_Match", profiler::colors::Red400);
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING);
        bf->knnMatch(desc1, desc2, knn_matches, 2);
        good_matches = applyRatioTest(knn_matches, 0.75f);
    }

    std::cout << "ORB: " << kp1.size() << "/" << kp2.size() << " keypoints, "
              << good_matches.size() << "/" << knn_matches.size() << " good matches" << std::endl;

    // Save visualization
    {
        EASY_BLOCK("ORB_Visualize", profiler::colors::Red500);
        cv::Mat vis;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, vis,
                        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::putText(vis, "ORB (" + std::to_string(good_matches.size()) + " matches)",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::imwrite("orb_profiling.png", vis);
    }
}

/**
 * @brief SIFT detection, description, and matching
 */
void profileSIFT(const cv::Mat& img1, const cv::Mat& img2, int nfeatures = 500) {
    EASY_FUNCTION(profiler::colors::Blue);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    // Create detector
    cv::Ptr<cv::SIFT> sift;
    {
        EASY_BLOCK("SIFT_Create", profiler::colors::Blue100);
        sift = cv::SIFT::create(nfeatures);
    }

    // Detection
    {
        EASY_BLOCK("SIFT_Detect", profiler::colors::Blue200);
        sift->detect(img1, kp1);
        sift->detect(img2, kp2);
    }

    // Description
    {
        EASY_BLOCK("SIFT_Compute", profiler::colors::Blue300);
        sift->compute(img1, kp1, desc1);
        sift->compute(img2, kp2, desc2);
    }

    // Matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_matches;
    {
        EASY_BLOCK("SIFT_Match", profiler::colors::Blue400);
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_L2);
        bf->knnMatch(desc1, desc2, knn_matches, 2);
        good_matches = applyRatioTest(knn_matches, 0.75f);
    }

    std::cout << "SIFT: " << kp1.size() << "/" << kp2.size() << " keypoints, "
              << good_matches.size() << "/" << knn_matches.size() << " good matches" << std::endl;

    // Save visualization
    {
        EASY_BLOCK("SIFT_Visualize", profiler::colors::Blue500);
        cv::Mat vis;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, vis,
                        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::putText(vis, "SIFT (" + std::to_string(good_matches.size()) + " matches)",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::imwrite("sift_profiling.png", vis);
    }
}

/**
 * @brief FAST + TEBLID detection, description, and matching
 *
 * TEBLID (Boosted Binary Local Image Descriptor):
 * - Uses machine learning to select optimal binary tests
 * - Better repeatability than ORB's BRIEF-based descriptor
 * - Two scales available: TEBLID_256 (256 bits) and TEBLID_512 (512 bits)
 */
void profileFAST_TEBLID(const cv::Mat& img1, const cv::Mat& img2, int nfeatures = 500) {
    EASY_FUNCTION(profiler::colors::Green);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;

    // Create detectors
    cv::Ptr<cv::FastFeatureDetector> fast;
    cv::Ptr<cv::xfeatures2d::TEBLID> teblid;
    {
        EASY_BLOCK("TEBLID_Create", profiler::colors::Green100);
        fast = cv::FastFeatureDetector::create(20, true, cv::FastFeatureDetector::TYPE_9_16);
        teblid = cv::xfeatures2d::TEBLID::create(6.25f, cv::xfeatures2d::TEBLID::SIZE_256_BITS);
    }

    // Detection
    {
        EASY_BLOCK("FAST_Detect", profiler::colors::Green200);
        fast->detect(img1, kp1);
        fast->detect(img2, kp2);
    }

    // Keep top N keypoints by response and set size for TEBLID
    {
        EASY_BLOCK("TEBLID_PrepareKeypoints", profiler::colors::Green250);
        auto sortAndLimit = [nfeatures](std::vector<cv::KeyPoint>& kps) {
            if (static_cast<int>(kps.size()) > nfeatures) {
                std::sort(kps.begin(), kps.end(),
                    [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                        return a.response > b.response;
                    });
                kps.resize(nfeatures);
            }
            // TEBLID requires keypoint size to be set
            for (auto& kp : kps) kp.size = 31.0f;
        };
        sortAndLimit(kp1);
        sortAndLimit(kp2);
    }

    // Description
    {
        EASY_BLOCK("TEBLID_Compute", profiler::colors::Green300);
        teblid->compute(img1, kp1, desc1);
        teblid->compute(img2, kp2, desc2);
    }

    // Matching
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> good_matches;
    {
        EASY_BLOCK("TEBLID_Match", profiler::colors::Green400);
        cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING);
        bf->knnMatch(desc1, desc2, knn_matches, 2);
        good_matches = applyRatioTest(knn_matches, 0.75f);
    }

    std::cout << "FAST+TEBLID: " << kp1.size() << "/" << kp2.size() << " keypoints, "
              << good_matches.size() << "/" << knn_matches.size() << " good matches" << std::endl;

    // Save visualization
    {
        EASY_BLOCK("TEBLID_Visualize", profiler::colors::Green500);
        cv::Mat vis;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, vis,
                        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::putText(vis, "FAST+TEBLID (" + std::to_string(good_matches.size()) + " matches)",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::imwrite("fast_teblid_profiling.png", vis);
    }
}

int main(int argc, char** argv) {
    // Initialize profiler
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    std::cout << "========================================" << std::endl;
    std::cout << "Feature Detection Profiling" << std::endl;
    std::cout << "(using easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << std::endl;

    cv::Mat img1, img2;

    // Default to data folder images
    std::string img1_path = (argc >= 3) ? argv[1] : "../data/1.jpg";
    std::string img2_path = (argc >= 3) ? argv[2] : "../data/2.jpg";

    {
        EASY_BLOCK("LoadImages", profiler::colors::Cyan);
        img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
        img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
    }

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images from " << img1_path << " and " << img2_path << std::endl;
        std::cerr << "Usage: " << argv[0] << " [image1] [image2]" << std::endl;
        return 1;
    }
    std::cout << "Loaded: " << img1_path << " and " << img2_path << std::endl;

    std::cout << "Image size: " << img1.cols << "x" << img1.rows << std::endl;
    std::cout << std::endl;

    // Profile each method
    std::cout << "Running profiling...\n" << std::endl;

    {
        EASY_BLOCK("AllMethods", profiler::colors::White);

        profileORB(img1, img2, 500);
        profileSIFT(img1, img2, 500);
        profileFAST_TEBLID(img1, img2, 500);
    }

    // Save profiling data
    std::string profile_path = "feature_profiling.prof";
    {
        EASY_BLOCK("SaveProfile", profiler::colors::Magenta);
        auto blocks_count = profiler::dumpBlocksToFile(profile_path.c_str());
        std::cout << "\n========================================" << std::endl;
        std::cout << "Profile saved to: " << profile_path << std::endl;
        std::cout << "Total blocks recorded: " << blocks_count << std::endl;
    }

    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  - feature_profiling.prof (open with profiler_gui)" << std::endl;
    std::cout << "  - orb_profiling.png" << std::endl;
    std::cout << "  - sift_profiling.png" << std::endl;
    std::cout << "  - fast_teblid_profiling.png" << std::endl;
    std::cout << std::endl;
    std::cout << "To visualize profiling data:" << std::endl;
    std::cout << "  profiler_gui feature_profiling.prof" << std::endl;
    std::cout << std::endl;

    return 0;
}
