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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

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
 * @brief Show an image in a resizable, screen-fitted window.
 *
 * Default `cv::imshow` opens a fixed-size window sized to the image, which is
 * unmanageable for tall stacked pipeline figures (6 rows × image height).
 * This helper creates a WINDOW_NORMAL window and pre-scales it to fit within
 * a target screen budget while preserving aspect ratio — the user can still
 * drag the window edges to resize further.
 */
void showResizable(const std::string& name, const cv::Mat& img,
                   int max_w = 1400, int max_h = 850) {
    cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    double scale = std::min({1.0,
                             static_cast<double>(max_w) / img.cols,
                             static_cast<double>(max_h) / img.rows});
    cv::resizeWindow(name,
                     static_cast<int>(img.cols * scale),
                     static_cast<int>(img.rows * scale));
    cv::imshow(name, img);
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
 * @brief Geometric verification with RANSAC
 *
 * Filters out matches that are geometrically inconsistent with the
 * epipolar geometry between the two views. This is the standard outlier
 * rejection step in SLAM pipelines — the ratio test removes ambiguous
 * matches, but it cannot detect e.g. matches on repeated structures
 * (windows, bricks) that survive the descriptor-distance test.
 *
 * Uses the fundamental matrix (works for any rigid 3D scene with parallax).
 * For planar / pure-rotation scenes, cv::findHomography is an alternative.
 */
std::vector<cv::DMatch> geometricVerification(const std::vector<cv::KeyPoint>& kp1,
                                              const std::vector<cv::KeyPoint>& kp2,
                                              const std::vector<cv::DMatch>& matches,
                                              double ransac_thresh = 3.0) {
    if (matches.size() < 8) return matches;  // need >=8 for findFundamentalMat
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
    std::vector<uchar> inlier_mask;
    cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, ransac_thresh, 0.99, inlier_mask);
    std::vector<cv::DMatch> inliers;
    inliers.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inlier_mask[i]) inliers.push_back(matches[i]);
    }
    return inliers;
}

/**
 * @brief Adaptive Non-Maximal Suppression — Suppression via Square Covering.
 *
 * The "SSC" variant by Bailo et al. (2018), as used by Kimera-VIO's feature
 * tracker. Given N detected keypoints, returns ~numToKeep spatially well-
 * distributed keypoints (the keypoint with highest response in each
 * suppression cell).
 *
 * Why: a vanilla "top-K by response" selection clusters keypoints in high-
 * contrast regions (e.g. one busy corner of the image) and leaves the rest
 * of the frame empty, which hurts pose estimation. ANMS enforces spatial
 * coverage while still preferring strong corners.
 *
 * Reference: Bailo et al., "Efficient adaptive non-maximal suppression
 * algorithms for homogeneous spatial keypoint distribution", PRL 2018.
 * Public implementation: https://github.com/BAILOOL/ANMS-Codes
 */
std::vector<cv::KeyPoint> anmsSSC(std::vector<cv::KeyPoint> keypoints,
                                  int numToKeep, float tolerance,
                                  int cols, int rows) {
    if (numToKeep <= 0 || keypoints.empty()) return {};
    if (static_cast<int>(keypoints.size()) <= numToKeep) return keypoints;

    // Sort by descending response (strongest corner first).
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    // Closed-form bounds on the suppression radius (see Bailo et al. eq. 7).
    long long exp1 = static_cast<long long>(rows) + cols + 2LL * numToKeep;
    long long exp2 = 4LL * cols + 4LL * numToKeep + 4LL * rows * numToKeep
                   + static_cast<long long>(rows) * rows
                   + static_cast<long long>(cols) * cols
                   - 2LL * rows * cols
                   + 4LL * rows * cols * numToKeep;
    double exp3 = std::sqrt(static_cast<double>(exp2));
    double exp4 = static_cast<double>(numToKeep - 1);

    double sol1 = -std::round((exp1 + exp3) / exp4);
    double sol2 = -std::round((exp1 - exp3) / exp4);

    int high = static_cast<int>(std::max(sol1, sol2));
    int low  = static_cast<int>(std::floor(std::sqrt(
                  static_cast<double>(keypoints.size()) / numToKeep)));
    if (low < 1) low = 1;

    unsigned Kmin = static_cast<unsigned>(std::round(numToKeep * (1.0f - tolerance)));
    unsigned Kmax = static_cast<unsigned>(std::round(numToKeep * (1.0f + tolerance)));

    std::vector<int> best;
    int prevWidth = -1;

    while (true) {
        int width = low + (high - low) / 2;
        if (width == prevWidth || low > high) break;

        std::vector<int> result;
        result.reserve(keypoints.size());
        const double c = width / 2.0;
        const int numCellCols = static_cast<int>(std::floor(cols / c));
        const int numCellRows = static_cast<int>(std::floor(rows / c));
        std::vector<std::vector<bool>> covered(
            numCellRows + 1, std::vector<bool>(numCellCols + 1, false));

        for (size_t i = 0; i < keypoints.size(); ++i) {
            const int row = static_cast<int>(std::floor(keypoints[i].pt.y / c));
            const int col = static_cast<int>(std::floor(keypoints[i].pt.x / c));
            if (row < 0 || col < 0 || row > numCellRows || col > numCellCols) continue;
            if (covered[row][col]) continue;

            result.push_back(static_cast<int>(i));
            const int half = static_cast<int>(std::floor(width / c));
            const int rowMin = std::max(0, row - half);
            const int rowMax = std::min(numCellRows, row + half);
            const int colMin = std::max(0, col - half);
            const int colMax = std::min(numCellCols, col + half);
            for (int r = rowMin; r <= rowMax; ++r)
                for (int cc = colMin; cc <= colMax; ++cc)
                    covered[r][cc] = true;
        }

        if (result.size() >= Kmin && result.size() <= Kmax) {
            best = std::move(result);
            break;
        } else if (result.size() < Kmin) {
            high = width - 1;
        } else {
            low = width + 1;
            best = std::move(result);
        }
        prevWidth = width;
    }

    std::vector<cv::KeyPoint> out;
    out.reserve(best.size());
    for (int idx : best) out.push_back(keypoints[idx]);
    return out;
}

/**
 * @brief Plain "top-K by response" selector — baseline to compare against ANMS.
 */
std::vector<cv::KeyPoint> topKByResponse(std::vector<cv::KeyPoint> keypoints, int K) {
    if (static_cast<int>(keypoints.size()) <= K) return keypoints;
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });
    keypoints.resize(K);
    return keypoints;
}

/**
 * @brief Run a BF matching pipeline (raw NN -> ratio -> RANSAC) on a
 *        pre-selected keypoint set.
 */
struct PipelineResult {
    std::vector<cv::DMatch> raw;
    std::vector<cv::DMatch> ratio;
    std::vector<cv::DMatch> ransac;
};

PipelineResult runBfPipeline(const cv::Ptr<cv::Feature2D>& describer,
                             const cv::Mat& img1, std::vector<cv::KeyPoint>& kp1,
                             const cv::Mat& img2, std::vector<cv::KeyPoint>& kp2,
                             int norm_type, float ratio_thresh) {
    PipelineResult r;
    cv::Mat desc1, desc2;
    describer->compute(img1, kp1, desc1);
    describer->compute(img2, kp2, desc2);
    if (desc1.empty() || desc2.empty()) return r;

    cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(norm_type);
    std::vector<std::vector<cv::DMatch>> knn;
    bf->knnMatch(desc1, desc2, knn, 2);

    r.raw.reserve(knn.size());
    for (const auto& mp : knn) if (!mp.empty()) r.raw.push_back(mp[0]);
    r.ratio = applyRatioTest(knn, ratio_thresh);
    r.ransac = geometricVerification(kp1, kp2, r.ratio);
    return r;
}

/**
 * @brief Draw a single labeled "matches" row (one image-pair view).
 */
cv::Mat drawMatchRow(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
                     const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
                     const std::vector<cv::DMatch>& matches,
                     const std::string& label, cv::Scalar line_color) {
    cv::Mat out;
    cv::drawMatches(img1, kp1, img2, kp2, matches, out,
                    line_color, cv::Scalar(255, 0, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::putText(out, label + " (" + std::to_string(matches.size()) + " matches)",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
    return out;
}

/**
 * @brief Build a 3-row "matching pipeline" figure: raw -> ratio -> RANSAC.
 *
 * Each row draws the same image pair with a different match set, labeled with
 * stage name and count. This makes the filtering effect immediately visible:
 * dense noisy lines at the top, sparse and consistent at the bottom.
 */
cv::Mat makePipelineFigure(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
                           const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
                           const std::vector<cv::DMatch>& raw,
                           const std::vector<cv::DMatch>& filtered,
                           const std::vector<cv::DMatch>& ransac,
                           const std::string& method) {
    cv::Mat row_raw      = drawMatchRow(img1, kp1, img2, kp2, raw,
                                        method + " | 1. Raw NN",           cv::Scalar(0,   0, 255));
    cv::Mat row_filtered = drawMatchRow(img1, kp1, img2, kp2, filtered,
                                        method + " | 2. After Ratio Test", cv::Scalar(0, 255, 255));
    cv::Mat row_ransac   = drawMatchRow(img1, kp1, img2, kp2, ransac,
                                        method + " | 3. After RANSAC",     cv::Scalar(0, 255,   0));

    cv::Mat figure;
    cv::vconcat(std::vector<cv::Mat>{row_raw, row_filtered, row_ransac}, figure);
    return figure;
}

/**
 * @brief Per-detector demo output, used by main() to build the cross-method
 *        final-comparison figure.
 */
struct DemoResult {
    cv::Mat final_row;  // single drawMatchRow showing RANSAC-verified inliers
};

/**
 * @brief Demo: ORB feature matching
 *
 * ORB uses binary descriptors, matched with Hamming distance.
 */
DemoResult demoORBMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n===== ORB Feature Matching (ANMS-SSC) =====" << std::endl;
    std::cout << "(Binary descriptor, Hamming distance)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    constexpr int N_POOL = 5000;
    constexpr int K      = 1500;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(N_POOL);
    std::vector<cv::KeyPoint> kp1_all, kp2_all;
    orb->detect(img1, kp1_all);
    orb->detect(img2, kp2_all);

    // ANMS-SSC: well-distributed keypoints (vs clustering from top-K by response).
    auto kp1 = anmsSSC(kp1_all, K, 0.1f, img1.cols, img1.rows);
    auto kp2 = anmsSSC(kp2_all, K, 0.1f, img2.cols, img2.rows);
    std::cout << "Pool: " << kp1_all.size() << "/" << kp2_all.size()
              << "  |  ANMS kept: " << kp1.size() << "/" << kp2.size() << std::endl;

    // Looser ratio (0.8) — binary descriptors discriminate less; RANSAC cleans up.
    auto r = runBfPipeline(orb, img1, kp1, img2, kp2, cv::NORM_HAMMING, 0.8f);
    printMatchStats("Raw NN",      r.raw.size(),   r.raw.size(),    0);
    printMatchStats("Ratio (0.8)", r.raw.size(),   r.ratio.size(),  0);
    printMatchStats("RANSAC",      r.ratio.size(), r.ransac.size(), 0);

    cv::Mat figure = makePipelineFigure(img1, kp1, img2, kp2,
                                        r.raw, r.ratio, r.ransac, "ORB");
    showResizable("ORB Matching Pipeline", figure);
    cv::imwrite("orb_matching_result.png", figure);

    DemoResult out;
    out.final_row = drawMatchRow(img1, kp1, img2, kp2, r.ransac,
                                 "ORB | RANSAC Inliers", cv::Scalar(0, 255, 0));
    return out;
}

/**
 * @brief Demo: SIFT feature matching
 *
 * SIFT uses float descriptors, matched with L2 (Euclidean) distance.
 */
DemoResult demoSIFTMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n===== SIFT Feature Matching (ANMS-SSC) =====" << std::endl;
    std::cout << "(Float descriptor, L2 distance)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    constexpr int K = 800;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(0);  // 0 = no cap on detections
    std::vector<cv::KeyPoint> kp1_all, kp2_all;
    sift->detect(img1, kp1_all);
    sift->detect(img2, kp2_all);

    auto kp1 = anmsSSC(kp1_all, K, 0.1f, img1.cols, img1.rows);
    auto kp2 = anmsSSC(kp2_all, K, 0.1f, img2.cols, img2.rows);
    std::cout << "Pool: " << kp1_all.size() << "/" << kp2_all.size()
              << "  |  ANMS kept: " << kp1.size() << "/" << kp2.size() << std::endl;

    auto r = runBfPipeline(sift, img1, kp1, img2, kp2, cv::NORM_L2, 0.75f);
    printMatchStats("Raw NN",       r.raw.size(),   r.raw.size(),    0);
    printMatchStats("Ratio (0.75)", r.raw.size(),   r.ratio.size(),  0);
    printMatchStats("RANSAC",       r.ratio.size(), r.ransac.size(), 0);

    cv::Mat figure = makePipelineFigure(img1, kp1, img2, kp2,
                                        r.raw, r.ratio, r.ransac, "SIFT");
    showResizable("SIFT Matching Pipeline", figure);
    cv::imwrite("sift_matching_result.png", figure);

    DemoResult out;
    out.final_row = drawMatchRow(img1, kp1, img2, kp2, r.ransac,
                                 "SIFT | RANSAC Inliers", cv::Scalar(0, 255, 0));
    return out;
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

    // Geometric verification with RANSAC (essential for loop closure reliability)
    auto inliers = geometricVerification(kp1, kp2, good_matches);
    printMatchStats("SIFT + Ratio + RANSAC", good_matches.size(), inliers.size(), 0);

    // Visualize verified inliers
    cv::Mat img_matches;
    cv::drawMatches(img1, kp1, img2, kp2, inliers, img_matches,
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::putText(img_matches,
                "Loop Closure (Strict Ratio + RANSAC, " + std::to_string(inliers.size()) + " inliers)",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    showResizable("Loop Closure Matching", img_matches);
    cv::imwrite("loop_closure_matching_result.png", img_matches);
}

/**
 * @brief Demo: FAST + TEBLID feature matching
 *
 * Uses FAST detector with TEBLID (learned binary) descriptor.
 * Better discriminative power than ORB's BRIEF while keeping
 * Hamming-distance matching speed.
 */
DemoResult demoFASTTEBLIDMatching(const cv::Mat& img1, const cv::Mat& img2) {
    std::cout << "\n===== FAST + TEBLID Feature Matching (ANMS-SSC) =====" << std::endl;
    std::cout << "(Learned binary descriptor, Hamming distance)" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    constexpr int K = 800;
    auto fast   = cv::FastFeatureDetector::create(10, true, cv::FastFeatureDetector::TYPE_9_16);
    // scale_factor=5.0f is the calibrated value for FAST/AGAST/AKAZE/BRISK.
    auto teblid = cv::xfeatures2d::TEBLID::create(5.0f, cv::xfeatures2d::TEBLID::SIZE_256_BITS);

    std::vector<cv::KeyPoint> kp1_all, kp2_all;
    fast->detect(img1, kp1_all);
    fast->detect(img2, kp2_all);

    auto kp1 = anmsSSC(kp1_all, K, 0.1f, img1.cols, img1.rows);
    auto kp2 = anmsSSC(kp2_all, K, 0.1f, img2.cols, img2.rows);
    std::cout << "Pool: " << kp1_all.size() << "/" << kp2_all.size()
              << "  |  ANMS kept: " << kp1.size() << "/" << kp2.size() << std::endl;

    // TEBLID requires keypoint size to be set (FAST keypoints carry no scale).
    for (auto& kp : kp1) kp.size = 31.0f;
    for (auto& kp : kp2) kp.size = 31.0f;

    auto r = runBfPipeline(teblid, img1, kp1, img2, kp2, cv::NORM_HAMMING, 0.75f);
    printMatchStats("Raw NN",       r.raw.size(),   r.raw.size(),    0);
    printMatchStats("Ratio (0.75)", r.raw.size(),   r.ratio.size(),  0);
    printMatchStats("RANSAC",       r.ratio.size(), r.ransac.size(), 0);

    cv::Mat figure = makePipelineFigure(img1, kp1, img2, kp2,
                                        r.raw, r.ratio, r.ransac, "FAST+TEBLID");
    showResizable("FAST+TEBLID Matching Pipeline", figure);
    cv::imwrite("fast_teblid_matching_result.png", figure);

    DemoResult out;
    out.final_row = drawMatchRow(img1, kp1, img2, kp2, r.ransac,
                                 "FAST+TEBLID | RANSAC Inliers", cv::Scalar(0, 255, 0));
    return out;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Feature Matching Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << std::endl;

    cv::Mat img1, img2;

    // Default to data folder images
    std::string img1_path = (argc >= 3) ? argv[1] : "../data/1.png";
    std::string img2_path = (argc >= 3) ? argv[2] : "../data/2.png";

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
    auto orb_out    = demoORBMatching(img1, img2);
    auto sift_out   = demoSIFTMatching(img1, img2);
    auto teblid_out = demoFASTTEBLIDMatching(img1, img2);
    demoLoopClosureMatching(img1, img2);

    // Cross-method comparison: stack the final RANSAC-verified-inlier rows
    // from each detector so the user can compare match density and spatial
    // coverage at a glance.
    cv::Mat comparison;
    cv::vconcat(std::vector<cv::Mat>{orb_out.final_row,
                                     sift_out.final_row,
                                     teblid_out.final_row}, comparison);
    showResizable("Final RANSAC Inliers: ORB vs SIFT vs FAST+TEBLID", comparison);
    cv::imwrite("final_comparison.png", comparison);

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
    std::cout << "  - fast_teblid_matching_result.png" << std::endl;
    std::cout << "  - loop_closure_matching_result.png" << std::endl;
    std::cout << "  - final_comparison.png  (ORB vs SIFT vs FAST+TEBLID)" << std::endl;
    std::cout << std::endl;
    std::cout << "Press any key to exit..." << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
