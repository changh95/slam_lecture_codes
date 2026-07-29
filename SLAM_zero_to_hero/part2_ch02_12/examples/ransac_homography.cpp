/**
 * RANSAC Homography Estimation
 *
 * Run all seven RANSAC / USAC variants on real ORB correspondences from a
 * EuRoC MAV frame pair (MH_01_easy cam0, 5 s apart). The machine-hall scene
 * is fully 3D with no dominant plane, so a homography is a mis-specified
 * model here: it can only explain the subset of matches consistent with a
 * single plane (~560-650 of 844) while parallax pushes the rest out -- a
 * high outlier ratio, which is exactly the regime that exercises RANSAC.
 * Contrast with ransac_fundamental, where the correct model keeps ~825.
 *
 * Pipeline (shared with the other demos via ransac_data.h):
 *   1. Load left/right images (data/000024.png, data/000025.png by default).
 *   2. ORB detect + BF-Hamming + Lowe ratio test (0.75) gives raw matches.
 *   3. Run RANSAC, LMEDS, four USAC flags, and a Custom UsacParams config.
 *   4. Per method, report mean inlier reprojection error, inlier count, time.
 *
 * No synthetic ground-truth H here: real images don't admit a single GT
 * homography. The error column is the *mean inlier reprojection error*
 * |H * x1 - x2| in pixels, which is the metric a robust estimator actually
 * minimizes.
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ransac_data.h"

struct MethodResult {
    std::string name;
    cv::Mat H;
    cv::Mat mask;
    double err;
    int inliers;
    double time_ms;
};

int main(int argc, char* argv[]) {
    std::cout << "=== RANSAC Homography Estimation (EuRoC MH_01_easy, 5 s apart) ===\n";
    std::cout << std::fixed << std::setprecision(4);

    cv::Mat img1, img2;
    std::vector<cv::Point2f> pts1, pts2;
    if (!loadRealPair(argc, argv, img1, img2, pts1, pts2, 4)) return 1;

    const double threshold = 3.0;
    Timer timer;

    auto run = [&](const std::string& name, auto fn) -> MethodResult {
        MethodResult r{name, {}, {}, -1.0, 0, 0.0};
        timer.start();
        r.H = fn(r.mask);
        r.time_ms = timer.elapsedMs();
        if (!r.H.empty()) {
            r.inliers = cv::countNonZero(r.mask);
            r.err = meanInlierReproj(pts1, pts2, r.H, r.mask);
        }
        std::cout << "\n--- " << name << " ---\n";
        if (r.H.empty()) {
            std::cout << "  (no solution)  Time: " << r.time_ms << " ms\n";
        } else {
            std::cout << "  Mean inlier reproj err: " << r.err
                      << " px, Inliers: " << r.inliers << "/" << pts1.size()
                      << ", Time: " << r.time_ms << " ms\n";
        }
        return r;
    };

    std::vector<MethodResult> results;

    results.push_back(run("RANSAC", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::RANSAC, threshold, m, 2000, 0.99);
    }));
    results.push_back(run("LMEDS", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::LMEDS, threshold, m);
    }));
    results.push_back(run("USAC_DEFAULT", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_DEFAULT, threshold, m);
    }));
    results.push_back(run("USAC_MAGSAC", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, threshold, m);
    }));
    results.push_back(run("USAC_PROSAC", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_PROSAC, threshold, m);
    }));
    results.push_back(run("USAC_ACCURATE", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, cv::USAC_ACCURATE, threshold, m);
    }));

    cv::UsacParams usacParams;
    usacParams.sampler = cv::SAMPLING_PROSAC;
    usacParams.score = cv::SCORE_METHOD_MAGSAC;
    usacParams.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
    usacParams.loIterations = 15;
    usacParams.final_polisher = cv::MAGSAC;
    usacParams.final_polisher_iterations = 10;
    usacParams.threshold = threshold;
    usacParams.confidence = 0.999;
    usacParams.maxIterations = 5000;
    // Deliberately serial -- see the note in ransac_fundamental.cpp: with
    // isParallel = true this row ranged over 644-651 inliers across runs,
    // which makes the committed results.csv unreproducible. Serial is
    // bit-identical run to run.
    usacParams.isParallel = false;
    results.push_back(run("Custom UsacParams", [&](cv::Mat& m) {
        return cv::findHomography(pts1, pts2, m, usacParams);
    }));

    std::cout << "\n=== Summary ===\n";
    std::cout << std::left << std::setw(20) << "Method"
              << std::right << std::setw(12) << "Reproj(px)"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::setw(12) << r.err
                  << std::setw(12) << r.inliers
                  << std::setw(12) << r.time_ms << "\n";
    }

    // Visualization: classic RANSAC vs USAC_MAGSAC inlier masks side by side.
    const auto& ransac = results[0];   // "RANSAC"
    const auto& magsac = results[3];   // "USAC_MAGSAC"
    if (!ransac.H.empty() && !magsac.H.empty()) {
        cv::Mat visRansac = drawMatchesVis(img1, img2, pts1, pts2, ransac.mask);
        cv::Mat visMagsac = drawMatchesVis(img1, img2, pts1, pts2, magsac.mask);
        cv::imwrite("homography_ransac_matches.jpg", visRansac);
        cv::imwrite("homography_usac_magsac_matches.jpg", visMagsac);
        std::cout << "\nSaved: homography_ransac_matches.jpg, "
                     "homography_usac_magsac_matches.jpg\n";
        if (std::getenv("DISPLAY") != nullptr) {
            cv::imshow("RANSAC matches (green=inlier, red=outlier)", visRansac);
            cv::imshow("USAC_MAGSAC matches (green=inlier, red=outlier)", visMagsac);
            std::cout << "Press any key to exit..." << std::endl;
            cv::waitKey(0);
        }
    }
    return 0;
}
