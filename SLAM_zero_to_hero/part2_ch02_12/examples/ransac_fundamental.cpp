/**
 * RANSAC Fundamental Matrix Estimation with USAC
 *
 * Run nine F-estimation methods on real ORB correspondences from a EuRoC MAV
 * frame pair (data/1403636579763555584.png and data/1403636584763555584.png by
 * default -- MH_01_easy cam0, 5 s apart).
 * No synthetic data: there is no closed-form GT F for arbitrary scenes, so
 * each method is judged on (a) Sampson error against the matches, and
 * (b) inlier count. The Sampson column is scored by shared code (meanSampson)
 * for every method, so it is comparable throughout. The INLIER column is not:
 * each method reported its own mask, and the classic and USAC paths do not use
 * the same inlier rule.
 *
 * Inlier-rule footgun: the classic path (FM_RANSAC, FM_LMEDS, FM_7POINT) scores
 * with the max-form symmetric epipolar distance, max(d^2/A, d^2/B) =
 * d^2/min(A,B) (fundam.cpp, FMEstimatorCallback::computeError), while every
 * USAC_* flag scores with Sampson, d^2/(A+B) (usac/estimator.cpp,
 * SampsonErrorImpl) -- with d = x2^T F x1, A = |(F^T x2)_xy|^2,
 * B = |(F x1)_xy|^2. Since min(A,B) <= (A+B)/2, the classic residual is always
 * at least twice Sampson's, so at one nominal 3 px threshold the classic rule
 * is strictly the tighter one. That is most of why FM_RANSAC reports ~739
 * inliers below while every Sampson-scored method lands at ~824-828: the split
 * tracks the rule, not the estimator's quality. ransac_custom prints the same
 * models under both rules if you want the like-for-like numbers.
 *
 * Methods (USAC flag semantics read off OpenCV 4.12
 * modules/calib3d/src/usac/ransac_solvers.cpp, setParameters()):
 *   1. FM_7POINT     (NOT the 7-point algorithm here -- see the footgun below)
 *   2. FM_8POINT     (8-point, single shot, no outlier rejection)
 *   3. FM_RANSAC     (RANSAC over the 7-point minimal solver)
 *   4. FM_LMEDS      (Least Median of Squares over the 7-point solver)
 *   5. USAC_DEFAULT  (Uniform + MSAC + Inner&Iter-LO)
 *   6. USAC_FM_8PTS  (Uniform + MSAC + Inner-LO, 8-point estimator)
 *   7. USAC_MAGSAC   (Uniform + MAGSAC scoring + sigma-consensus LO)
 *   8. Custom USAC   (PROSAC + MAGSAC + Inner&Iter-LO + MAGSAC polisher)
 *   9. USAC_ACCURATE (Uniform + MSAC + Graph-Cut LO)
 *
 * Note that only USAC_MAGSAC scores with MAGSAC; every other USAC flag here
 * uses MSAC. What actually differs between them is the sampler and the local
 * optimization, not the scoring.
 *
 * FM_7POINT footgun: cv::findFundamentalMat runs the 7-point algorithm only
 * when handed exactly 7 correspondences. With more (844 here) fundam.cpp
 * takes its else branch, and because (FM_7POINT & ~3) != FM_RANSAC it lands
 * in createLMeDSPointSetRegistrator -- LMedS, not 7-point. That is why
 * FM_7POINT and FM_LMEDS below report bit-identical models and masks, and why
 * both are ~40x slower than FM_RANSAC. With exactly 7 points the call instead
 * returns a 9-row Mat: the 3 roots of the 7-point cubic.
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
    cv::Mat F;
    cv::Mat mask;
    double sampson;
    int inliers;
    double time_ms;
};

// Epipolar geometry: for a subset of inliers, draw each point and the
// epipolar line of its correspondence in the other image (same color).
// Points must lie on their lines if F is correct.
static cv::Mat drawEpipolarVis(const cv::Mat& img1, const cv::Mat& img2,
                               const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               const cv::Mat& F, const cv::Mat& mask,
                               int maxLines = 25) {
    std::vector<cv::Point2f> in1, in2;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (mask.at<uchar>(i)) {
            in1.push_back(pts1[i]);
            in2.push_back(pts2[i]);
        }
    }
    const int step = std::max(1, static_cast<int>(in1.size()) / maxLines);
    std::vector<cv::Point2f> s1, s2;
    for (size_t i = 0; i < in1.size(); i += step) {
        s1.push_back(in1[i]);
        s2.push_back(in2[i]);
    }

    std::vector<cv::Vec3f> linesIn2, linesIn1;
    cv::computeCorrespondEpilines(s1, 1, F, linesIn2);
    cv::computeCorrespondEpilines(s2, 2, F, linesIn1);

    cv::Mat top, bottom;
    cv::cvtColor(img1, top, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, bottom, cv::COLOR_GRAY2BGR);
    cv::RNG rng(12345);
    auto drawLine = [](cv::Mat& img, const cv::Vec3f& l, const cv::Scalar& color) {
        if (std::abs(l[1]) < 1e-6f) return;  // near-vertical line, skip
        cv::Point2d p0(0.0, -l[2] / l[1]);
        cv::Point2d p1(img.cols, -(l[2] + l[0] * img.cols) / l[1]);
        cv::line(img, p0, p1, color, 1);
    };
    for (size_t i = 0; i < s1.size(); ++i) {
        cv::Scalar color(rng.uniform(64, 255), rng.uniform(64, 255), rng.uniform(64, 255));
        drawLine(top, linesIn1[i], color);
        drawLine(bottom, linesIn2[i], color);
        cv::circle(top, s1[i], 4, color, -1);
        cv::circle(bottom, s2[i], 4, color, -1);
    }
    cv::Mat vis;
    cv::vconcat(top, bottom, vis);
    return vis;
}

int main(int argc, char* argv[]) {
    std::cout << "=== RANSAC Fundamental Matrix Estimation (EuRoC MH_01_easy, 5 s apart) ===\n";
    std::cout << std::fixed << std::setprecision(6);

    cv::Mat img1, img2;
    std::vector<cv::Point2f> pts1, pts2;
    if (!loadRealPair(argc, argv, img1, img2, pts1, pts2, 8)) return 1;

    // EuRoC cam0 intrinsics from mav0/cam0/sensor.yaml (for reference only;
    // F is uncalibrated, so nothing below consumes K). Note these images are
    // the raw, still-distorted cam0 frames (radial-tangential model, k1 =
    // -0.2834), which is why the estimators are compared on pixel residuals
    // rather than on a calibrated E.
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        458.654, 0.0, 367.215,
        0.0, 457.296, 248.375,
        0.0, 0.0, 1.0);
    std::cout << "\nReference K (EuRoC cam0, raw/unrectified):\n" << K << "\n\n";

    const double threshold = 3.0;
    Timer timer;

    auto run_masked = [&](const std::string& name, auto fn) -> MethodResult {
        MethodResult r{name, {}, {}, -1.0, 0, 0.0};
        timer.start();
        r.F = fn(r.mask);
        r.time_ms = timer.elapsedMs();
        if (!r.F.empty() && r.F.rows == 3) {
            r.sampson = meanSampson(r.F, pts1, pts2, r.mask);
            r.inliers = r.mask.empty() ? 0 : cv::countNonZero(r.mask);
        }
        std::cout << "\n--- " << name << " ---\n";
        if (r.F.empty()) {
            std::cout << "  (no solution)  Time: " << r.time_ms << " ms\n";
        } else if (r.F.rows != 3) {
            std::cout << "  Multiple solutions (" << r.F.rows / 3 << "), Time: "
                      << r.time_ms << " ms\n";
        } else {
            std::cout << "  Sampson err (inliers): " << r.sampson
                      << ", Inliers: " << r.inliers << "/" << pts1.size()
                      << ", Time: " << r.time_ms << " ms\n";
        }
        return r;
    };

    std::vector<MethodResult> results;

    results.push_back(run_masked("FM_7POINT", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_7POINT, threshold, 0.99, m);
    }));
    results.push_back(run_masked("FM_8POINT", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_8POINT, threshold, 0.99, m);
    }));
    results.push_back(run_masked("FM_RANSAC", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, threshold, 0.99, m);
    }));
    results.push_back(run_masked("FM_LMEDS", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::FM_LMEDS, threshold, 0.99, m);
    }));
    results.push_back(run_masked("USAC_DEFAULT", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_DEFAULT, threshold, 0.99, m);
    }));
    results.push_back(run_masked("USAC_FM_8PTS", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_FM_8PTS, threshold, 0.99, m);
    }));
    results.push_back(run_masked("USAC_MAGSAC", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_MAGSAC, threshold, 0.99, m);
    }));

    cv::UsacParams usacParams;
    usacParams.sampler = cv::SAMPLING_PROSAC;
    usacParams.score = cv::SCORE_METHOD_MAGSAC;
    usacParams.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
    usacParams.loIterations = 15;
    usacParams.neighborsSearch = cv::NEIGH_FLANN_KNN;
    usacParams.final_polisher = cv::MAGSAC;
    usacParams.final_polisher_iterations = 10;
    usacParams.threshold = threshold;
    usacParams.confidence = 0.999;
    usacParams.maxIterations = 10000;
    // Deliberately serial. randomGeneratorState already defaults to 0, but
    // isParallel = true still makes USAC nondeterministic: thread scheduling
    // decides which hypotheses get scored first, so the inlier count drifts
    // run to run. results.csv is committed as a reproducible baseline, so this
    // row has to be stable. Flip it to true to see the effect.
    usacParams.isParallel = false;
    results.push_back(run_masked("Custom USAC", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, m, usacParams);
    }));

    results.push_back(run_masked("USAC_ACCURATE", [&](cv::Mat& m) {
        return cv::findFundamentalMat(pts1, pts2, cv::USAC_ACCURATE, threshold, 0.99, m);
    }));

    std::cout << "\n=== Summary ===\n";
    std::cout << std::left << std::setw(18) << "Method"
              << std::right << std::setw(14) << "Sampson"
              << std::setw(12) << "Inliers"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(56, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(18) << r.name
                  << std::right << std::setw(14) << r.sampson
                  << std::setw(12) << r.inliers
                  << std::setw(12) << r.time_ms << "\n";
    }

    // Sanity check: epipolar constraint on USAC_MAGSAC inliers.
    const auto& magsac = results[6];
    if (!magsac.F.empty() && magsac.F.rows == 3 && !magsac.mask.empty()) {
        double total = 0.0;
        int n = 0;
        for (size_t i = 0; i < pts1.size(); ++i) {
            if (magsac.mask.at<uchar>(i)) {
                cv::Mat p1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
                cv::Mat p2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);
                total += std::abs(p2.dot(magsac.F * p1));
                ++n;
            }
        }
        std::cout << "\nUSAC_MAGSAC mean |x2^T F x1| on inliers: "
                  << (n ? total / n : 0.0) << " (expected ~0)\n";
    }

    // Visualization: USAC_MAGSAC inlier/outlier matches + epipolar geometry.
    if (!magsac.F.empty() && magsac.F.rows == 3 && !magsac.mask.empty()) {
        cv::Mat visMatches = drawMatchesVis(img1, img2, pts1, pts2, magsac.mask);
        cv::Mat visEpi = drawEpipolarVis(img1, img2, pts1, pts2, magsac.F, magsac.mask);
        cv::imwrite("fundamental_usac_magsac_matches.jpg", visMatches);
        cv::imwrite("fundamental_epipolar_lines.jpg", visEpi);
        std::cout << "Saved: fundamental_usac_magsac_matches.jpg, "
                     "fundamental_epipolar_lines.jpg\n";
        if (std::getenv("DISPLAY") != nullptr) {
            cv::imshow("USAC_MAGSAC matches (green=inlier, red=outlier)", visMatches);
            cv::imshow("Epipolar lines (USAC_MAGSAC inliers)", visEpi);
            std::cout << "Press any key to exit..." << std::endl;
            cv::waitKey(0);
        }
    }
    return 0;
}
