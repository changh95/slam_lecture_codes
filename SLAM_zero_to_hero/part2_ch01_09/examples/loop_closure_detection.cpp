/**
 * Loop Closure Detection Example using DBoW2
 *
 * This example demonstrates how to:
 * 1. Create or load a visual vocabulary
 * 2. Build an image database with inverted index
 * 3. Query for loop closure candidates
 * 4. Perform geometric verification
 *
 * This is the core technique used in visual SLAM systems like ORB-SLAM
 * for detecting when the robot revisits a previously seen location.
 *
 * The sequence is always read from a directory of real images. By default it
 * uses the sample frames bundled in this chapter's data/ folder; pass a
 * different directory with --data <dir>.
 */

#include "DBoW2/DBoW2.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// DBoW2 types for ORB descriptors
using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor,
                                                  DBoW2::FORB>;
using OrbDatabase = DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor,
                                              DBoW2::FORB>;

/**
 * Convert cv::Mat descriptors to vector of single-row cv::Mat
 */
std::vector<cv::Mat> toDescriptorVector(const cv::Mat& descriptors) {
    std::vector<cv::Mat> desc_vec;
    desc_vec.reserve(descriptors.rows);
    for (int i = 0; i < descriptors.rows; ++i) {
        desc_vec.push_back(descriptors.row(i));
    }
    return desc_vec;
}

/**
 * Structure to hold keyframe data
 */
struct Keyframe {
    int id;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Mat> descriptors_vec;  // DBoW2 format
    DBoW2::BowVector bow_vector;
    DBoW2::FeatureVector feature_vector;
};

/**
 * Match features between two keyframes using FeatureVector
 * This is much faster than brute-force matching
 */
int matchFeatures(const Keyframe& kf1, const Keyframe& kf2,
                  std::vector<cv::DMatch>& matches) {
    matches.clear();

    // Use FeatureVector for efficient matching
    // Features that share the same vocabulary node are likely to match
    auto it1 = kf1.feature_vector.begin();
    auto it2 = kf2.feature_vector.begin();

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    while (it1 != kf1.feature_vector.end() && it2 != kf2.feature_vector.end()) {
        if (it1->first == it2->first) {
            // Same node - compare features
            const auto& indices1 = it1->second;
            const auto& indices2 = it2->second;

            for (unsigned int idx1 : indices1) {
                int best_idx2 = -1;
                int best_dist = 256;  // Max Hamming distance for ORB

                for (unsigned int idx2 : indices2) {
                    int dist = cv::norm(kf1.descriptors.row(idx1),
                                        kf2.descriptors.row(idx2),
                                        cv::NORM_HAMMING);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx2 = idx2;
                    }
                }

                // Accept match if distance is below threshold
                if (best_dist < 50 && best_idx2 >= 0) {
                    matches.push_back(cv::DMatch(idx1, best_idx2, best_dist));
                }
            }
            ++it1;
            ++it2;
        } else if (it1->first < it2->first) {
            ++it1;
        } else {
            ++it2;
        }
    }

    return matches.size();
}

/**
 * Geometric verification using fundamental matrix estimation
 * Returns true if enough inliers are found. When provided, `inlier_matches`
 * is populated with the RANSAC-inlier subset of `matches`.
 */
bool geometricVerification(const Keyframe& kf1, const Keyframe& kf2,
                           const std::vector<cv::DMatch>& matches,
                           int min_inliers = 12,
                           std::vector<cv::DMatch>* inlier_matches = nullptr) {
    if (inlier_matches) inlier_matches->clear();
    if (matches.size() < 8) return false;

    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(kf1.keypoints[match.queryIdx].pt);
        pts2.push_back(kf2.keypoints[match.trainIdx].pt);
    }

    // Estimate fundamental matrix with RANSAC
    std::vector<uchar> inlier_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                        3.0, 0.99, inlier_mask);

    if (F.empty()) return false;

    // Count inliers
    int num_inliers = cv::countNonZero(inlier_mask);
    if (inlier_matches) {
        for (size_t i = 0; i < matches.size(); ++i) {
            if (inlier_mask[i]) inlier_matches->push_back(matches[i]);
        }
    }

    return num_inliers >= min_inliers;
}

/**
 * Render an N x N similarity matrix as a colored heatmap with grid + labels.
 */
cv::Mat renderSimilarityHeatmap(const std::vector<std::vector<double>>& sim,
                                int target_size = 720) {
    const int N = static_cast<int>(sim.size());
    if (N == 0) return cv::Mat();

    // Build 0..255 grayscale (clip at 1.0)
    cv::Mat raw(N, N, CV_8UC1);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            raw.at<uchar>(i, j) = cv::saturate_cast<uchar>(
                std::min(1.0, sim[i][j]) * 255.0);

    int cell = std::max(20, target_size / N);
    cv::Mat upscaled;
    cv::resize(raw, upscaled, cv::Size(N * cell, N * cell), 0, 0,
               cv::INTER_NEAREST);
    cv::Mat heat;
    cv::applyColorMap(upscaled, heat, cv::COLORMAP_JET);

    // Grid lines + tick labels in a margin
    const int margin = 40;
    cv::Mat canvas(heat.rows + margin, heat.cols + margin, CV_8UC3,
                   cv::Scalar(30, 30, 30));
    heat.copyTo(canvas(cv::Rect(margin, 0, heat.cols, heat.rows)));

    for (int i = 0; i <= N; ++i) {
        cv::line(canvas, cv::Point(margin, i * cell),
                 cv::Point(margin + N * cell, i * cell),
                 cv::Scalar(60, 60, 60), 1);
        cv::line(canvas, cv::Point(margin + i * cell, 0),
                 cv::Point(margin + i * cell, N * cell),
                 cv::Scalar(60, 60, 60), 1);
    }
    for (int i = 0; i < N; ++i) {
        cv::putText(canvas, std::to_string(i),
                    cv::Point(4, i * cell + cell / 2 + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(220, 220, 220), 1);
        cv::putText(canvas, std::to_string(i),
                    cv::Point(margin + i * cell + cell / 3,
                              N * cell + margin - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(220, 220, 220), 1);
    }
    return canvas;
}

int main(int argc, char* argv[]) {
    std::cout << "=== DBoW2 Loop Closure Detection ===" << std::endl;
    std::cout << std::endl;

    // CLI flags:
    //   --no-vis / --headless        disable OpenCV windows
    //   --data <dir>                 image directory to load
    //                                (default: bundled part2_ch01_09/data)
    //   --stride <N>                 take every Nth image (default 1)
    //   --max <N>                    cap loaded images at N (default unlimited)
    //   --min-inliers <N>            RANSAC inliers required for a LOOP (def 80)
    //   --score-threshold <X>        min BoW score for a candidate (def 0.1)
    //   --temporal-gap <N>           min keyframe distance for a candidate
    bool enable_vis = true;
    std::string data_dir;
    int stride = 1;
    int max_frames = 0;  // 0 = unlimited
    int min_inliers = 80;
    double score_threshold = 0.1;
    int temporal_gap = 10;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-vis" || arg == "--headless") {
            enable_vis = false;
        } else if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--stride" && i + 1 < argc) {
            stride = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--max" && i + 1 < argc) {
            max_frames = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--min-inliers" && i + 1 < argc) {
            min_inliers = std::max(8, std::atoi(argv[++i]));
        } else if (arg == "--score-threshold" && i + 1 < argc) {
            score_threshold = std::atof(argv[++i]);
        } else if (arg == "--temporal-gap" && i + 1 < argc) {
            temporal_gap = std::max(1, std::atoi(argv[++i]));
        }
    }

    // Parameters
    const int k = 9;       // Branching factor
    const int L = 3;       // Depth levels (smaller for faster demo)
    const int direct_index_level = 3;  // Level for FeatureVector

    std::cout << "Parameters:" << std::endl;
    std::cout << "  Vocabulary: k=" << k << ", L=" << L << std::endl;
    std::cout << "  Direct index level: " << direct_index_level << std::endl;
    std::cout << "  Score threshold:    " << score_threshold << std::endl;
    std::cout << "  Temporal gap:       " << temporal_gap << " frames"
              << std::endl;
    std::cout << "  Min RANSAC inliers: " << min_inliers << std::endl;
    std::cout << std::endl;

    // Create ORB detector
    auto orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2,
                                cv::ORB::HARRIS_SCORE, 31, 20);

    // Resolve the image directory. A --data override wins; otherwise fall back
    // to the sample frames shipped in this chapter's data/ folder. The binary
    // normally runs from build/, so "../data" resolves to part2_ch01_09/data.
    if (data_dir.empty()) {
        for (const char* candidate : {"../data", "data", "./data"}) {
            if (std::filesystem::is_directory(candidate)) {
                data_dir = candidate;
                break;
            }
        }
        if (data_dir.empty()) data_dir = "../data";
    }

    if (!std::filesystem::is_directory(data_dir)) {
        std::cerr << "Error: image directory not found: " << data_dir << "\n"
                  << "       Pass one with --data <dir> (expected the bundled "
                  << "part2_ch01_09/data folder)." << std::endl;
        return 1;
    }

    // Load the real image sequence from disk.
    std::cout << "Loading images from: " << data_dir << "  (stride=" << stride;
    if (max_frames > 0) std::cout << ", max=" << max_frames;
    std::cout << ")" << std::endl;

    std::vector<std::filesystem::path> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        const auto& p = entry.path();
        const std::string ext = p.extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
            ext == ".bmp" || ext == ".PNG" || ext == ".JPG") {
            image_paths.push_back(p);
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    std::vector<cv::Mat> image_sequence;
    for (size_t i = 0; i < image_paths.size(); i += stride) {
        cv::Mat img = cv::imread(image_paths[i].string(), cv::IMREAD_GRAYSCALE);
        if (!img.empty()) image_sequence.push_back(img);
        if (max_frames > 0 &&
            static_cast<int>(image_sequence.size()) >= max_frames)
            break;
    }
    std::cout << "  Loaded " << image_sequence.size() << " frames from disk"
              << std::endl;

    if (image_sequence.empty()) {
        std::cerr << "Error: no images loaded from " << data_dir
                  << " (expected .png/.jpg/.jpeg/.bmp files)." << std::endl;
        return 1;
    }
    std::cout << "  Sequence length: " << image_sequence.size() << " frames"
              << std::endl;
    std::cout << std::endl;

    // Extract features from all images
    std::cout << "Extracting features..." << std::endl;
    std::vector<Keyframe> keyframes;
    std::vector<std::vector<cv::Mat>> all_features;

    for (size_t i = 0; i < image_sequence.size(); ++i) {
        Keyframe kf;
        kf.id = i;
        kf.image = image_sequence[i];
        orb->detectAndCompute(kf.image, cv::noArray(),
                              kf.keypoints, kf.descriptors);

        if (!kf.descriptors.empty()) {
            kf.descriptors_vec = toDescriptorVector(kf.descriptors);
            all_features.push_back(kf.descriptors_vec);
            keyframes.push_back(kf);
        }
    }
    std::cout << "  Processed " << keyframes.size() << " keyframes" << std::endl;
    std::cout << std::endl;

    // Create vocabulary from the sequence
    std::cout << "Creating vocabulary from sequence..." << std::endl;
    auto start_voc = std::chrono::high_resolution_clock::now();

    OrbVocabulary vocabulary(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM);
    vocabulary.create(all_features);

    auto end_voc = std::chrono::high_resolution_clock::now();
    auto voc_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_voc - start_voc).count();

    std::cout << "  Vocabulary size: " << vocabulary.size() << " words" << std::endl;
    std::cout << "  Creation time: " << voc_time << " ms" << std::endl;
    std::cout << std::endl;

    // Transform all keyframes to BoW representation
    std::cout << "Computing BoW vectors for all keyframes..." << std::endl;
    for (auto& kf : keyframes) {
        vocabulary.transform(kf.descriptors_vec, kf.bow_vector,
                             kf.feature_vector, direct_index_level);
    }
    std::cout << std::endl;

    // Create database with direct index
    std::cout << "Creating image database..." << std::endl;
    OrbDatabase database(vocabulary, true, direct_index_level);

    const std::string loop_window = "DBoW2 Loop Closure (press any key, ESC quits)";
    if (enable_vis) {
        cv::namedWindow(loop_window, cv::WINDOW_NORMAL);
        cv::resizeWindow(loop_window, 1280, 480);
    }

    // Simulate online SLAM: process each keyframe sequentially
    std::cout << std::endl;
    std::cout << "=== Simulating Online Loop Closure Detection ===" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(6) << "Frame"
              << std::setw(12) << "DB Size"
              << std::setw(10) << "Best ID"
              << std::setw(12) << "Score"
              << std::setw(12) << "Matches"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    int num_loops_detected = 0;

    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto& current_kf = keyframes[i];

        std::cout << std::setw(6) << i;
        std::cout << std::setw(12) << database.size();

        // Query database (only if we have enough frames)
        if (database.size() > 0) {
            DBoW2::QueryResults results;
            database.query(current_kf.bow_vector, results, 5);

            // Find best candidate respecting temporal gap
            int best_candidate = -1;
            double best_score = 0;

            for (const auto& result : results) {
                // Skip recent frames (temporal consistency)
                if (static_cast<int>(i) - static_cast<int>(result.Id) > temporal_gap) {
                    if (result.Score > best_score && result.Score > score_threshold) {
                        best_score = result.Score;
                        best_candidate = result.Id;
                    }
                }
            }

            if (best_candidate >= 0) {
                std::cout << std::setw(10) << best_candidate;
                std::cout << std::setw(12) << std::fixed << std::setprecision(4)
                          << best_score;

                // Perform geometric verification
                std::vector<cv::DMatch> matches;
                int num_matches = matchFeatures(keyframes[best_candidate],
                                                current_kf, matches);
                std::cout << std::setw(12) << num_matches;

                std::vector<cv::DMatch> inlier_matches;
                bool verified = geometricVerification(
                    keyframes[best_candidate], current_kf, matches,
                    min_inliers, &inlier_matches);
                if (verified) {
                    std::cout << std::setw(10) << "LOOP!";
                    num_loops_detected++;
                } else {
                    std::cout << std::setw(10) << "rejected";
                }

                if (enable_vis) {
                    // Draw all matches in light gray, inliers in green/red
                    cv::Mat matches_vis;
                    cv::Scalar match_color = verified
                        ? cv::Scalar(0, 255, 0)    // green for LOOP
                        : cv::Scalar(0, 0, 255);   // red for REJECTED
                    cv::drawMatches(
                        keyframes[best_candidate].image,
                        keyframes[best_candidate].keypoints,
                        current_kf.image, current_kf.keypoints,
                        inlier_matches, matches_vis, match_color,
                        cv::Scalar(180, 180, 180), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                    const int banner_h = 60;
                    cv::Mat vis(matches_vis.rows + banner_h,
                                matches_vis.cols, matches_vis.type());
                    cv::Scalar banner_bg = verified
                        ? cv::Scalar(0, 180, 0)    // green
                        : cv::Scalar(0, 0, 180);   // red
                    cv::Scalar border = verified
                        ? cv::Scalar(0, 255, 255)  // cyan
                        : cv::Scalar(0, 0, 255);   // red
                    const char* header = verified
                        ? "LOOP FOUND!"
                        : "REJECTED (inliers below threshold)";

                    vis(cv::Rect(0, 0, vis.cols, banner_h)).setTo(banner_bg);
                    matches_vis.copyTo(
                        vis(cv::Rect(0, banner_h, matches_vis.cols,
                                     matches_vis.rows)));
                    cv::putText(vis, header, cv::Point(20, 44),
                                cv::FONT_HERSHEY_SIMPLEX, 1.1,
                                cv::Scalar(255, 255, 255), 3);

                    std::ostringstream caption;
                    caption << "Keyframe " << best_candidate
                            << "  <->  Current " << i
                            << "    score=" << std::fixed
                            << std::setprecision(3) << best_score
                            << "    inliers=" << inlier_matches.size()
                            << "/" << matches.size()
                            << "    threshold=" << min_inliers;
                    cv::putText(vis, caption.str(),
                                cv::Point(20, banner_h + 25),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                cv::Scalar(0, 255, 255), 2);

                    cv::rectangle(vis,
                                  cv::Point(0, 0),
                                  cv::Point(vis.cols - 1, vis.rows - 1),
                                  border, 6);

                    cv::imshow(loop_window, vis);
                    int key = cv::waitKey(0) & 0xff;
                    if (key == 27) enable_vis = false;
                }
            } else {
                std::cout << std::setw(10) << "-"
                          << std::setw(12) << "-"
                          << std::setw(12) << "-"
                          << std::setw(10) << "new place";
            }
        } else {
            std::cout << std::setw(10) << "-"
                      << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(10) << "init";
        }

        std::cout << std::endl;

        // Add current frame to database
        database.add(current_kf.bow_vector);
    }

    std::cout << std::endl;
    std::cout << "=== Loop Closure Detection Summary ===" << std::endl;
    std::cout << "  Total keyframes:     " << keyframes.size() << std::endl;
    std::cout << "  Loop closures found: " << num_loops_detected << std::endl;
    std::cout << std::endl;

    // Compute full pairwise similarity matrix once, reuse for print + heatmap
    const int N = static_cast<int>(keyframes.size());
    std::vector<std::vector<double>> sim(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sim[i][j] = vocabulary.score(keyframes[i].bow_vector,
                                         keyframes[j].bow_vector);
        }
    }

    // Print similarity matrix (first 10 frames for readability)
    std::cout << "=== Pairwise Similarity Matrix (first 10 frames) ===" << std::endl;
    std::cout << std::endl;

    int display_size = std::min(10, N);
    std::cout << "     ";
    for (int j = 0; j < display_size; ++j) {
        std::cout << std::setw(6) << j;
    }
    std::cout << std::endl;

    for (int i = 0; i < display_size; ++i) {
        std::cout << std::setw(4) << i << " ";
        for (int j = 0; j < display_size; ++j) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                      << sim[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Note: High scores on diagonal indicate self-similarity." << std::endl;
    std::cout << "Look for high off-diagonal scores = potential loop closures." << std::endl;
    std::cout << std::endl;

    if (enable_vis) {
        cv::Mat heatmap = renderSimilarityHeatmap(sim, 720);
        const std::string heat_window =
            "Pairwise Similarity (JET colormap, press any key)";
        cv::namedWindow(heat_window, cv::WINDOW_NORMAL);
        cv::imshow(heat_window, heatmap);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "=== Loop Closure Detection Complete ===" << std::endl;

    return 0;
}
