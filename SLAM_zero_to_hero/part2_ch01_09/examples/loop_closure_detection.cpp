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
 */

#include "DBoW2/DBoW2.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
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
 * Generate a sequence of images simulating robot motion with loop closures
 *
 * Creates a scenario where the robot:
 * 1. Moves through different areas (unique images)
 * 2. Returns to previously visited areas (loop closures)
 */
std::vector<cv::Mat> generateSequenceWithLoops(int num_unique_places,
                                                int width = 640,
                                                int height = 480) {
    std::vector<cv::Mat> sequence;
    std::vector<cv::Mat> unique_places;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate unique place images
    for (int place = 0; place < num_unique_places; ++place) {
        cv::Mat img(height, width, CV_8UC1);

        // Create distinct pattern for each place
        double freq_x = 0.02 + place * 0.005;
        double freq_y = 0.015 + place * 0.003;
        double phase = place * 0.5;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double val = 128.0 +
                    60.0 * std::sin(x * freq_x + phase) +
                    60.0 * std::cos(y * freq_y + phase);
                img.at<uchar>(y, x) = cv::saturate_cast<uchar>(val);
            }
        }

        // Add distinctive features (shapes) for each place
        std::uniform_int_distribution<> pos_dist_x(50, width - 50);
        std::uniform_int_distribution<> pos_dist_y(50, height - 50);
        std::uniform_int_distribution<> size_dist(15, 50);

        int num_features = 20 + place * 3;
        for (int f = 0; f < num_features; ++f) {
            int x = pos_dist_x(gen);
            int y = pos_dist_y(gen);
            int size = size_dist(gen);

            // Use place ID to create consistent patterns within same place
            int shape_type = (place + f) % 3;
            uchar color = static_cast<uchar>((place * 37 + f * 13) % 256);

            switch (shape_type) {
                case 0:
                    cv::circle(img, cv::Point(x, y), size/2, cv::Scalar(color), -1);
                    break;
                case 1:
                    cv::rectangle(img, cv::Point(x-size/2, y-size/2),
                                  cv::Point(x+size/2, y+size/2),
                                  cv::Scalar(color), -1);
                    break;
                case 2:
                    cv::ellipse(img, cv::Point(x, y), cv::Size(size, size/2),
                                45.0 * place, 0, 360, cv::Scalar(color), -1);
                    break;
            }
        }

        // Add slight noise
        cv::Mat noise(height, width, CV_8UC1);
        cv::randn(noise, 0, 5);
        img += noise;

        cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
        unique_places.push_back(img);
    }

    // Create sequence with intentional loop closures
    // Trajectory: 0->1->2->3->4->5->6->7->8->9 -> 3->4 -> 10->11 -> 0->1
    //             ^^^^^^^^^^^^^^^               ^^^^^    ^^^^^     ^^^^^
    //             new places                    loop     new      loop

    // First pass through places 0-9
    for (int i = 0; i < std::min(10, num_unique_places); ++i) {
        // Add with slight viewpoint change simulation
        cv::Mat view = unique_places[i].clone();
        sequence.push_back(view);
    }

    // Loop closure: revisit places 3 and 4
    if (num_unique_places > 4) {
        for (int i = 3; i <= 4; ++i) {
            cv::Mat view = unique_places[i].clone();
            // Add slight perspective variation
            std::normal_distribution<> noise_dist(0, 3);
            cv::Mat noise(height, width, CV_8UC1);
            cv::randn(noise, 0, 3);
            view += noise;
            sequence.push_back(view);
        }
    }

    // New places
    for (int i = 10; i < std::min(12, num_unique_places); ++i) {
        sequence.push_back(unique_places[i].clone());
    }

    // Loop closure: revisit places 0 and 1
    for (int i = 0; i <= 1; ++i) {
        cv::Mat view = unique_places[i].clone();
        cv::Mat noise(height, width, CV_8UC1);
        cv::randn(noise, 0, 3);
        view += noise;
        sequence.push_back(view);
    }

    return sequence;
}

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
 * Returns true if enough inliers are found
 */
bool geometricVerification(const Keyframe& kf1, const Keyframe& kf2,
                           const std::vector<cv::DMatch>& matches,
                           int min_inliers = 12) {
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

    return num_inliers >= min_inliers;
}

int main(int argc, char* argv[]) {
    std::cout << "=== DBoW2 Loop Closure Detection ===" << std::endl;
    std::cout << std::endl;

    // Parameters
    const int k = 9;       // Branching factor
    const int L = 3;       // Depth levels (smaller for faster demo)
    const int direct_index_level = 3;  // Level for FeatureVector
    const double score_threshold = 0.03;  // Minimum similarity score
    const int temporal_gap = 5;  // Minimum frames between loop candidates

    std::cout << "Parameters:" << std::endl;
    std::cout << "  Vocabulary: k=" << k << ", L=" << L << std::endl;
    std::cout << "  Direct index level: " << direct_index_level << std::endl;
    std::cout << "  Score threshold: " << score_threshold << std::endl;
    std::cout << "  Temporal gap: " << temporal_gap << " frames" << std::endl;
    std::cout << std::endl;

    // Create ORB detector
    auto orb = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2,
                                cv::ORB::HARRIS_SCORE, 31, 20);

    // Generate image sequence with loop closures
    std::cout << "Generating image sequence with simulated loop closures..."
              << std::endl;
    const int num_unique_places = 12;
    std::vector<cv::Mat> image_sequence =
        generateSequenceWithLoops(num_unique_places);
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

                if (geometricVerification(keyframes[best_candidate],
                                          current_kf, matches)) {
                    std::cout << std::setw(10) << "LOOP!";
                    num_loops_detected++;
                } else {
                    std::cout << std::setw(10) << "rejected";
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

    // Print similarity matrix
    std::cout << "=== Pairwise Similarity Matrix (first 10 frames) ===" << std::endl;
    std::cout << std::endl;

    int display_size = std::min(10, static_cast<int>(keyframes.size()));
    std::cout << "     ";
    for (int j = 0; j < display_size; ++j) {
        std::cout << std::setw(6) << j;
    }
    std::cout << std::endl;

    for (int i = 0; i < display_size; ++i) {
        std::cout << std::setw(4) << i << " ";
        for (int j = 0; j < display_size; ++j) {
            double score = vocabulary.score(keyframes[i].bow_vector,
                                            keyframes[j].bow_vector);
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << score;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Note: High scores on diagonal indicate self-similarity." << std::endl;
    std::cout << "Look for high off-diagonal scores = potential loop closures." << std::endl;
    std::cout << std::endl;

    std::cout << "=== Loop Closure Detection Complete ===" << std::endl;

    return 0;
}
