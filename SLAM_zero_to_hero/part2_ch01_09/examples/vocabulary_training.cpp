/**
 * Vocabulary Training Example using DBoW2
 *
 * This example demonstrates how to:
 * 1. Extract ORB features from training images
 * 2. Create a visual vocabulary using hierarchical k-means
 * 3. Save the vocabulary for later use
 *
 * The vocabulary is the foundation of the Bag of Visual Words approach
 * used in visual place recognition and loop closure detection.
 */

#include "DBoW2/DBoW2.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

// Vocabulary type using ORB descriptors
using OrbVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor,
                                                  DBoW2::FORB>;

/**
 * Convert cv::Mat descriptors to vector of single-row cv::Mat
 * DBoW2 expects descriptors in this format
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
 * Generate synthetic training images with random patterns
 * In real applications, you would load actual images from your dataset
 */
std::vector<cv::Mat> generateSyntheticTrainingImages(int num_images,
                                                      int width = 640,
                                                      int height = 480) {
    std::vector<cv::Mat> images;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> pos_dist(50, std::min(width, height) - 50);
    std::uniform_int_distribution<> size_dist(20, 80);
    std::uniform_int_distribution<> gray_dist(0, 255);
    std::uniform_int_distribution<> shape_dist(0, 2);

    std::cout << "Generating " << num_images << " synthetic training images..."
              << std::endl;

    for (int i = 0; i < num_images; ++i) {
        // Create base image with gradient background
        cv::Mat img(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                img.at<uchar>(y, x) = static_cast<uchar>(
                    (128 + 50 * std::sin(x * 0.01 + i) +
                     50 * std::cos(y * 0.01 + i)));
            }
        }

        // Add random shapes (circles, rectangles, lines) to create features
        int num_shapes = 15 + (i % 10);
        for (int j = 0; j < num_shapes; ++j) {
            int x = pos_dist(gen);
            int y = pos_dist(gen);
            int size = size_dist(gen);
            uchar color = static_cast<uchar>(gray_dist(gen));

            switch (shape_dist(gen)) {
                case 0:  // Circle
                    cv::circle(img, cv::Point(x, y), size / 2, cv::Scalar(color),
                               -1);
                    break;
                case 1:  // Rectangle
                    cv::rectangle(img, cv::Point(x, y),
                                  cv::Point(x + size, y + size),
                                  cv::Scalar(color), -1);
                    break;
                case 2:  // Line
                    cv::line(img, cv::Point(x, y),
                             cv::Point(x + size, y + size),
                             cv::Scalar(color), 2);
                    break;
            }
        }

        // Add some noise for more realistic features
        cv::Mat noise(height, width, CV_8UC1);
        cv::randn(noise, 0, 15);
        img += noise;

        // Add Gaussian blur to smooth edges
        cv::GaussianBlur(img, img, cv::Size(3, 3), 0);

        images.push_back(img);
    }

    return images;
}

int main(int argc, char* argv[]) {
    std::cout << "=== DBoW2 Vocabulary Training ===" << std::endl;
    std::cout << std::endl;

    // Vocabulary parameters
    // k: branching factor (children per node)
    // L: tree depth levels
    // Total possible words: k^L
    const int k = 10;      // Branching factor
    const int L = 5;       // Depth levels (10^5 = 100,000 possible words)

    // Weighting and scoring types
    const DBoW2::WeightingType weighting = DBoW2::TF_IDF;
    const DBoW2::ScoringType scoring = DBoW2::L1_NORM;

    std::cout << "Vocabulary Parameters:" << std::endl;
    std::cout << "  Branching factor (k): " << k << std::endl;
    std::cout << "  Depth levels (L):     " << L << std::endl;
    std::cout << "  Max words:            " << static_cast<int>(std::pow(k, L))
              << std::endl;
    std::cout << "  Weighting:            TF-IDF" << std::endl;
    std::cout << "  Scoring:              L1-NORM" << std::endl;
    std::cout << std::endl;

    // Number of training images
    const int num_training_images = 50;

    // Generate or load training images
    std::vector<cv::Mat> training_images =
        generateSyntheticTrainingImages(num_training_images);

    // Create ORB feature detector
    // Using default parameters similar to ORB-SLAM
    auto orb = cv::ORB::create(
        1000,   // nfeatures: max features per image
        1.2f,   // scaleFactor: pyramid decimation ratio
        8,      // nlevels: number of pyramid levels
        31,     // edgeThreshold: border where no features detected
        0,      // firstLevel: level of pyramid to put source image
        2,      // WTA_K: number of points for oriented BRIEF
        cv::ORB::HARRIS_SCORE,  // scoreType: HARRIS or FAST
        31,     // patchSize: size of patch used by oriented BRIEF
        20      // fastThreshold: threshold for FAST detector
    );

    // Extract features from all training images
    std::cout << "Extracting ORB features from training images..." << std::endl;
    std::vector<std::vector<cv::Mat>> all_features;
    int total_features = 0;

    auto start_extract = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < training_images.size(); ++i) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(training_images[i], cv::noArray(),
                              keypoints, descriptors);

        if (descriptors.empty()) {
            std::cout << "  Warning: No features in image " << i << std::endl;
            continue;
        }

        // Convert to DBoW2 format
        std::vector<cv::Mat> desc_vec = toDescriptorVector(descriptors);
        all_features.push_back(desc_vec);
        total_features += descriptors.rows;

        if ((i + 1) % 10 == 0) {
            std::cout << "  Processed " << (i + 1) << "/" << training_images.size()
                      << " images" << std::endl;
        }
    }

    auto end_extract = std::chrono::high_resolution_clock::now();
    auto extract_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_extract - start_extract).count();

    std::cout << std::endl;
    std::cout << "Feature Extraction Results:" << std::endl;
    std::cout << "  Total features extracted: " << total_features << std::endl;
    std::cout << "  Average features/image:   "
              << total_features / training_images.size() << std::endl;
    std::cout << "  Extraction time:          " << extract_time << " ms"
              << std::endl;
    std::cout << std::endl;

    // Create vocabulary using hierarchical k-means
    std::cout << "Creating vocabulary (this may take a while)..." << std::endl;

    auto start_voc = std::chrono::high_resolution_clock::now();

    OrbVocabulary vocabulary(k, L, weighting, scoring);
    vocabulary.create(all_features);

    auto end_voc = std::chrono::high_resolution_clock::now();
    auto voc_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_voc - start_voc).count();

    std::cout << std::endl;
    std::cout << "Vocabulary Created:" << std::endl;
    std::cout << "  Number of words:    " << vocabulary.size() << std::endl;
    std::cout << "  Creation time:      " << voc_time << " ms" << std::endl;
    std::cout << std::endl;

    // Save vocabulary to file
    const std::string voc_filename = "orb_vocabulary.yml.gz";
    std::cout << "Saving vocabulary to " << voc_filename << "..." << std::endl;

    auto start_save = std::chrono::high_resolution_clock::now();
    vocabulary.save(voc_filename);
    auto end_save = std::chrono::high_resolution_clock::now();
    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_save - start_save).count();

    std::cout << "  Save time: " << save_time << " ms" << std::endl;

    // Get file size
    if (std::filesystem::exists(voc_filename)) {
        auto file_size = std::filesystem::file_size(voc_filename);
        std::cout << "  File size: " << file_size / 1024 << " KB" << std::endl;
    }
    std::cout << std::endl;

    // Test loading the vocabulary
    std::cout << "Testing vocabulary loading..." << std::endl;

    auto start_load = std::chrono::high_resolution_clock::now();
    OrbVocabulary loaded_voc(voc_filename);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_load - start_load).count();

    std::cout << "  Loaded vocabulary with " << loaded_voc.size() << " words"
              << std::endl;
    std::cout << "  Load time: " << load_time << " ms" << std::endl;
    std::cout << std::endl;

    // Demonstrate vocabulary usage: convert an image to BoW vector
    std::cout << "Demonstrating BoW vector conversion..." << std::endl;

    // Use the first training image as a test
    cv::Mat test_image = training_images[0];
    std::vector<cv::KeyPoint> test_keypoints;
    cv::Mat test_descriptors;
    orb->detectAndCompute(test_image, cv::noArray(),
                          test_keypoints, test_descriptors);

    std::vector<cv::Mat> test_desc_vec = toDescriptorVector(test_descriptors);

    // Convert to BoW vector
    DBoW2::BowVector bow_vec;
    DBoW2::FeatureVector feat_vec;
    vocabulary.transform(test_desc_vec, bow_vec, feat_vec, 4);  // Direct index level 4

    std::cout << "  Input features:     " << test_descriptors.rows << std::endl;
    std::cout << "  BoW vector size:    " << bow_vec.size() << " (non-zero words)"
              << std::endl;
    std::cout << "  Feature vector nodes: " << feat_vec.size() << std::endl;
    std::cout << std::endl;

    // Show some entries of the BoW vector
    std::cout << "Sample BoW vector entries (word_id: weight):" << std::endl;
    int count = 0;
    for (const auto& entry : bow_vec) {
        std::cout << "  " << entry.first << ": " << entry.second << std::endl;
        if (++count >= 5) break;  // Show only first 5
    }
    std::cout << "  ..." << std::endl;
    std::cout << std::endl;

    std::cout << "=== Vocabulary Training Complete ===" << std::endl;
    std::cout << std::endl;
    std::cout << "The vocabulary can now be used for:" << std::endl;
    std::cout << "  1. Place recognition" << std::endl;
    std::cout << "  2. Loop closure detection" << std::endl;
    std::cout << "  3. Image retrieval" << std::endl;
    std::cout << std::endl;
    std::cout << "See loop_closure_detection example for usage." << std::endl;

    return 0;
}
