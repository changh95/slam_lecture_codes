//
// Utility Functions for SuperPoint/SuperGlue
//

#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

/**
 * @brief Check if file exists
 */
inline bool FileExists(const std::string& path) {
    return fs::exists(path);
}

/**
 * @brief Concatenate folder and filename with proper separator
 */
inline std::string ConcatenateFolderAndFileName(const std::string& folder,
                                                 const std::string& filename) {
    if (folder.empty()) return filename;
    if (folder.back() == '/') return folder + filename;
    return folder + "/" + filename;
}

/**
 * @brief Get sorted list of image files in a directory
 */
inline void GetFileNames(const std::string& path, std::vector<std::string>& filenames) {
    filenames.clear();

    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            // Filter for image files
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                filenames.push_back(entry.path().string());
            }
        }
    }

    // Sort alphabetically
    std::sort(filenames.begin(), filenames.end());

    std::cout << "Found " << filenames.size() << " images in " << path << std::endl;
}

/**
 * @brief Visualize feature matches between two images
 *
 * Creates a side-by-side visualization with lines connecting matched keypoints.
 * Shows processing time and number of matches.
 */
inline void VisualizeMatching(const cv::Mat& image0,
                              const std::vector<cv::KeyPoint>& keypoints0,
                              const cv::Mat& image1,
                              const std::vector<cv::KeyPoint>& keypoints1,
                              const std::vector<cv::DMatch>& matches,
                              cv::Mat& output,
                              double time_ms = -1) {
    // Convert grayscale to BGR if needed
    cv::Mat img0_color, img1_color;
    if (image0.channels() == 1) {
        cv::cvtColor(image0, img0_color, cv::COLOR_GRAY2BGR);
    } else {
        img0_color = image0.clone();
    }

    if (image1.channels() == 1) {
        cv::cvtColor(image1, img1_color, cv::COLOR_GRAY2BGR);
    } else {
        img1_color = image1.clone();
    }

    // Draw matches
    cv::drawMatches(img0_color, keypoints0, img1_color, keypoints1,
                   matches, output,
                   cv::Scalar(0, 255, 0),   // Match color (green)
                   cv::Scalar(255, 0, 0),   // Single point color (blue)
                   std::vector<char>(),
                   cv::DrawMatchesFlags::DEFAULT);

    // Add info text
    std::string info = "Matches: " + std::to_string(matches.size());
    if (time_ms > 0) {
        info += " | Time: " + std::to_string(static_cast<int>(time_ms)) + " ms";
    }

    cv::putText(output, info, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8,
               cv::Scalar(0, 255, 0), 2);

    // Display
    cv::imshow("SuperPoint + SuperGlue Matches", output);
    cv::waitKey(1);
}

#endif // UTILS_H_
