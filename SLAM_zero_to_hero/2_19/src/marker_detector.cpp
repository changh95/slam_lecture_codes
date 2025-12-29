/**
 * @file marker_detector.cpp
 * @brief Detect ArUco markers in images or camera feed
 *
 * Demonstrates basic marker detection without pose estimation.
 * Useful for verifying marker visibility and detection quality.
 */

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  --image <file>     Input image file\n"
              << "  --camera <id>      Camera device ID (default: 0)\n"
              << "  --dict <name>      Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "  --output <file>    Save output image (optional)\n"
              << "  --no-display       Don't show GUI window\n"
              << "\nExamples:\n"
              << "  " << programName << " --image test.jpg\n"
              << "  " << programName << " --camera 0\n";
}

cv::aruco::Dictionary getDictionary(const std::string& name) {
    if (name == "4x4") {
        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    } else if (name == "5x5") {
        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    } else if (name == "6x6") {
        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    } else if (name == "7x7") {
        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);
    } else {
        std::cerr << "Unknown dictionary: " << name << ", using 6x6\n";
        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    }
}

void detectAndDraw(cv::Mat& image, cv::aruco::ArucoDetector& detector) {
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    std::vector<int> markerIds;

    // Detect markers
    detector.detectMarkers(image, markerCorners, markerIds, rejectedCandidates);

    // Draw detected markers
    if (!markerIds.empty()) {
        cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

        // Print detection info
        std::cout << "Detected " << markerIds.size() << " marker(s): ";
        for (size_t i = 0; i < markerIds.size(); i++) {
            std::cout << markerIds[i];
            if (i < markerIds.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";

        // Draw corner indices for each marker
        for (size_t i = 0; i < markerCorners.size(); i++) {
            const auto& corners = markerCorners[i];
            for (int j = 0; j < 4; j++) {
                cv::putText(image, std::to_string(j),
                           cv::Point(static_cast<int>(corners[j].x) + 5,
                                    static_cast<int>(corners[j].y) + 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4,
                           cv::Scalar(255, 0, 0), 1);
            }

            // Draw marker center
            cv::Point2f center(0, 0);
            for (const auto& corner : corners) {
                center += corner;
            }
            center *= 0.25f;
            cv::circle(image, center, 4, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Optionally draw rejected candidates for debugging
    if (!rejectedCandidates.empty()) {
        cv::aruco::drawDetectedMarkers(image, rejectedCandidates,
                                       cv::noArray(), cv::Scalar(100, 0, 255));
    }

    // Display info overlay
    std::string info = "Detected: " + std::to_string(markerIds.size()) +
                       " | Rejected: " + std::to_string(rejectedCandidates.size());
    cv::putText(image, info, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

void processImage(const std::string& imagePath, cv::aruco::ArucoDetector& detector,
                  const std::string& outputPath, bool display) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << "\n";
        return;
    }

    std::cout << "Processing image: " << imagePath << "\n";
    std::cout << "Size: " << image.cols << "x" << image.rows << "\n\n";

    detectAndDraw(image, detector);

    if (!outputPath.empty()) {
        cv::imwrite(outputPath, image);
        std::cout << "Saved output to: " << outputPath << "\n";
    }

    if (display) {
        cv::imshow("ArUco Detection", image);
        std::cout << "\nPress any key to exit...\n";
        cv::waitKey(0);
    }
}

void processCamera(int cameraId, cv::aruco::ArucoDetector& detector,
                   const std::string& outputPath, bool display) {
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraId << "\n";
        return;
    }

    std::cout << "Opened camera " << cameraId << "\n";
    std::cout << "Press 'q' to quit, 's' to save current frame\n\n";

    cv::Mat frame;
    int frameCount = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame\n";
            break;
        }

        // Detect markers (only print every 30 frames to reduce console spam)
        cv::Mat displayFrame = frame.clone();
        if (frameCount % 30 == 0) {
            detectAndDraw(displayFrame, detector);
        } else {
            std::vector<std::vector<cv::Point2f>> corners;
            std::vector<int> ids;
            detector.detectMarkers(displayFrame, corners, ids);
            if (!ids.empty()) {
                cv::aruco::drawDetectedMarkers(displayFrame, corners, ids);
            }
            std::string info = "Detected: " + std::to_string(ids.size());
            cv::putText(displayFrame, info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        if (display) {
            cv::imshow("ArUco Detection", displayFrame);
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == 's') {
            std::string filename = "capture_" + std::to_string(frameCount) + ".png";
            cv::imwrite(filename, frame);
            std::cout << "Saved: " << filename << "\n";
        }

        frameCount++;
    }

    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    // Default parameters
    std::string imagePath;
    int cameraId = -1;
    std::string dictName = "6x6";
    std::string outputPath;
    bool display = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraId = std::stoi(argv[++i]);
        } else if (arg == "--dict" && i + 1 < argc) {
            dictName = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--no-display") {
            display = false;
        }
    }

    std::cout << "=== ArUco Marker Detector ===\n\n";

    // Set up detector
    cv::aruco::Dictionary dictionary = getDictionary(dictName);
    cv::aruco::DetectorParameters params;

    // Tune detection parameters for better robustness
    params.adaptiveThreshWinSizeMin = 3;
    params.adaptiveThreshWinSizeMax = 23;
    params.adaptiveThreshWinSizeStep = 10;
    params.minMarkerPerimeterRate = 0.03;
    params.maxMarkerPerimeterRate = 4.0;
    params.polygonalApproxAccuracyRate = 0.03;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    cv::aruco::ArucoDetector detector(dictionary, params);

    std::cout << "Dictionary: " << dictName << "\n\n";

    if (!imagePath.empty()) {
        processImage(imagePath, detector, outputPath, display);
    } else {
        // Default to camera if no image specified
        if (cameraId < 0) cameraId = 0;
        processCamera(cameraId, detector, outputPath, display);
    }

    return 0;
}
