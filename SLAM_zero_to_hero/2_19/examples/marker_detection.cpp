/**
 * @file marker_detection.cpp
 * @brief ArUco marker detection and generation demo
 *
 * This demo demonstrates:
 * - Generating ArUco markers for printing
 * - Detecting ArUco markers in images and camera feed
 * - Visualizing detection results with corner indices and IDs
 *
 * ArUco markers are square fiducial markers used for:
 * - Robot localization
 * - Augmented reality
 * - Camera calibration
 * - Ground truth collection for SLAM
 */

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

// Structure to hold marker information
struct DetectedMarker {
    int id;
    std::vector<cv::Point2f> corners;
    cv::Point2f center;
    double area;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nDetection:\n"
              << "  --image <file>     Input image file\n"
              << "  --camera <id>      Camera device ID (default: 0)\n"
              << "  --dict <name>      Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "  --output <file>    Save output image\n"
              << "  --no-display       Don't show GUI window\n"
              << "\nGeneration:\n"
              << "  --generate         Generate marker instead of detecting\n"
              << "  --id <n>           Marker ID to generate (default: 0)\n"
              << "  --size <pixels>    Marker size in pixels (default: 200)\n"
              << "  --border <pixels>  White border size (default: 20)\n"
              << "\nExamples:\n"
              << "  " << programName << " --generate --id 42 --output marker_42.png\n"
              << "  " << programName << " --image test.jpg\n"
              << "  " << programName << " --camera 0 --dict 6x6\n";
}

// Get predefined dictionary by name
cv::aruco::Dictionary getDictionary(const std::string& name) {
    static const std::map<std::string, cv::aruco::PredefinedDictionaryType> dictMap = {
        {"4x4_50", cv::aruco::DICT_4X4_50},
        {"4x4_100", cv::aruco::DICT_4X4_100},
        {"4x4_250", cv::aruco::DICT_4X4_250},
        {"4x4_1000", cv::aruco::DICT_4X4_1000},
        {"4x4", cv::aruco::DICT_4X4_50},
        {"5x5_50", cv::aruco::DICT_5X5_50},
        {"5x5_100", cv::aruco::DICT_5X5_100},
        {"5x5_250", cv::aruco::DICT_5X5_250},
        {"5x5_1000", cv::aruco::DICT_5X5_1000},
        {"5x5", cv::aruco::DICT_5X5_100},
        {"6x6_50", cv::aruco::DICT_6X6_50},
        {"6x6_100", cv::aruco::DICT_6X6_100},
        {"6x6_250", cv::aruco::DICT_6X6_250},
        {"6x6_1000", cv::aruco::DICT_6X6_1000},
        {"6x6", cv::aruco::DICT_6X6_250},
        {"7x7_50", cv::aruco::DICT_7X7_50},
        {"7x7_100", cv::aruco::DICT_7X7_100},
        {"7x7_250", cv::aruco::DICT_7X7_250},
        {"7x7_1000", cv::aruco::DICT_7X7_1000},
        {"7x7", cv::aruco::DICT_7X7_1000},
        {"apriltag_16h5", cv::aruco::DICT_APRILTAG_16h5},
        {"apriltag_25h9", cv::aruco::DICT_APRILTAG_25h9},
        {"apriltag_36h10", cv::aruco::DICT_APRILTAG_36h10},
        {"apriltag_36h11", cv::aruco::DICT_APRILTAG_36h11}
    };

    auto it = dictMap.find(name);
    if (it != dictMap.end()) {
        return cv::aruco::getPredefinedDictionary(it->second);
    }

    std::cerr << "Unknown dictionary: " << name << ", using 6x6_250\n";
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

// Generate a single ArUco marker
void generateMarker(int id, int size, int borderSize,
                    const std::string& dictName, const std::string& outputPath) {
    cv::aruco::Dictionary dictionary = getDictionary(dictName);

    // Check valid ID range
    int maxId = static_cast<int>(dictionary.bytesList.rows);
    if (id < 0 || id >= maxId) {
        std::cerr << "Error: Marker ID must be in range [0, " << maxId - 1 << "] for " << dictName << "\n";
        return;
    }

    // Generate marker image
    cv::Mat markerImage;
    cv::aruco::generateImageMarker(dictionary, id, size, markerImage, 1);

    // Add white border for printing
    cv::Mat outputImage;
    cv::copyMakeBorder(markerImage, outputImage,
                       borderSize, borderSize, borderSize, borderSize,
                       cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // Save marker
    if (cv::imwrite(outputPath, outputImage)) {
        std::cout << "Generated ArUco marker:\n";
        std::cout << "  ID: " << id << "\n";
        std::cout << "  Dictionary: " << dictName << "\n";
        std::cout << "  Size: " << size << " pixels\n";
        std::cout << "  Border: " << borderSize << " pixels\n";
        std::cout << "  Total size: " << outputImage.cols << "x" << outputImage.rows << "\n";
        std::cout << "  Output: " << outputPath << "\n";
    } else {
        std::cerr << "Error: Failed to save marker to " << outputPath << "\n";
    }
}

// Detect markers in an image and return detailed information
std::vector<DetectedMarker> detectMarkers(
    const cv::Mat& image,
    cv::aruco::ArucoDetector& detector,
    std::vector<std::vector<cv::Point2f>>& rejectedCandidates) {

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;

    detector.detectMarkers(image, corners, ids, rejectedCandidates);

    std::vector<DetectedMarker> markers;
    for (size_t i = 0; i < ids.size(); i++) {
        DetectedMarker marker;
        marker.id = ids[i];
        marker.corners = corners[i];

        // Compute center
        marker.center = cv::Point2f(0, 0);
        for (const auto& corner : corners[i]) {
            marker.center += corner;
        }
        marker.center *= 0.25f;

        // Compute area using shoelace formula
        const auto& c = corners[i];
        marker.area = 0.5f * std::abs(
            (c[0].x * c[1].y - c[1].x * c[0].y) +
            (c[1].x * c[2].y - c[2].x * c[1].y) +
            (c[2].x * c[3].y - c[3].x * c[2].y) +
            (c[3].x * c[0].y - c[0].x * c[3].y)
        );

        markers.push_back(marker);
    }

    return markers;
}

// Draw detection results on image
void drawDetectionResults(cv::Mat& image,
                          const std::vector<DetectedMarker>& markers,
                          const std::vector<std::vector<cv::Point2f>>& rejected,
                          bool showRejected = false) {
    // Draw detected markers
    for (const auto& marker : markers) {
        // Draw marker boundary
        std::vector<std::vector<cv::Point2f>> markerCorners = {marker.corners};
        std::vector<int> markerIds = {marker.id};
        cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

        // Draw corner indices (0-3)
        for (int j = 0; j < 4; j++) {
            cv::putText(image, std::to_string(j),
                       cv::Point(static_cast<int>(marker.corners[j].x) + 5,
                                static_cast<int>(marker.corners[j].y) - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
        }

        // Draw center point
        cv::circle(image, marker.center, 4, cv::Scalar(0, 0, 255), -1);

        // Draw area text
        std::stringstream ss;
        ss << "Area: " << static_cast<int>(marker.area);
        cv::putText(image, ss.str(),
                   cv::Point(static_cast<int>(marker.center.x) - 30,
                            static_cast<int>(marker.center.y) + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
    }

    // Optionally draw rejected candidates (for debugging)
    if (showRejected && !rejected.empty()) {
        cv::aruco::drawDetectedMarkers(image, rejected, cv::noArray(),
                                       cv::Scalar(100, 0, 255));
    }

    // Draw info overlay
    std::stringstream info;
    info << "Detected: " << markers.size();
    if (showRejected) {
        info << " | Rejected: " << rejected.size();
    }
    cv::putText(image, info.str(), cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

// Process a single image
void processImage(const std::string& imagePath,
                  cv::aruco::ArucoDetector& detector,
                  const std::string& outputPath,
                  bool display) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << "\n";
        return;
    }

    std::cout << "Processing image: " << imagePath << "\n";
    std::cout << "Size: " << image.cols << "x" << image.rows << "\n\n";

    // Detect markers
    std::vector<std::vector<cv::Point2f>> rejected;
    auto markers = detectMarkers(image, detector, rejected);

    // Print detection results
    if (!markers.empty()) {
        std::cout << "Detected " << markers.size() << " marker(s):\n";
        for (const auto& marker : markers) {
            std::cout << "  Marker ID " << marker.id << ":\n";
            std::cout << "    Center: (" << marker.center.x << ", " << marker.center.y << ")\n";
            std::cout << "    Area: " << marker.area << " pixels^2\n";
            std::cout << "    Corners:\n";
            for (int j = 0; j < 4; j++) {
                std::cout << "      [" << j << "]: ("
                          << marker.corners[j].x << ", "
                          << marker.corners[j].y << ")\n";
            }
        }
    } else {
        std::cout << "No markers detected.\n";
    }

    // Draw results
    drawDetectionResults(image, markers, rejected, true);

    // Save output if specified
    if (!outputPath.empty()) {
        if (cv::imwrite(outputPath, image)) {
            std::cout << "\nSaved output to: " << outputPath << "\n";
        }
    }

    // Display if requested
    if (display) {
        cv::imshow("ArUco Detection", image);
        std::cout << "\nPress any key to exit...\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

// Process camera feed
void processCamera(int cameraId,
                   cv::aruco::ArucoDetector& detector,
                   bool display) {
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera " << cameraId << "\n";
        return;
    }

    std::cout << "Camera opened. Press:\n";
    std::cout << "  'q' or ESC - Quit\n";
    std::cout << "  's' - Save current frame\n";
    std::cout << "  'r' - Toggle rejected candidates display\n\n";

    bool showRejected = false;
    int frameCount = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture frame\n";
            break;
        }

        // Detect markers
        std::vector<std::vector<cv::Point2f>> rejected;
        auto markers = detectMarkers(frame, detector, rejected);

        // Draw results
        drawDetectionResults(frame, markers, rejected, showRejected);

        // Add FPS counter
        static auto lastTime = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        double fps = 1.0 / std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;

        std::stringstream fpsText;
        fpsText << "FPS: " << static_cast<int>(fps);
        cv::putText(frame, fpsText.str(), cv::Point(frame.cols - 100, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // Display
        if (display) {
            cv::imshow("ArUco Detection", frame);
        }

        // Print detection info periodically
        if (frameCount % 30 == 0 && !markers.empty()) {
            std::cout << "Frame " << frameCount << ": Detected markers: ";
            for (size_t i = 0; i < markers.size(); i++) {
                std::cout << markers[i].id;
                if (i < markers.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        }

        // Handle key presses
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == 's') {
            std::string filename = "capture_" + std::to_string(frameCount) + ".png";
            cv::imwrite(filename, frame);
            std::cout << "Saved: " << filename << "\n";
        } else if (key == 'r') {
            showRejected = !showRejected;
            std::cout << "Show rejected: " << (showRejected ? "ON" : "OFF") << "\n";
        }

        frameCount++;
    }

    cap.release();
    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    // Default parameters
    bool generate = false;
    int markerId = 0;
    int markerSize = 200;
    int borderSize = 20;
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
        } else if (arg == "--generate") {
            generate = true;
        } else if (arg == "--id" && i + 1 < argc) {
            markerId = std::stoi(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            markerSize = std::stoi(argv[++i]);
        } else if (arg == "--border" && i + 1 < argc) {
            borderSize = std::stoi(argv[++i]);
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

    std::cout << "=== ArUco Marker Detection Demo ===\n\n";

    // Generate marker
    if (generate) {
        if (outputPath.empty()) {
            outputPath = "marker_" + std::to_string(markerId) + ".png";
        }
        generateMarker(markerId, markerSize, borderSize, dictName, outputPath);
        return 0;
    }

    // Set up detector
    cv::aruco::Dictionary dictionary = getDictionary(dictName);
    cv::aruco::DetectorParameters params;

    // Tune detection parameters for robustness
    params.adaptiveThreshWinSizeMin = 3;
    params.adaptiveThreshWinSizeMax = 23;
    params.adaptiveThreshWinSizeStep = 10;
    params.minMarkerPerimeterRate = 0.03;
    params.maxMarkerPerimeterRate = 4.0;
    params.polygonalApproxAccuracyRate = 0.03;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params.cornerRefinementWinSize = 5;
    params.cornerRefinementMaxIterations = 30;
    params.cornerRefinementMinAccuracy = 0.1;

    cv::aruco::ArucoDetector detector(dictionary, params);

    std::cout << "Dictionary: " << dictName << "\n\n";

    // Process image or camera
    if (!imagePath.empty()) {
        processImage(imagePath, detector, outputPath, display);
    } else {
        if (cameraId < 0) cameraId = 0;
        processCamera(cameraId, detector, display);
    }

    std::cout << "\nDone.\n";
    return 0;
}
