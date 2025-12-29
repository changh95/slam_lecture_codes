/**
 * @file charuco_calibration.cpp
 * @brief Camera calibration using ChArUco boards
 *
 * ChArUco boards combine the benefits of:
 * - ArUco markers: Easy detection, unique IDs
 * - Chessboard patterns: Sub-pixel accurate corners
 *
 * This makes ChArUco ideal for camera calibration, especially when:
 * - Board may be partially occluded
 * - Camera is far from board
 * - Fast calibration is needed
 *
 * The calibration process:
 * 1. Generate and print a ChArUco board
 * 2. Capture images from multiple viewpoints
 * 3. Detect board in each image
 * 4. Run camera calibration to estimate intrinsics
 * 5. Save calibration to file
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nBoard Generation:\n"
              << "  --generate           Generate ChArUco board image\n"
              << "  --board-output <f>   Output file for board (default: charuco_board.png)\n"
              << "\nCalibration:\n"
              << "  --camera <id>        Camera device ID (default: 0)\n"
              << "  --images <pattern>   Image file pattern (e.g., 'calib_*.png')\n"
              << "  --output <file>      Calibration output (default: camera_calib.yaml)\n"
              << "\nBoard Configuration:\n"
              << "  --cols <n>           Number of columns (default: 5)\n"
              << "  --rows <n>           Number of rows (default: 7)\n"
              << "  --square-size <m>    Square size in meters (default: 0.04)\n"
              << "  --marker-size <m>    Marker size in meters (default: 0.03)\n"
              << "  --dict <name>        Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "\nOptions:\n"
              << "  --min-frames <n>     Minimum frames for calibration (default: 15)\n"
              << "  --no-display         Don't show GUI\n"
              << "\nInteractive Controls (camera mode):\n"
              << "  'c' - Capture frame\n"
              << "  'r' - Run calibration\n"
              << "  's' - Save calibration\n"
              << "  'u' - Toggle undistortion preview\n"
              << "  'q' - Quit\n"
              << "\nExamples:\n"
              << "  " << programName << " --generate --cols 7 --rows 5 --board-output board.png\n"
              << "  " << programName << " --camera 0 --output my_camera.yaml\n"
              << "  " << programName << " --images 'calib_*.jpg' --output camera.yaml\n";
}

// Get predefined dictionary
cv::aruco::Dictionary getDictionary(const std::string& name) {
    if (name == "4x4") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    if (name == "5x5") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    if (name == "6x6") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    if (name == "7x7") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

// Generate ChArUco board image
void generateBoard(int cols, int rows, float squareSize, float markerSize,
                   const std::string& dictName, const std::string& outputPath) {
    cv::aruco::Dictionary dictionary = getDictionary(dictName);

    // Marker size should be smaller than square size
    float actualMarkerSize = std::min(markerSize, squareSize * 0.8f);

    cv::aruco::CharucoBoard board(
        cv::Size(cols, rows),
        squareSize,
        actualMarkerSize,
        dictionary
    );

    // Generate image
    int pixelsPerMeter = 2000;  // High resolution for printing
    cv::Size imageSize(
        static_cast<int>(cols * squareSize * pixelsPerMeter + squareSize * pixelsPerMeter),
        static_cast<int>(rows * squareSize * pixelsPerMeter + squareSize * pixelsPerMeter)
    );

    cv::Mat boardImage;
    board.generateImage(imageSize, boardImage, static_cast<int>(squareSize * pixelsPerMeter / 2), 1);

    if (cv::imwrite(outputPath, boardImage)) {
        std::cout << "Generated ChArUco board:\n";
        std::cout << "  Columns: " << cols << "\n";
        std::cout << "  Rows: " << rows << "\n";
        std::cout << "  Square size: " << squareSize << " m\n";
        std::cout << "  Marker size: " << actualMarkerSize << " m\n";
        std::cout << "  Dictionary: " << dictName << "\n";
        std::cout << "  Image size: " << imageSize.width << "x" << imageSize.height << "\n";
        std::cout << "  Output: " << outputPath << "\n\n";
        std::cout << "IMPORTANT: When printing, measure the actual printed sizes\n";
        std::cout << "and update --square-size and --marker-size accordingly.\n";
    } else {
        std::cerr << "Error: Failed to save board to " << outputPath << "\n";
    }
}

// Calibrator class to manage the calibration process
class CharucoCalibrator {
public:
    CharucoCalibrator(int cols, int rows, float squareSize, float markerSize,
                      const cv::aruco::Dictionary& dictionary)
        : cols_(cols), rows_(rows), squareSize_(squareSize), markerSize_(markerSize) {

        // Create board
        board_ = std::make_unique<cv::aruco::CharucoBoard>(
            cv::Size(cols, rows), squareSize, markerSize, dictionary
        );

        // Setup detectors
        cv::aruco::DetectorParameters arucoParams;
        arucoParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        arucoDetector_ = std::make_unique<cv::aruco::ArucoDetector>(dictionary, arucoParams);

        cv::aruco::CharucoParameters charucoParams;
        charucoDetector_ = std::make_unique<cv::aruco::CharucoDetector>(*board_, charucoParams);

        // Expected number of corners
        numExpectedCorners_ = (cols - 1) * (rows - 1);
        minCornersRequired_ = numExpectedCorners_ / 2;

        std::cout << "ChArUco board: " << cols << "x" << rows << "\n";
        std::cout << "Expected corners: " << numExpectedCorners_ << "\n";
        std::cout << "Minimum corners required: " << minCornersRequired_ << "\n\n";
    }

    // Process a frame and optionally capture it
    bool processFrame(cv::Mat& image, bool capture = false) {
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Detect ArUco markers
        std::vector<std::vector<cv::Point2f>> markerCorners;
        std::vector<int> markerIds;
        arucoDetector_->detectMarkers(gray, markerCorners, markerIds);

        // Draw detected markers
        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
        }

        // Detect ChArUco corners
        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;

        if (!markerCorners.empty()) {
            charucoDetector_->detectBoard(gray, charucoCorners, charucoIds,
                                          markerCorners, markerIds);
        }

        // Draw ChArUco corners
        if (!charucoCorners.empty()) {
            cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds);
        }

        // Update image size
        if (imageSize_.empty()) {
            imageSize_ = gray.size();
        }

        // Display info
        bool validFrame = charucoCorners.size() >= static_cast<size_t>(minCornersRequired_);

        std::stringstream ss;
        ss << "Corners: " << charucoCorners.size() << "/" << numExpectedCorners_;
        cv::putText(image, ss.str(), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        ss.str(""); ss.clear();
        ss << "Captured: " << allCharucoCorners_.size();
        cv::putText(image, ss.str(), cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        if (validFrame) {
            cv::putText(image, "VALID - Press 'c' to capture", cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(image, "Need more corners", cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        if (calibrated_) {
            ss.str(""); ss.clear();
            ss << std::fixed << std::setprecision(3)
               << "RMS: " << reprojectionError_ << " px";
            cv::putText(image, ss.str(), cv::Point(10, 120),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        }

        // Capture frame if requested and valid
        if (capture && validFrame) {
            allCharucoCorners_.push_back(charucoCorners);
            allCharucoIds_.push_back(charucoIds);
            std::cout << "Captured frame " << allCharucoCorners_.size()
                      << " with " << charucoCorners.size() << " corners\n";
            return true;
        }

        return false;
    }

    // Run camera calibration
    bool calibrate() {
        if (allCharucoCorners_.size() < 5) {
            std::cerr << "Error: Need at least 5 captured frames\n";
            return false;
        }

        std::cout << "\n=== Running Calibration ===\n";
        std::cout << "Frames: " << allCharucoCorners_.size() << "\n";
        std::cout << "Image size: " << imageSize_.width << "x" << imageSize_.height << "\n\n";

        // Initialize camera matrix with reasonable guess
        cameraMatrix_ = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix_.at<double>(0, 0) = imageSize_.width;   // fx
        cameraMatrix_.at<double>(1, 1) = imageSize_.width;   // fy
        cameraMatrix_.at<double>(0, 2) = imageSize_.width / 2.0;   // cx
        cameraMatrix_.at<double>(1, 2) = imageSize_.height / 2.0;  // cy

        distCoeffs_ = cv::Mat::zeros(5, 1, CV_64F);

        std::vector<cv::Mat> rvecs, tvecs;

        // Run calibration
        reprojectionError_ = cv::aruco::calibrateCameraCharuco(
            allCharucoCorners_, allCharucoIds_, *board_, imageSize_,
            cameraMatrix_, distCoeffs_, rvecs, tvecs,
            cv::CALIB_FIX_ASPECT_RATIO
        );

        calibrated_ = true;

        // Print results
        std::cout << "=== Calibration Results ===\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Reprojection error: " << reprojectionError_ << " pixels\n\n";

        std::cout << "Camera matrix:\n" << cameraMatrix_ << "\n\n";
        std::cout << "Distortion coefficients:\n" << distCoeffs_.t() << "\n\n";

        double fx = cameraMatrix_.at<double>(0, 0);
        double fy = cameraMatrix_.at<double>(1, 1);
        double cx = cameraMatrix_.at<double>(0, 2);
        double cy = cameraMatrix_.at<double>(1, 2);

        std::cout << "Intrinsic parameters:\n";
        std::cout << "  fx = " << fx << " px\n";
        std::cout << "  fy = " << fy << " px\n";
        std::cout << "  cx = " << cx << " px\n";
        std::cout << "  cy = " << cy << " px\n\n";

        std::cout << "Distortion (k1, k2, p1, p2, k3):\n";
        for (int i = 0; i < distCoeffs_.rows; i++) {
            std::cout << "  [" << i << "] = " << distCoeffs_.at<double>(i) << "\n";
        }

        return true;
    }

    // Save calibration to YAML file
    bool saveCalibration(const std::string& filename) {
        if (!calibrated_) {
            std::cerr << "Error: Not calibrated yet\n";
            return false;
        }

        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Error: Cannot open file: " << filename << "\n";
            return false;
        }

        fs << "image_width" << imageSize_.width;
        fs << "image_height" << imageSize_.height;
        fs << "camera_matrix" << cameraMatrix_;
        fs << "distortion_coefficients" << distCoeffs_;
        fs << "reprojection_error" << reprojectionError_;
        fs << "num_frames" << static_cast<int>(allCharucoCorners_.size());
        fs << "board_cols" << cols_;
        fs << "board_rows" << rows_;
        fs << "square_size" << squareSize_;
        fs << "marker_size" << markerSize_;

        fs.release();

        std::cout << "\nCalibration saved to: " << filename << "\n";
        return true;
    }

    // Show undistorted comparison
    void showUndistorted(const cv::Mat& image) {
        if (!calibrated_) return;

        cv::Mat undistorted;
        cv::undistort(image, undistorted, cameraMatrix_, distCoeffs_);

        // Side by side comparison
        cv::Mat combined;
        cv::hconcat(image, undistorted, combined);
        cv::resize(combined, combined, cv::Size(), 0.5, 0.5);

        cv::putText(combined, "Original", cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(combined, "Undistorted", cv::Point(combined.cols / 2 + 10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Undistortion", combined);
    }

    size_t numFrames() const { return allCharucoCorners_.size(); }
    bool isCalibrated() const { return calibrated_; }

private:
    int cols_, rows_;
    float squareSize_, markerSize_;
    int numExpectedCorners_, minCornersRequired_;

    std::unique_ptr<cv::aruco::CharucoBoard> board_;
    std::unique_ptr<cv::aruco::ArucoDetector> arucoDetector_;
    std::unique_ptr<cv::aruco::CharucoDetector> charucoDetector_;

    std::vector<std::vector<cv::Point2f>> allCharucoCorners_;
    std::vector<std::vector<int>> allCharucoIds_;
    cv::Size imageSize_;

    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    double reprojectionError_ = 0.0;
    bool calibrated_ = false;
};

int main(int argc, char** argv) {
    // Default parameters
    bool generate = false;
    std::string boardOutput = "charuco_board.png";
    int cameraId = 0;
    std::string imagePattern;
    std::string outputFile = "camera_calib.yaml";
    int cols = 5;
    int rows = 7;
    float squareSize = 0.04f;
    float markerSize = 0.03f;
    std::string dictName = "6x6";
    int minFrames = 15;
    bool display = true;
    bool useCamera = true;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--generate") {
            generate = true;
        } else if (arg == "--board-output" && i + 1 < argc) {
            boardOutput = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraId = std::stoi(argv[++i]);
            useCamera = true;
        } else if (arg == "--images" && i + 1 < argc) {
            imagePattern = argv[++i];
            useCamera = false;
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--cols" && i + 1 < argc) {
            cols = std::stoi(argv[++i]);
        } else if (arg == "--rows" && i + 1 < argc) {
            rows = std::stoi(argv[++i]);
        } else if (arg == "--square-size" && i + 1 < argc) {
            squareSize = std::stof(argv[++i]);
        } else if (arg == "--marker-size" && i + 1 < argc) {
            markerSize = std::stof(argv[++i]);
        } else if (arg == "--dict" && i + 1 < argc) {
            dictName = argv[++i];
        } else if (arg == "--min-frames" && i + 1 < argc) {
            minFrames = std::stoi(argv[++i]);
        } else if (arg == "--no-display") {
            display = false;
        }
    }

    std::cout << "=== ChArUco Camera Calibration ===\n\n";

    // Generate board if requested
    if (generate) {
        generateBoard(cols, rows, squareSize, markerSize, dictName, boardOutput);
        return 0;
    }

    // Display configuration
    std::cout << "Board configuration:\n";
    std::cout << "  Columns: " << cols << "\n";
    std::cout << "  Rows: " << rows << "\n";
    std::cout << "  Square size: " << squareSize << " m\n";
    std::cout << "  Marker size: " << markerSize << " m\n";
    std::cout << "  Dictionary: " << dictName << "\n";
    std::cout << "  Min frames: " << minFrames << "\n";
    std::cout << "  Output: " << outputFile << "\n\n";

    // Create calibrator
    cv::aruco::Dictionary dictionary = getDictionary(dictName);
    CharucoCalibrator calibrator(cols, rows, squareSize, markerSize, dictionary);

    if (useCamera) {
        // Camera mode - interactive calibration
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera " << cameraId << "\n";
            return 1;
        }

        std::cout << "Camera opened. Controls:\n";
        std::cout << "  'c' - Capture frame\n";
        std::cout << "  'r' - Run calibration\n";
        std::cout << "  's' - Save calibration\n";
        std::cout << "  'u' - Toggle undistortion\n";
        std::cout << "  'q' - Quit\n\n";

        bool showUndistort = false;

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            cv::Mat displayFrame = frame.clone();
            calibrator.processFrame(displayFrame, false);

            if (display) {
                cv::imshow("ChArUco Calibration", displayFrame);

                if (showUndistort && calibrator.isCalibrated()) {
                    calibrator.showUndistorted(frame);
                }
            }

            int key = cv::waitKey(30) & 0xFF;

            if (key == 'q' || key == 27) {
                break;
            } else if (key == 'c') {
                cv::Mat captureFrame = frame.clone();
                calibrator.processFrame(captureFrame, true);
            } else if (key == 'r') {
                if (calibrator.numFrames() >= static_cast<size_t>(minFrames)) {
                    calibrator.calibrate();
                } else {
                    std::cout << "Need " << minFrames << " frames, have "
                              << calibrator.numFrames() << "\n";
                }
            } else if (key == 's') {
                calibrator.saveCalibration(outputFile);
            } else if (key == 'u') {
                showUndistort = !showUndistort;
                if (!showUndistort) {
                    cv::destroyWindow("Undistortion");
                }
            }
        }

        cap.release();
        cv::destroyAllWindows();

    } else {
        // Batch mode - process images from pattern
        std::vector<cv::String> imageFiles;
        cv::glob(imagePattern, imageFiles, false);

        if (imageFiles.empty()) {
            std::cerr << "Error: No images match pattern: " << imagePattern << "\n";
            return 1;
        }

        std::cout << "Found " << imageFiles.size() << " images\n\n";

        for (const auto& imagePath : imageFiles) {
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "Warning: Cannot load " << imagePath << "\n";
                continue;
            }

            std::cout << "Processing: " << imagePath << "\n";
            cv::Mat displayImage = image.clone();

            if (calibrator.processFrame(displayImage, true)) {
                // Captured
            }

            if (display) {
                cv::imshow("ChArUco Calibration", displayImage);
                cv::waitKey(100);
            }
        }

        cv::destroyAllWindows();

        // Run calibration
        if (calibrator.numFrames() >= static_cast<size_t>(minFrames)) {
            if (calibrator.calibrate()) {
                calibrator.saveCalibration(outputFile);
            }
        } else {
            std::cerr << "Error: Not enough valid frames. Got "
                      << calibrator.numFrames() << ", need " << minFrames << "\n";
            return 1;
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
