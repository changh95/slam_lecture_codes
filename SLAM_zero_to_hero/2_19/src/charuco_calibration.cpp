/**
 * @file charuco_calibration.cpp
 * @brief Camera calibration using ChArUco boards
 *
 * ChArUco boards combine ArUco markers with chessboard patterns for
 * robust detection and sub-pixel accurate corner localization.
 * This is the recommended approach for camera calibration in SLAM systems.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  --camera <id>         Camera device ID (default: 0)\n"
              << "  --images <pattern>    Image file pattern (e.g., 'calib_*.png')\n"
              << "  --cols <n>            Board columns (default: 5)\n"
              << "  --rows <n>            Board rows (default: 7)\n"
              << "  --square-size <m>     Chessboard square size in meters (default: 0.04)\n"
              << "  --marker-size <m>     Marker size in meters (default: 0.03)\n"
              << "  --dict <name>         Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "  --output <file>       Output calibration file (default: camera_calib.yaml)\n"
              << "  --min-frames <n>      Minimum frames for calibration (default: 15)\n"
              << "\nExamples:\n"
              << "  " << programName << " --camera 0 --cols 5 --rows 7\n"
              << "  " << programName << " --images 'calib_*.png' --output my_camera.yaml\n"
              << "\nDuring capture:\n"
              << "  'c' - Capture current frame\n"
              << "  'r' - Run calibration with captured frames\n"
              << "  's' - Save calibration result\n"
              << "  'q' - Quit\n";
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
        return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    }
}

class CharucoCalibrator {
public:
    CharucoCalibrator(int cols, int rows, float squareSize, float markerSize,
                      const cv::aruco::Dictionary& dictionary)
        : cols_(cols), rows_(rows), squareSize_(squareSize), markerSize_(markerSize),
          calibrated_(false) {

        // Create CharUco board
        board_ = cv::aruco::CharucoBoard(
            cv::Size(cols, rows), squareSize, markerSize, dictionary
        );

        // Set up detector
        cv::aruco::DetectorParameters params;
        params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        detector_ = std::make_unique<cv::aruco::ArucoDetector>(dictionary, params);
        charucoDetector_ = std::make_unique<cv::aruco::CharucoDetector>(board_);
    }

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
        detector_->detectMarkers(gray, markerCorners, markerIds);

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

        // Display info
        int minCorners = (cols_ - 1) * (rows_ - 1) / 2;
        std::stringstream ss;
        ss << "ChArUco corners: " << charucoCorners.size() << "/" << (cols_ - 1) * (rows_ - 1);
        cv::putText(image, ss.str(), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        ss.str(""); ss.clear();
        ss << "Captured frames: " << allCharucoCorners_.size();
        cv::putText(image, ss.str(), cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        bool validFrame = charucoCorners.size() >= static_cast<size_t>(minCorners);

        if (validFrame) {
            cv::putText(image, "GOOD - Press 'c' to capture", cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(image, "Need more corners visible", cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        // Capture frame if requested
        if (capture && validFrame) {
            allCharucoCorners_.push_back(charucoCorners);
            allCharucoIds_.push_back(charucoIds);
            imageSize_ = gray.size();
            std::cout << "Captured frame " << allCharucoCorners_.size()
                      << " with " << charucoCorners.size() << " corners\n";
            return true;
        }

        return false;
    }

    bool calibrate() {
        if (allCharucoCorners_.size() < 5) {
            std::cerr << "Error: Need at least 5 captured frames\n";
            return false;
        }

        std::cout << "\nRunning calibration with " << allCharucoCorners_.size() << " frames...\n";

        // Initialize camera matrix with reasonable guess
        cameraMatrix_ = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix_.at<double>(0, 0) = imageSize_.width;  // fx
        cameraMatrix_.at<double>(1, 1) = imageSize_.width;  // fy
        cameraMatrix_.at<double>(0, 2) = imageSize_.width / 2.0;  // cx
        cameraMatrix_.at<double>(1, 2) = imageSize_.height / 2.0; // cy

        distCoeffs_ = cv::Mat::zeros(5, 1, CV_64F);

        std::vector<cv::Mat> rvecs, tvecs;

        // Run calibration
        double repError = cv::aruco::calibrateCameraCharuco(
            allCharucoCorners_, allCharucoIds_, board_, imageSize_,
            cameraMatrix_, distCoeffs_, rvecs, tvecs,
            cv::CALIB_FIX_ASPECT_RATIO
        );

        calibrated_ = true;
        reprojectionError_ = repError;

        std::cout << "\n=== Calibration Results ===\n";
        std::cout << "Reprojection error: " << repError << " pixels\n";
        std::cout << "\nCamera matrix:\n" << cameraMatrix_ << "\n";
        std::cout << "\nDistortion coefficients:\n" << distCoeffs_.t() << "\n";

        // Extract parameters
        double fx = cameraMatrix_.at<double>(0, 0);
        double fy = cameraMatrix_.at<double>(1, 1);
        double cx = cameraMatrix_.at<double>(0, 2);
        double cy = cameraMatrix_.at<double>(1, 2);

        std::cout << "\nIntrinsic parameters:\n";
        std::cout << "  fx = " << fx << "\n";
        std::cout << "  fy = " << fy << "\n";
        std::cout << "  cx = " << cx << "\n";
        std::cout << "  cy = " << cy << "\n";

        if (distCoeffs_.rows >= 5) {
            std::cout << "\nDistortion (k1, k2, p1, p2, k3):\n";
            std::cout << "  k1 = " << distCoeffs_.at<double>(0) << "\n";
            std::cout << "  k2 = " << distCoeffs_.at<double>(1) << "\n";
            std::cout << "  p1 = " << distCoeffs_.at<double>(2) << "\n";
            std::cout << "  p2 = " << distCoeffs_.at<double>(3) << "\n";
            std::cout << "  k3 = " << distCoeffs_.at<double>(4) << "\n";
        }

        return true;
    }

    bool saveCalibration(const std::string& filename) {
        if (!calibrated_) {
            std::cerr << "Error: Not calibrated yet\n";
            return false;
        }

        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open file for writing: " << filename << "\n";
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

    void showUndistorted(const cv::Mat& image) {
        if (!calibrated_) return;

        cv::Mat undistorted;
        cv::undistort(image, undistorted, cameraMatrix_, distCoeffs_);

        // Create side-by-side comparison
        cv::Mat combined;
        cv::hconcat(image, undistorted, combined);
        cv::resize(combined, combined, cv::Size(), 0.5, 0.5);

        cv::putText(combined, "Original", cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(combined, "Undistorted", cv::Point(combined.cols / 2 + 10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Undistortion Comparison", combined);
    }

    size_t numCapturedFrames() const { return allCharucoCorners_.size(); }
    bool isCalibrated() const { return calibrated_; }

private:
    int cols_, rows_;
    float squareSize_, markerSize_;
    cv::aruco::CharucoBoard board_;
    std::unique_ptr<cv::aruco::ArucoDetector> detector_;
    std::unique_ptr<cv::aruco::CharucoDetector> charucoDetector_;

    std::vector<std::vector<cv::Point2f>> allCharucoCorners_;
    std::vector<std::vector<int>> allCharucoIds_;
    cv::Size imageSize_;

    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    double reprojectionError_;
    bool calibrated_;
};

int main(int argc, char** argv) {
    // Default parameters
    int cameraId = 0;
    std::string imagePattern;
    int cols = 5;
    int rows = 7;
    float squareSize = 0.04f;
    float markerSize = 0.03f;
    std::string dictName = "6x6";
    std::string outputFile = "camera_calib.yaml";
    int minFrames = 15;
    bool useCamera = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraId = std::stoi(argv[++i]);
            useCamera = true;
        } else if (arg == "--images" && i + 1 < argc) {
            imagePattern = argv[++i];
            useCamera = false;
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
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--min-frames" && i + 1 < argc) {
            minFrames = std::stoi(argv[++i]);
        }
    }

    std::cout << "=== ChArUco Camera Calibration ===\n\n";
    std::cout << "Board configuration:\n";
    std::cout << "  Columns: " << cols << "\n";
    std::cout << "  Rows: " << rows << "\n";
    std::cout << "  Square size: " << squareSize << " m\n";
    std::cout << "  Marker size: " << markerSize << " m\n";
    std::cout << "  Dictionary: " << dictName << "\n";
    std::cout << "  Output: " << outputFile << "\n\n";

    // Create calibrator
    cv::aruco::Dictionary dictionary = getDictionary(dictName);
    CharucoCalibrator calibrator(cols, rows, squareSize, markerSize, dictionary);

    if (useCamera) {
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << cameraId << "\n";
            return 1;
        }

        std::cout << "Camera opened. Instructions:\n";
        std::cout << "  'c' - Capture current frame\n";
        std::cout << "  'r' - Run calibration\n";
        std::cout << "  's' - Save calibration\n";
        std::cout << "  'u' - Toggle undistortion preview\n";
        std::cout << "  'q' - Quit\n\n";

        cv::Mat frame;
        bool showUndistort = false;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            cv::Mat displayFrame = frame.clone();
            calibrator.processFrame(displayFrame, false);

            cv::imshow("ChArUco Calibration", displayFrame);

            if (showUndistort && calibrator.isCalibrated()) {
                calibrator.showUndistorted(frame);
            }

            int key = cv::waitKey(30) & 0xFF;

            if (key == 'q' || key == 27) {
                break;
            } else if (key == 'c') {
                cv::Mat captureFrame = frame.clone();
                calibrator.processFrame(captureFrame, true);
            } else if (key == 'r') {
                if (calibrator.numCapturedFrames() >= static_cast<size_t>(minFrames)) {
                    calibrator.calibrate();
                } else {
                    std::cout << "Need at least " << minFrames << " frames. Current: "
                              << calibrator.numCapturedFrames() << "\n";
                }
            } else if (key == 's') {
                calibrator.saveCalibration(outputFile);
            } else if (key == 'u') {
                showUndistort = !showUndistort;
                if (!showUndistort) {
                    cv::destroyWindow("Undistortion Comparison");
                }
            }
        }

        cap.release();
        cv::destroyAllWindows();
    } else {
        // Process images from pattern
        std::vector<cv::String> imageFiles;
        cv::glob(imagePattern, imageFiles, false);

        if (imageFiles.empty()) {
            std::cerr << "Error: No images found matching pattern: " << imagePattern << "\n";
            return 1;
        }

        std::cout << "Found " << imageFiles.size() << " images\n\n";

        for (const auto& imagePath : imageFiles) {
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "Warning: Could not load " << imagePath << "\n";
                continue;
            }

            std::cout << "Processing: " << imagePath << "\n";
            cv::Mat displayImage = image.clone();

            if (calibrator.processFrame(displayImage, true)) {
                // Successfully captured
            }

            cv::imshow("ChArUco Calibration", displayImage);
            cv::waitKey(100);
        }

        cv::destroyAllWindows();

        // Run calibration
        if (calibrator.numCapturedFrames() >= static_cast<size_t>(minFrames)) {
            if (calibrator.calibrate()) {
                calibrator.saveCalibration(outputFile);
            }
        } else {
            std::cerr << "Error: Not enough valid frames. Got "
                      << calibrator.numCapturedFrames() << ", need " << minFrames << "\n";
            return 1;
        }
    }

    return 0;
}
