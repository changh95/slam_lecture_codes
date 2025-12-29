/**
 * @file pose_estimator.cpp
 * @brief Estimate 6-DOF poses from ArUco markers for robot localization
 *
 * This program performs marker detection and pose estimation, providing
 * ground truth trajectories for SLAM evaluation. Supports both single
 * markers and marker boards for improved accuracy.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Marker pose in world frame (for known marker positions)
struct MarkerWorldPose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  --camera <id>        Camera device ID (default: 0)\n"
              << "  --image <file>       Process single image instead of camera\n"
              << "  --calib <file>       Camera calibration YAML file (required)\n"
              << "  --marker-size <m>    Marker size in meters (default: 0.05)\n"
              << "  --dict <name>        Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "  --marker-map <file>  Marker world positions (for robot localization)\n"
              << "  --output <file>      Output trajectory file (TUM format)\n"
              << "  --no-display         Don't show GUI window\n"
              << "\nExamples:\n"
              << "  " << programName << " --camera 0 --calib camera.yaml --marker-size 0.1\n"
              << "  " << programName << " --image test.jpg --calib camera.yaml\n";
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

bool loadCameraCalibration(const std::string& filename, cv::Mat& cameraMatrix,
                           cv::Mat& distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open calibration file: " << filename << "\n";
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        std::cerr << "Error: Invalid calibration data in " << filename << "\n";
        return false;
    }

    std::cout << "Loaded camera calibration from: " << filename << "\n";
    std::cout << "Camera matrix:\n" << cameraMatrix << "\n";
    std::cout << "Distortion coefficients: " << distCoeffs << "\n\n";

    return true;
}

bool loadMarkerMap(const std::string& filename,
                   std::map<int, MarkerWorldPose>& markerMap) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open marker map file: " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int id;
        double x, y, z, qx, qy, qz, qw;

        if (iss >> id >> x >> y >> z >> qx >> qy >> qz >> qw) {
            MarkerWorldPose pose;
            pose.position = Eigen::Vector3d(x, y, z);
            pose.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
            pose.orientation.normalize();
            markerMap[id] = pose;
        }
    }

    std::cout << "Loaded " << markerMap.size() << " marker world poses\n";
    return true;
}

// Convert OpenCV rotation vector and translation to Eigen pose
void cvToEigen(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
               Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = R_cv.at<double>(i, j);
        }
        t(i) = tvec[i];
    }
}

// Compute camera pose in world frame given marker pose
void computeCameraPoseInWorld(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                               const MarkerWorldPose& markerWorld,
                               Eigen::Vector3d& camPosition,
                               Eigen::Quaterniond& camOrientation) {
    // Get marker pose in camera frame
    Eigen::Matrix3d R_marker_cam;
    Eigen::Vector3d t_marker_cam;
    cvToEigen(rvec, tvec, R_marker_cam, t_marker_cam);

    // Camera pose in marker frame
    Eigen::Matrix3d R_cam_marker = R_marker_cam.transpose();
    Eigen::Vector3d t_cam_marker = -R_cam_marker * t_marker_cam;

    // Transform to world frame
    Eigen::Matrix3d R_marker_world = markerWorld.orientation.toRotationMatrix();
    Eigen::Matrix3d R_cam_world = R_marker_world * R_cam_marker;
    camPosition = markerWorld.position + R_marker_world * t_cam_marker;
    camOrientation = Eigen::Quaterniond(R_cam_world);
}

void drawAxisAndInfo(cv::Mat& image, const cv::Mat& cameraMatrix,
                     const cv::Mat& distCoeffs, const cv::Vec3d& rvec,
                     const cv::Vec3d& tvec, int markerId, float markerSize) {
    // Draw coordinate axes
    cv::drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize * 0.5f);

    // Compute distance and angles
    double distance = cv::norm(tvec);
    double rx = std::atan2(tvec[1], tvec[2]) * 180.0 / M_PI;
    double ry = std::atan2(tvec[0], tvec[2]) * 180.0 / M_PI;

    // Convert rotation vector to Euler angles
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    double roll = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)) * 180.0 / M_PI;
    double pitch = std::atan2(-R.at<double>(2, 0),
                              std::sqrt(R.at<double>(2, 1) * R.at<double>(2, 1) +
                                       R.at<double>(2, 2) * R.at<double>(2, 2))) * 180.0 / M_PI;
    double yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180.0 / M_PI;

    // Display pose information
    int yOffset = 60 + markerId * 80;
    std::stringstream ss;

    ss << "Marker " << markerId << ":";
    cv::putText(image, ss.str(), cv::Point(10, yOffset),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    ss.str(""); ss.clear();
    ss << std::fixed << std::setprecision(3)
       << "  t: [" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << "]";
    cv::putText(image, ss.str(), cv::Point(10, yOffset + 15),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    ss.str(""); ss.clear();
    ss << std::fixed << std::setprecision(1)
       << "  rpy: [" << roll << ", " << pitch << ", " << yaw << "] deg";
    cv::putText(image, ss.str(), cv::Point(10, yOffset + 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    ss.str(""); ss.clear();
    ss << std::fixed << std::setprecision(3) << "  dist: " << distance << " m";
    cv::putText(image, ss.str(), cv::Point(10, yOffset + 45),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}

class TrajectoryWriter {
public:
    TrajectoryWriter(const std::string& filename) : file_(filename) {
        if (file_.is_open()) {
            file_ << "# Ground truth trajectory from ArUco markers\n";
            file_ << "# timestamp tx ty tz qx qy qz qw\n";
            file_ << std::fixed << std::setprecision(6);
        }
    }

    void writePose(double timestamp, const Eigen::Vector3d& position,
                   const Eigen::Quaterniond& orientation) {
        if (file_.is_open()) {
            file_ << timestamp << " "
                  << position.x() << " " << position.y() << " " << position.z() << " "
                  << orientation.x() << " " << orientation.y() << " "
                  << orientation.z() << " " << orientation.w() << "\n";
        }
    }

    bool isOpen() const { return file_.is_open(); }

private:
    std::ofstream file_;
};

void processFrame(cv::Mat& image, cv::aruco::ArucoDetector& detector,
                  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                  float markerSize, const std::map<int, MarkerWorldPose>& markerMap,
                  TrajectoryWriter* trajWriter, double timestamp) {

    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<int> markerIds;

    // Detect markers
    detector.detectMarkers(image, markerCorners, markerIds);

    if (!markerIds.empty()) {
        // Draw detected markers
        cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

        // Estimate pose for each marker
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners, markerSize,
                                             cameraMatrix, distCoeffs, rvecs, tvecs);

        // Process each detected marker
        for (size_t i = 0; i < markerIds.size(); i++) {
            int id = markerIds[i];

            // Draw axes and info
            drawAxisAndInfo(image, cameraMatrix, distCoeffs,
                           rvecs[i], tvecs[i], id, markerSize);

            // If marker world position is known, compute camera pose
            auto it = markerMap.find(id);
            if (it != markerMap.end() && trajWriter != nullptr) {
                Eigen::Vector3d camPosition;
                Eigen::Quaterniond camOrientation;
                computeCameraPoseInWorld(rvecs[i], tvecs[i], it->second,
                                        camPosition, camOrientation);
                trajWriter->writePose(timestamp, camPosition, camOrientation);

                // Display camera world position
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3)
                   << "Camera: [" << camPosition.x() << ", "
                   << camPosition.y() << ", " << camPosition.z() << "]";
                cv::putText(image, ss.str(), cv::Point(10, image.rows - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    // Display frame info
    std::string info = "Markers: " + std::to_string(markerIds.size());
    cv::putText(image, info, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

int main(int argc, char** argv) {
    // Default parameters
    int cameraId = 0;
    std::string imagePath;
    std::string calibFile;
    std::string markerMapFile;
    std::string outputFile;
    std::string dictName = "6x6";
    float markerSize = 0.05f;
    bool display = true;
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
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
            useCamera = false;
        } else if (arg == "--calib" && i + 1 < argc) {
            calibFile = argv[++i];
        } else if (arg == "--marker-size" && i + 1 < argc) {
            markerSize = std::stof(argv[++i]);
        } else if (arg == "--dict" && i + 1 < argc) {
            dictName = argv[++i];
        } else if (arg == "--marker-map" && i + 1 < argc) {
            markerMapFile = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--no-display") {
            display = false;
        }
    }

    std::cout << "=== ArUco Pose Estimator ===\n\n";

    // Load camera calibration
    cv::Mat cameraMatrix, distCoeffs;
    if (calibFile.empty()) {
        std::cerr << "Error: Camera calibration file required (--calib)\n";
        std::cerr << "Generate one using charuco_calibration first.\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!loadCameraCalibration(calibFile, cameraMatrix, distCoeffs)) {
        return 1;
    }

    // Load marker world positions if available
    std::map<int, MarkerWorldPose> markerMap;
    if (!markerMapFile.empty()) {
        loadMarkerMap(markerMapFile, markerMap);
    }

    // Set up trajectory writer
    std::unique_ptr<TrajectoryWriter> trajWriter;
    if (!outputFile.empty()) {
        trajWriter = std::make_unique<TrajectoryWriter>(outputFile);
        if (trajWriter->isOpen()) {
            std::cout << "Writing trajectory to: " << outputFile << "\n";
        }
    }

    // Set up detector
    cv::aruco::Dictionary dictionary = getDictionary(dictName);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(dictionary, params);

    std::cout << "Marker size: " << markerSize << " m\n";
    std::cout << "Dictionary: " << dictName << "\n\n";

    if (!useCamera) {
        // Process single image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << imagePath << "\n";
            return 1;
        }

        processFrame(image, detector, cameraMatrix, distCoeffs, markerSize,
                    markerMap, trajWriter.get(), 0.0);

        if (display) {
            cv::imshow("Pose Estimation", image);
            cv::waitKey(0);
        }
    } else {
        // Process camera feed
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << cameraId << "\n";
            return 1;
        }

        std::cout << "Camera opened. Press 'q' to quit.\n";

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            // Compute timestamp
            auto now = std::chrono::steady_clock::now();
            double timestamp = std::chrono::duration<double>(now - startTime).count();

            processFrame(frame, detector, cameraMatrix, distCoeffs, markerSize,
                        markerMap, trajWriter.get(), timestamp);

            if (display) {
                cv::imshow("Pose Estimation", frame);
            }

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
        }

        cap.release();
        cv::destroyAllWindows();
    }

    std::cout << "\nDone.\n";
    return 0;
}
