/**
 * @file marker_pose_estimation.cpp
 * @brief 6-DoF pose estimation from ArUco markers
 *
 * This demo shows how to:
 * - Estimate camera pose relative to ArUco markers
 * - Visualize coordinate axes on markers
 * - Transform poses between marker and world frames
 * - Export trajectories in TUM format for SLAM evaluation
 *
 * The marker pose estimation process:
 * 1. Detect ArUco markers in image
 * 2. Use cv::aruco::estimatePoseSingleMarkers to get marker pose
 * 3. The pose is the marker's position in the camera frame
 * 4. Optionally transform to world frame using known marker positions
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Marker pose in world frame
struct MarkerWorldPose {
    int id;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

// Estimated camera pose
struct CameraPose {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    int observedMarkerId;
    double reprojectionError;
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nOptions:\n"
              << "  --camera <id>        Camera device ID (default: 0)\n"
              << "  --image <file>       Process single image\n"
              << "  --calib <file>       Camera calibration YAML (required)\n"
              << "  --size <meters>      Marker size in meters (default: 0.05)\n"
              << "  --dict <name>        Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "  --marker-map <file>  Known marker positions (for localization)\n"
              << "  --output <file>      Output trajectory (TUM format)\n"
              << "  --no-display         Don't show GUI window\n"
              << "\nCalibration file format (YAML):\n"
              << "  camera_matrix: !!opencv-matrix { rows: 3, cols: 3, dt: d, data: [...] }\n"
              << "  distortion_coefficients: !!opencv-matrix { rows: 5, cols: 1, dt: d, data: [...] }\n"
              << "\nMarker map file format (text):\n"
              << "  # id tx ty tz qx qy qz qw\n"
              << "  0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
              << "  1 1.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
              << "\nExamples:\n"
              << "  " << programName << " --camera 0 --calib camera.yaml --size 0.1\n"
              << "  " << programName << " --image test.jpg --calib camera.yaml --output traj.txt\n";
}

// Get predefined dictionary
cv::aruco::Dictionary getDictionary(const std::string& name) {
    if (name == "4x4") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    if (name == "5x5") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    if (name == "6x6") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    if (name == "7x7") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

// Load camera calibration from YAML file
bool loadCalibration(const std::string& filename,
                     cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Cannot open calibration file: " << filename << "\n";
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        std::cerr << "Error: Invalid calibration data in " << filename << "\n";
        return false;
    }

    std::cout << "Loaded calibration from: " << filename << "\n";
    std::cout << "Camera matrix:\n" << cameraMatrix << "\n";
    std::cout << "Distortion: " << distCoeffs.t() << "\n\n";

    return true;
}

// Load marker world poses from file
bool loadMarkerMap(const std::string& filename,
                   std::map<int, MarkerWorldPose>& markerMap) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open marker map: " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int id;
        double tx, ty, tz, qx, qy, qz, qw;

        if (iss >> id >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
            MarkerWorldPose pose;
            pose.id = id;
            pose.position = Eigen::Vector3d(tx, ty, tz);
            pose.orientation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            markerMap[id] = pose;
        }
    }

    std::cout << "Loaded " << markerMap.size() << " marker world poses\n";
    return true;
}

// Convert OpenCV rotation vector and translation to Eigen
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

// Compute camera pose in world frame from marker observation
CameraPose computeCameraWorldPose(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                                   const MarkerWorldPose& markerWorld,
                                   double timestamp, int markerId) {
    CameraPose camPose;
    camPose.timestamp = timestamp;
    camPose.observedMarkerId = markerId;

    // Get marker pose in camera frame (what estimatePoseSingleMarkers gives us)
    Eigen::Matrix3d R_marker_cam;
    Eigen::Vector3d t_marker_cam;
    cvToEigen(rvec, tvec, R_marker_cam, t_marker_cam);

    // Camera pose in marker frame (inverse transform)
    Eigen::Matrix3d R_cam_marker = R_marker_cam.transpose();
    Eigen::Vector3d t_cam_marker = -R_cam_marker * t_marker_cam;

    // Transform to world frame using known marker world pose
    Eigen::Matrix3d R_marker_world = markerWorld.orientation.toRotationMatrix();
    Eigen::Matrix3d R_cam_world = R_marker_world * R_cam_marker;
    camPose.position = markerWorld.position + R_marker_world * t_cam_marker;
    camPose.orientation = Eigen::Quaterniond(R_cam_world);

    return camPose;
}

// Convert rotation matrix to Euler angles (roll, pitch, yaw)
Eigen::Vector3d rotationToEuler(const Eigen::Matrix3d& R) {
    double roll = std::atan2(R(2, 1), R(2, 2));
    double pitch = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    double yaw = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw) * 180.0 / M_PI;  // degrees
}

// Draw coordinate axes and pose info on image
void drawPoseInfo(cv::Mat& image, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                  const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                  int markerId, float markerSize, int yOffset) {
    // Draw coordinate axes
    cv::drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, markerSize * 0.6f);

    // Compute distance and Euler angles
    double distance = cv::norm(tvec);

    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = R_cv.at<double>(i, j);
        }
    }
    Eigen::Vector3d euler = rotationToEuler(R);

    // Display info
    std::stringstream ss;
    int x = 10;

    ss << "Marker " << markerId << ":";
    cv::putText(image, ss.str(), cv::Point(x, yOffset),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    ss.str(""); ss.clear();
    ss << std::fixed << std::setprecision(3)
       << "  t: [" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << "] m";
    cv::putText(image, ss.str(), cv::Point(x, yOffset + 18),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    ss.str(""); ss.clear();
    ss << std::fixed << std::setprecision(1)
       << "  rpy: [" << euler.x() << ", " << euler.y() << ", " << euler.z() << "] deg";
    cv::putText(image, ss.str(), cv::Point(x, yOffset + 36),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    ss.str(""); ss.clear();
    ss << std::fixed << std::setprecision(3) << "  dist: " << distance << " m";
    cv::putText(image, ss.str(), cv::Point(x, yOffset + 54),
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}

// Trajectory writer in TUM format
class TrajectoryWriter {
public:
    explicit TrajectoryWriter(const std::string& filename) : file_(filename) {
        if (file_.is_open()) {
            file_ << "# Ground truth trajectory from ArUco markers\n";
            file_ << "# timestamp tx ty tz qx qy qz qw\n";
            file_ << std::fixed << std::setprecision(6);
            std::cout << "Writing trajectory to: " << filename << "\n";
        }
    }

    void write(const CameraPose& pose) {
        if (!file_.is_open()) return;

        file_ << pose.timestamp << " "
              << pose.position.x() << " "
              << pose.position.y() << " "
              << pose.position.z() << " "
              << pose.orientation.x() << " "
              << pose.orientation.y() << " "
              << pose.orientation.z() << " "
              << pose.orientation.w() << "\n";
        poseCount_++;
    }

    size_t count() const { return poseCount_; }
    bool isOpen() const { return file_.is_open(); }

private:
    std::ofstream file_;
    size_t poseCount_ = 0;
};

// Main pose estimation function
void estimatePoses(cv::Mat& image, cv::aruco::ArucoDetector& detector,
                   const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                   float markerSize, const std::map<int, MarkerWorldPose>& markerMap,
                   TrajectoryWriter* trajWriter, double timestamp) {
    // Detect markers
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(image, corners, ids);

    if (ids.empty()) {
        cv::putText(image, "No markers detected", cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        return;
    }

    // Draw detected markers
    cv::aruco::drawDetectedMarkers(image, corners, ids);

    // Estimate pose for each marker
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, markerSize,
                                          cameraMatrix, distCoeffs, rvecs, tvecs);

    // Process each detected marker
    int yOffset = 60;
    for (size_t i = 0; i < ids.size(); i++) {
        int markerId = ids[i];

        // Draw axes and info
        drawPoseInfo(image, cameraMatrix, distCoeffs,
                     rvecs[i], tvecs[i], markerId, markerSize, yOffset);
        yOffset += 80;

        // If marker world position is known, compute camera world pose
        auto it = markerMap.find(markerId);
        if (it != markerMap.end() && trajWriter != nullptr) {
            CameraPose camPose = computeCameraWorldPose(
                rvecs[i], tvecs[i], it->second, timestamp, markerId);
            trajWriter->write(camPose);

            // Display camera world position
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3)
               << "Camera world pos: [" << camPose.position.x() << ", "
               << camPose.position.y() << ", " << camPose.position.z() << "]";
            cv::putText(image, ss.str(), cv::Point(10, image.rows - 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Display marker count
    std::stringstream info;
    info << "Markers: " << ids.size();
    cv::putText(image, info.str(), cv::Point(10, 30),
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

    // Parse arguments
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
        } else if (arg == "--size" && i + 1 < argc) {
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

    std::cout << "=== Marker Pose Estimation Demo ===\n\n";

    // Load camera calibration
    cv::Mat cameraMatrix, distCoeffs;
    if (calibFile.empty()) {
        std::cerr << "Error: Camera calibration file required (--calib)\n";
        printUsage(argv[0]);
        return 1;
    }
    if (!loadCalibration(calibFile, cameraMatrix, distCoeffs)) {
        return 1;
    }

    // Load marker world poses
    std::map<int, MarkerWorldPose> markerMap;
    if (!markerMapFile.empty()) {
        loadMarkerMap(markerMapFile, markerMap);
    }

    // Setup trajectory writer
    std::unique_ptr<TrajectoryWriter> trajWriter;
    if (!outputFile.empty()) {
        trajWriter = std::make_unique<TrajectoryWriter>(outputFile);
    }

    // Setup detector
    cv::aruco::Dictionary dictionary = getDictionary(dictName);
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(dictionary, params);

    std::cout << "Marker size: " << markerSize << " m\n";
    std::cout << "Dictionary: " << dictName << "\n\n";

    // Process image or camera
    if (!useCamera) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error: Cannot load image: " << imagePath << "\n";
            return 1;
        }

        estimatePoses(image, detector, cameraMatrix, distCoeffs, markerSize,
                      markerMap, trajWriter.get(), 0.0);

        if (display) {
            cv::imshow("Marker Pose Estimation", image);
            std::cout << "Press any key to exit...\n";
            cv::waitKey(0);
        }

        // Save result if output specified
        if (!outputFile.empty()) {
            std::string imageOutput = outputFile + ".png";
            cv::imwrite(imageOutput, image);
            std::cout << "Saved visualization to: " << imageOutput << "\n";
        }
    } else {
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera " << cameraId << "\n";
            return 1;
        }

        std::cout << "Camera opened. Press 'q' to quit.\n\n";

        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            auto now = std::chrono::steady_clock::now();
            double timestamp = std::chrono::duration<double>(now - startTime).count();

            estimatePoses(frame, detector, cameraMatrix, distCoeffs, markerSize,
                          markerMap, trajWriter.get(), timestamp);

            if (display) {
                cv::imshow("Marker Pose Estimation", frame);
            }

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
        }

        cap.release();
        cv::destroyAllWindows();
    }

    // Print summary
    if (trajWriter && trajWriter->count() > 0) {
        std::cout << "\nWrote " << trajWriter->count() << " poses to trajectory file.\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
