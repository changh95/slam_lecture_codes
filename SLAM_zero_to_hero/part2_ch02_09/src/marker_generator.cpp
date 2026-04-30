/**
 * @file marker_generator.cpp
 * @brief Generate ArUco markers and ChArUco calibration boards
 *
 * This utility generates fiducial markers for robot localization and
 * camera calibration. Supports both single ArUco markers and ChArUco boards.
 */

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]\n"
              << "\nGenerate ArUco marker:\n"
              << "  --id <n>           Marker ID (default: 0)\n"
              << "  --size <pixels>    Marker size in pixels (default: 200)\n"
              << "  --dict <name>      Dictionary: 4x4, 5x5, 6x6, 7x7 (default: 6x6)\n"
              << "  --output <file>    Output filename (default: marker.png)\n"
              << "\nGenerate ChArUco board:\n"
              << "  --charuco          Generate ChArUco board instead of single marker\n"
              << "  --cols <n>         Number of columns (default: 5)\n"
              << "  --rows <n>         Number of rows (default: 7)\n"
              << "  --square <pixels>  Square size in pixels (default: 100)\n"
              << "\nExamples:\n"
              << "  " << programName << " --id 42 --size 300 --output marker_42.png\n"
              << "  " << programName << " --charuco --cols 5 --rows 7 --output board.png\n";
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

void generateSingleMarker(int id, int size, const std::string& dictName,
                          const std::string& output) {
    cv::aruco::Dictionary dictionary = getDictionary(dictName);

    // Check if ID is valid for this dictionary
    int maxId = static_cast<int>(dictionary.bytesList.rows);
    if (id < 0 || id >= maxId) {
        std::cerr << "Error: Marker ID must be in range [0, " << maxId - 1 << "]\n";
        return;
    }

    cv::Mat markerImage;
    cv::aruco::generateImageMarker(dictionary, id, size, markerImage, 1);

    // Add white border for printing
    int borderSize = size / 10;
    cv::Mat output_image;
    cv::copyMakeBorder(markerImage, output_image, borderSize, borderSize,
                       borderSize, borderSize, cv::BORDER_CONSTANT,
                       cv::Scalar(255, 255, 255));

    if (cv::imwrite(output, output_image)) {
        std::cout << "Generated ArUco marker:\n"
                  << "  ID: " << id << "\n"
                  << "  Dictionary: " << dictName << "\n"
                  << "  Size: " << size << " pixels\n"
                  << "  Output: " << output << "\n";
    } else {
        std::cerr << "Error: Failed to write image to " << output << "\n";
    }
}

void generateCharucoBoard(int cols, int rows, int squareSize,
                          const std::string& dictName, const std::string& output) {
    cv::aruco::Dictionary dictionary = getDictionary(dictName);

    // Marker size is 80% of square size
    int markerSize = static_cast<int>(squareSize * 0.8);

    // Create ChArUco board
    cv::aruco::CharucoBoard board(
        cv::Size(cols, rows),
        static_cast<float>(squareSize),
        static_cast<float>(markerSize),
        dictionary
    );

    // Generate board image
    cv::Size imageSize(cols * squareSize + squareSize,
                       rows * squareSize + squareSize);
    cv::Mat boardImage;
    board.generateImage(imageSize, boardImage, squareSize / 2, 1);

    if (cv::imwrite(output, boardImage)) {
        std::cout << "Generated ChArUco board:\n"
                  << "  Columns: " << cols << "\n"
                  << "  Rows: " << rows << "\n"
                  << "  Square size: " << squareSize << " pixels\n"
                  << "  Marker size: " << markerSize << " pixels\n"
                  << "  Dictionary: " << dictName << "\n"
                  << "  Output: " << output << "\n"
                  << "\nFor calibration, measure the actual printed sizes:\n"
                  << "  - Square size in meters\n"
                  << "  - Marker size in meters\n";
    } else {
        std::cerr << "Error: Failed to write image to " << output << "\n";
    }
}

int main(int argc, char** argv) {
    // Default parameters
    int markerId = 0;
    int markerSize = 200;
    int boardCols = 5;
    int boardRows = 7;
    int squareSize = 100;
    std::string dictName = "6x6";
    std::string outputFile = "marker.png";
    bool generateBoard = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--id" && i + 1 < argc) {
            markerId = std::stoi(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            markerSize = std::stoi(argv[++i]);
        } else if (arg == "--dict" && i + 1 < argc) {
            dictName = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--charuco") {
            generateBoard = true;
        } else if (arg == "--cols" && i + 1 < argc) {
            boardCols = std::stoi(argv[++i]);
        } else if (arg == "--rows" && i + 1 < argc) {
            boardRows = std::stoi(argv[++i]);
        } else if (arg == "--square" && i + 1 < argc) {
            squareSize = std::stoi(argv[++i]);
        }
    }

    std::cout << "=== ArUco Marker Generator ===\n\n";

    if (generateBoard) {
        generateCharucoBoard(boardCols, boardRows, squareSize, dictName, outputFile);
    } else {
        generateSingleMarker(markerId, markerSize, dictName, outputFile);
    }

    return 0;
}
