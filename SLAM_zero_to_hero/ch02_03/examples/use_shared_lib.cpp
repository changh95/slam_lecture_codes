#include <iostream>
#include <vector>
#include "mathlib/mathlib.hpp"

int main() {
    std::cout << "=== Using Shared Library (mathlib) ===" << std::endl;
    std::cout << "Shared libraries are loaded at runtime" << std::endl;
    std::cout << "Multiple programs can share the same library\n" << std::endl;

    // Same code as static library - the difference is in linking!
    std::vector<double> descriptor1 = {0.5, 0.3, 0.8, 0.2};
    std::vector<double> descriptor2 = {0.4, 0.4, 0.7, 0.3};

    // Feature descriptor operations (common in SLAM)
    std::cout << "Feature Descriptor Operations:" << std::endl;

    // Compute norms
    double norm1 = mathlib::computeNorm(descriptor1);
    double norm2 = mathlib::computeNorm(descriptor2);
    std::cout << "  descriptor1 norm: " << norm1 << std::endl;
    std::cout << "  descriptor2 norm: " << norm2 << std::endl;

    // Compute similarity (dot product of normalized vectors)
    mathlib::normalize(descriptor1);
    mathlib::normalize(descriptor2);
    double similarity = mathlib::dotProduct(descriptor1, descriptor2);
    std::cout << "  Cosine similarity: " << similarity << std::endl;

    // Distance between 3D points (map points)
    std::vector<double> mapPoint1 = {10.5, -3.2, 25.0};
    std::vector<double> mapPoint2 = {11.0, -2.8, 24.5};
    double dist = mathlib::euclideanDistance(mapPoint1, mapPoint2);
    std::cout << "\nMap point distance: " << dist << " meters" << std::endl;

    std::cout << "\n--- Shared Library Benefits ---" << std::endl;
    std::cout << "- Smaller executable size" << std::endl;
    std::cout << "- Can update library without recompiling" << std::endl;
    std::cout << "- Memory efficient (shared across processes)" << std::endl;
    std::cout << "- OpenCV, PCL, ROS use shared libraries" << std::endl;

    return 0;
}
