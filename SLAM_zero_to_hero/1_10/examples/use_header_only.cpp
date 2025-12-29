#include <iostream>
#include <cmath>
#include "geometry/geometry.hpp"

int main() {
    std::cout << "=== Using Header-Only Library (geometry) ===" << std::endl;
    std::cout << "Header-only: just #include, no linking needed" << std::endl;
    std::cout << "Examples: Eigen, Sophus, nlohmann/json\n" << std::endl;

    // Create 3D points
    geometry::Point3d camera_pos(0.0, 0.0, 0.0);
    geometry::Point3d landmark1(5.0, 2.0, 10.0);
    geometry::Point3d landmark2(3.0, -1.0, 8.0);

    std::cout << "Camera position: (0, 0, 0)" << std::endl;
    std::cout << "Landmark 1: (5, 2, 10)" << std::endl;
    std::cout << "Landmark 2: (3, -1, 8)\n" << std::endl;

    // Vector operations
    std::cout << "1) Vector from camera to landmark1:" << std::endl;
    auto ray1 = landmark1 - camera_pos;
    std::cout << "   ray1 = (" << ray1.x << ", " << ray1.y << ", " << ray1.z << ")" << std::endl;
    std::cout << "   distance = " << ray1.norm() << std::endl;

    // Normalized direction
    std::cout << "\n2) Normalized direction:" << std::endl;
    auto dir1 = ray1.normalized();
    std::cout << "   dir1 = (" << dir1.x << ", " << dir1.y << ", " << dir1.z << ")" << std::endl;
    std::cout << "   norm = " << dir1.norm() << " (should be 1)" << std::endl;

    // Dot product (for angle computation)
    std::cout << "\n3) Angle between rays:" << std::endl;
    auto ray2 = landmark2 - camera_pos;
    double angle_rad = geometry::angleBetween(ray1, ray2);
    double angle_deg = angle_rad * 180.0 / M_PI;
    std::cout << "   Angle = " << angle_deg << " degrees" << std::endl;

    // Cross product (for normal computation)
    std::cout << "\n4) Cross product (plane normal):" << std::endl;
    auto normal = ray1.cross(ray2);
    auto normal_unit = normal.normalized();
    std::cout << "   normal = (" << normal_unit.x << ", " << normal_unit.y << ", " << normal_unit.z << ")" << std::endl;

    // Using float version
    std::cout << "\n5) Using Point3f (float precision):" << std::endl;
    geometry::Point3f pixel(320.5f, 240.3f, 1.0f);
    std::cout << "   Pixel: (" << pixel.x << ", " << pixel.y << ", " << pixel.z << ")" << std::endl;

    std::cout << "\n--- Header-Only Library Benefits ---" << std::endl;
    std::cout << "- No compilation needed, just include" << std::endl;
    std::cout << "- Templates enable compile-time optimization" << std::endl;
    std::cout << "- Easy to distribute (just headers)" << std::endl;
    std::cout << "- Eigen is header-only for performance!" << std::endl;

    return 0;
}
