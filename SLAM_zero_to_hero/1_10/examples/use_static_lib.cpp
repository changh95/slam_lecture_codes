#include <iostream>
#include <vector>
#include "mathlib/mathlib.hpp"

int main() {
    std::cout << "=== Using Static Library (mathlib) ===" << std::endl;
    std::cout << "Static libraries are linked at compile time" << std::endl;
    std::cout << "The library code is embedded into the executable\n" << std::endl;

    // Example: Using mathlib functions
    std::vector<double> point1 = {1.0, 2.0, 3.0};
    std::vector<double> point2 = {4.0, 5.0, 6.0};

    // Compute norm
    std::cout << "1) Compute norm:" << std::endl;
    std::cout << "   point1 = [1, 2, 3]" << std::endl;
    std::cout << "   norm(point1) = " << mathlib::computeNorm(point1) << std::endl;

    // Compute dot product
    std::cout << "\n2) Dot product:" << std::endl;
    std::cout << "   point2 = [4, 5, 6]" << std::endl;
    std::cout << "   dot(point1, point2) = " << mathlib::dotProduct(point1, point2) << std::endl;

    // Normalize
    std::cout << "\n3) Normalize:" << std::endl;
    std::vector<double> vec = {3.0, 4.0};
    std::cout << "   Before: [" << vec[0] << ", " << vec[1] << "]" << std::endl;
    mathlib::normalize(vec);
    std::cout << "   After:  [" << vec[0] << ", " << vec[1] << "]" << std::endl;
    std::cout << "   New norm: " << mathlib::computeNorm(vec) << std::endl;

    // Euclidean distance
    std::cout << "\n4) Euclidean distance:" << std::endl;
    std::cout << "   distance(point1, point2) = " << mathlib::euclideanDistance(point1, point2) << std::endl;

    std::cout << "\n--- Static Library Benefits ---" << std::endl;
    std::cout << "- No runtime dependencies" << std::endl;
    std::cout << "- Slightly faster (no dynamic loading)" << std::endl;
    std::cout << "- Larger executable size" << std::endl;

    return 0;
}
