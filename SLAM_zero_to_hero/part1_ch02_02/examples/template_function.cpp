#include <iostream>
#include <vector>
#include <cmath>

// 1. Basic template function
template <typename T>
T findMax(T a, T b) {
    return (a > b) ? a : b;
}

// 2. Template function with multiple types
template <typename T1, typename T2>
auto add(T1 a, T2 b) -> decltype(a + b) {
    return a + b;
}

// 3. Template function for computing L2 norm (SLAM-relevant)
template <typename T>
T computeL2Norm(const std::vector<T>& vec) {
    T sum = 0;
    for (const auto& elem : vec) {
        sum += elem * elem;
    }
    return std::sqrt(sum);
}

// 4. Template function with default type
template <typename T = double>
T computeMean(const std::vector<T>& vec) {
    if (vec.empty()) return T{0};
    T sum = 0;
    for (const auto& elem : vec) {
        sum += elem;
    }
    return sum / static_cast<T>(vec.size());
}

int main() {
    std::cout << "=== Template Function Examples ===" << std::endl;

    // 1. Basic template
    std::cout << "\n1) findMax template:" << std::endl;
    std::cout << "  findMax(3, 7) = " << findMax(3, 7) << std::endl;
    std::cout << "  findMax(3.14, 2.71) = " << findMax(3.14, 2.71) << std::endl;
    std::cout << "  findMax('a', 'z') = " << findMax('a', 'z') << std::endl;

    // 2. Multiple type template
    std::cout << "\n2) add template (mixed types):" << std::endl;
    std::cout << "  add(5, 3.14) = " << add(5, 3.14) << std::endl;
    std::cout << "  add(10, 20) = " << add(10, 20) << std::endl;

    // 3. L2 norm (commonly used in SLAM for descriptor distance)
    std::cout << "\n3) L2 norm computation:" << std::endl;
    std::vector<double> descriptor = {0.5, 0.3, 0.8, 0.1};
    std::cout << "  L2 norm of descriptor = " << computeL2Norm(descriptor) << std::endl;

    std::vector<float> point3d = {1.0f, 2.0f, 3.0f};
    std::cout << "  L2 norm of 3D point = " << computeL2Norm(point3d) << std::endl;

    // 4. Mean computation
    std::cout << "\n4) Mean computation:" << std::endl;
    std::vector<double> depths = {1.5, 2.3, 1.8, 2.1, 1.9};
    std::cout << "  Mean depth = " << computeMean(depths) << std::endl;

    return 0;
}
