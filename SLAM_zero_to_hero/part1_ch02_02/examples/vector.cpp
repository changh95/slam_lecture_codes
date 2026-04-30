#include <iostream>
#include <vector>

int main() {
    std::cout << "=== std::vector Examples ===" << std::endl;

    // 1. Initialize vector
    std::vector<int> keypoint_ids = {1, 2, 3, 4, 5};
    std::cout << "\n1) Initial vector: ";
    for (const auto& id : keypoint_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // 2. push_back - Add element to end
    keypoint_ids.push_back(6);
    keypoint_ids.push_back(7);
    std::cout << "\n2) After push_back(6, 7): ";
    for (const auto& id : keypoint_ids) {
        std::cout << id << " ";
    }
    std::cout << "(size: " << keypoint_ids.size() << ")" << std::endl;

    // 3. pop_back - Remove last element
    keypoint_ids.pop_back();
    std::cout << "\n3) After pop_back(): ";
    for (const auto& id : keypoint_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // 4. Access elements
    std::cout << "\n4) Element access:" << std::endl;
    std::cout << "  keypoint_ids[0] = " << keypoint_ids[0] << std::endl;
    std::cout << "  keypoint_ids.front() = " << keypoint_ids.front() << std::endl;
    std::cout << "  keypoint_ids.back() = " << keypoint_ids.back() << std::endl;

    // 5. Reserve vs Resize (important for performance in SLAM)
    std::vector<double> descriptors;
    descriptors.reserve(1000);  // Allocates memory, size still 0
    std::cout << "\n5) After reserve(1000): size=" << descriptors.size()
              << ", capacity=" << descriptors.capacity() << std::endl;

    // 6. Clear vector
    keypoint_ids.clear();
    std::cout << "\n6) After clear(): size=" << keypoint_ids.size()
              << ", empty=" << std::boolalpha << keypoint_ids.empty() << std::endl;

    return 0;
}
