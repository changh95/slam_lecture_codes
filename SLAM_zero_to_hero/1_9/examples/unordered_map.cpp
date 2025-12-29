#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::cout << "=== std::unordered_map Examples ===" << std::endl;
    std::cout << "(Hash-based - O(1) average lookup, great for feature matching)" << std::endl;

    // 1. Initialize unordered_map (feature_id -> descriptor_index)
    std::unordered_map<int, int> feature_to_descriptor = {
        {100, 0},
        {250, 1},
        {375, 2}
    };

    std::cout << "\n1) Initial unordered_map:" << std::endl;
    for (const auto& [feature_id, desc_idx] : feature_to_descriptor) {
        std::cout << "  feature " << feature_id << " -> descriptor " << desc_idx << std::endl;
    }

    // 2. Insert elements
    feature_to_descriptor[500] = 3;
    feature_to_descriptor.insert({600, 4});
    std::cout << "\n2) After insertions, size: " << feature_to_descriptor.size() << std::endl;

    // 3. Fast lookup (O(1) average)
    std::cout << "\n3) Fast lookup:" << std::endl;
    if (feature_to_descriptor.count(250)) {
        std::cout << "  Feature 250 maps to descriptor " << feature_to_descriptor[250] << std::endl;
    }

    // 4. Difference from std::map: unordered (no sorting)
    std::cout << "\n4) Iteration order (not sorted!):" << std::endl;
    for (const auto& [key, value] : feature_to_descriptor) {
        std::cout << "  " << key << " -> " << value << std::endl;
    }

    // 5. Performance comparison note
    std::cout << "\n5) When to use:" << std::endl;
    std::cout << "  - std::map: Need ordered keys, O(log n) operations" << std::endl;
    std::cout << "  - std::unordered_map: Fast lookup needed, O(1) average" << std::endl;
    std::cout << "  - SLAM example: Feature matching uses unordered_map for speed" << std::endl;

    return 0;
}
