#include <iostream>
#include <vector>
#include <map>

int main() {
    std::cout << "=== For Loop Examples ===" << std::endl;

    // 1. Traditional index-based for loop
    std::cout << "\n1) Index-based for loop:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  i = " << i << std::endl;
    }

    // 2. Iterating over a vector with index
    std::vector<int> vec = {10, 20, 30, 40, 50};
    std::cout << "\n2) Vector iteration with index:" << std::endl;
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << "  vec[" << i << "] = " << vec[i] << std::endl;
    }

    // 3. Range-based for loop (C++11) - Most common in SLAM code
    std::cout << "\n3) Range-based for loop (recommended):" << std::endl;
    for (const auto& elem : vec) {
        std::cout << "  elem = " << elem << std::endl;
    }

    // 4. Range-based for loop with structured bindings (C++17)
    std::map<std::string, int> landmarks = {
        {"landmark_1", 100},
        {"landmark_2", 200},
        {"landmark_3", 300}
    };
    std::cout << "\n4) Structured bindings (C++17):" << std::endl;
    for (const auto& [key, value] : landmarks) {
        std::cout << "  " << key << " -> " << value << std::endl;
    }

    return 0;
}
