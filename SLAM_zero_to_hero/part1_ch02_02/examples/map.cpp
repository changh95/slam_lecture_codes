#include <iostream>
#include <map>
#include <string>

void print_map(const std::string& comment, const std::map<int, std::string>& m) {
    std::cout << comment;
    for (const auto& [key, value] : m) {
        std::cout << "[" << key << "]=" << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== std::map Examples ===" << std::endl;
    std::cout << "(Ordered by key - useful for keyframe storage)" << std::endl;

    // 1. Initialize map (keyframe_id -> description)
    std::map<int, std::string> keyframes = {
        {0, "initial"},
        {5, "loop_closure"},
        {10, "relocalization"}
    };
    print_map("\n1) Initial map: ", keyframes);

    // 2. Insert/update elements
    keyframes[15] = "final";
    keyframes[5] = "loop_closure_updated";  // Update existing
    print_map("2) After insert/update: ", keyframes);

    // 3. Find element
    std::cout << "3) Find operations:" << std::endl;
    auto it = keyframes.find(10);
    if (it != keyframes.end()) {
        std::cout << "  Found key 10: " << it->second << std::endl;
    }

    it = keyframes.find(100);
    if (it == keyframes.end()) {
        std::cout << "  Key 100 not found" << std::endl;
    }

    // 4. Check if key exists (C++20: contains(), or use count())
    std::cout << "4) Key existence check:" << std::endl;
    std::cout << "  keyframes.count(5) = " << keyframes.count(5) << std::endl;
    std::cout << "  keyframes.count(100) = " << keyframes.count(100) << std::endl;

    // 5. Erase element
    keyframes.erase(15);
    print_map("5) After erase(15): ", keyframes);

    // 6. Size and clear
    std::cout << "6) Size: " << keyframes.size() << std::endl;
    keyframes.clear();
    std::cout << "   After clear, empty: " << std::boolalpha << keyframes.empty() << std::endl;

    return 0;
}
