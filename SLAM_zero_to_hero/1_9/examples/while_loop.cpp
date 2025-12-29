#include <iostream>

int main() {
    std::cout << "=== While Loop Examples ===" << std::endl;

    // 1. Basic while loop
    std::cout << "\n1) Basic while loop:" << std::endl;
    int count = 0;
    while (count < 5) {
        std::cout << "  count = " << count << std::endl;
        count++;
    }

    // 2. Do-while loop (executes at least once)
    std::cout << "\n2) Do-while loop:" << std::endl;
    int num = 0;
    do {
        std::cout << "  num = " << num << std::endl;
        num++;
    } while (num < 3);

    // 3. While loop with break (common in SLAM main loops)
    std::cout << "\n3) While loop with break (SLAM-style main loop):" << std::endl;
    int frame_id = 0;
    while (true) {
        std::cout << "  Processing frame " << frame_id << std::endl;
        frame_id++;
        if (frame_id >= 5) {
            std::cout << "  Reached max frames, breaking..." << std::endl;
            break;
        }
    }

    return 0;
}
