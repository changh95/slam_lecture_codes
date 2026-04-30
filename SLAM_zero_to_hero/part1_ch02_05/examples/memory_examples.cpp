#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <thread>
#include <chrono>

// Demonstration of various memory patterns for profiling with Valgrind/Massif

//=============================================================================
// Example 1: Gradual Memory Buildup (Common in SLAM map accumulation)
//=============================================================================
class MapPoint {
public:
    double x, y, z;
    std::vector<float> descriptor;  // 256-dim descriptor
    int observations;

    MapPoint(double x_, double y_, double z_) : x(x_), y(y_), z(z_), observations(0) {
        descriptor.resize(256);  // Allocate descriptor memory
        for (int i = 0; i < 256; i++) {
            descriptor[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
};

void simulateMapGrowth() {
    std::cout << "\n=== Simulating SLAM Map Growth ===" << std::endl;
    std::vector<std::unique_ptr<MapPoint>> map_points;

    // Simulate adding map points over time (like in visual SLAM)
    for (int frame = 0; frame < 100; frame++) {
        // Each frame adds 50-100 new map points
        int new_points = 50 + (rand() % 50);

        for (int i = 0; i < new_points; i++) {
            double x = static_cast<double>(rand()) / RAND_MAX * 100.0;
            double y = static_cast<double>(rand()) / RAND_MAX * 100.0;
            double z = static_cast<double>(rand()) / RAND_MAX * 10.0;
            map_points.push_back(std::make_unique<MapPoint>(x, y, z));
        }

        if (frame % 20 == 0) {
            std::cout << "Frame " << frame << ": " << map_points.size()
                      << " map points (~" << map_points.size() * sizeof(MapPoint) / 1024
                      << " KB metadata)" << std::endl;
        }

        // Small delay to see memory growth in profiler
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Final map size: " << map_points.size() << " points" << std::endl;
    // Memory is properly freed when map_points goes out of scope
}

//=============================================================================
// Example 2: Memory Leak (BAD PRACTICE - for demonstration only!)
//=============================================================================
class SensorData {
public:
    double* measurements;
    int size;

    SensorData(int n) : size(n) {
        measurements = new double[n];
        for (int i = 0; i < n; i++) {
            measurements[i] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    // BUG: Missing destructor causes memory leak!
    // ~SensorData() { delete[] measurements; }  // This line is intentionally commented out
};

void demonstrateMemoryLeak() {
    std::cout << "\n=== Demonstrating Memory Leak (Intentional Bug) ===" << std::endl;

    // Simulate processing sensor data without proper cleanup
    for (int i = 0; i < 1000; i++) {
        // LEAK: Raw pointer without delete
        SensorData* data = new SensorData(1000);

        // Process data...
        double sum = 0;
        for (int j = 0; j < data->size; j++) {
            sum += data->measurements[j];
        }

        // BUG: Forgot to delete data!
        // delete data;  // This line is intentionally commented out

        if (i % 200 == 0) {
            std::cout << "Processed batch " << i << " (leaking ~"
                      << (i + 1) * sizeof(SensorData) / 1024 << " KB)" << std::endl;
        }
    }

    std::cout << "Finished processing - memory was NOT freed!" << std::endl;
}

//=============================================================================
// Example 3: Proper Memory Management with Smart Pointers
//=============================================================================
class ImageBuffer {
public:
    std::vector<uint8_t> data;
    int width, height;

    ImageBuffer(int w, int h) : width(w), height(h) {
        data.resize(w * h * 3);  // RGB image
        std::cout << "  Allocated image buffer: " << w << "x" << h
                  << " (" << data.size() / 1024 << " KB)" << std::endl;
    }

    ~ImageBuffer() {
        std::cout << "  Freed image buffer: " << width << "x" << height << std::endl;
    }
};

void demonstrateProperMemoryManagement() {
    std::cout << "\n=== Demonstrating Proper Memory Management ===" << std::endl;

    {
        std::cout << "Creating image buffers with unique_ptr..." << std::endl;
        std::vector<std::unique_ptr<ImageBuffer>> frames;

        for (int i = 0; i < 5; i++) {
            frames.push_back(std::make_unique<ImageBuffer>(1920, 1080));
        }

        std::cout << "Processing frames..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::cout << "Leaving scope - automatic cleanup:" << std::endl;
    }

    std::cout << "All memory properly freed!" << std::endl;
}

//=============================================================================
// Example 4: Array vs Vector Memory Patterns
//=============================================================================
void demonstrateContainerMemory() {
    std::cout << "\n=== Container Memory Patterns ===" << std::endl;

    // Vector with reserve (efficient)
    {
        std::cout << "Vector with reserve():" << std::endl;
        std::vector<double> efficient_vec;
        efficient_vec.reserve(10000);  // Single allocation

        for (int i = 0; i < 10000; i++) {
            efficient_vec.push_back(static_cast<double>(i));
        }
        std::cout << "  Capacity: " << efficient_vec.capacity()
                  << ", Size: " << efficient_vec.size() << std::endl;
    }

    // Vector without reserve (multiple reallocations)
    {
        std::cout << "Vector without reserve():" << std::endl;
        std::vector<double> inefficient_vec;

        for (int i = 0; i < 10000; i++) {
            inefficient_vec.push_back(static_cast<double>(i));
            if (i > 0 && (i & (i - 1)) == 0) {  // Power of 2
                std::cout << "  At size " << i << ", capacity: "
                          << inefficient_vec.capacity() << std::endl;
            }
        }
    }
}

//=============================================================================
// Example 5: Large Temporary Allocations (Common in image processing)
//=============================================================================
void demonstrateTemporaryAllocations() {
    std::cout << "\n=== Temporary Allocation Patterns ===" << std::endl;

    for (int iteration = 0; iteration < 5; iteration++) {
        std::cout << "Iteration " << iteration << ":" << std::endl;

        // Simulate image processing pipeline with large temporaries
        {
            std::vector<float> input_image(1920 * 1080, 0.0f);
            std::cout << "  Allocated input: " << input_image.size() * sizeof(float) / 1024 / 1024 << " MB" << std::endl;

            std::vector<float> filtered(1920 * 1080, 0.0f);
            std::cout << "  Allocated filtered: " << filtered.size() * sizeof(float) / 1024 / 1024 << " MB" << std::endl;

            std::vector<float> output(1920 * 1080, 0.0f);
            std::cout << "  Allocated output: " << output.size() * sizeof(float) / 1024 / 1024 << " MB" << std::endl;

            // Simulate processing
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            std::cout << "  Peak memory this iteration: ~"
                      << 3 * 1920 * 1080 * sizeof(float) / 1024 / 1024 << " MB" << std::endl;
        }
        // All temporaries freed here
        std::cout << "  Temporaries freed" << std::endl;
    }
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char** argv) {
    std::cout << "=== Memory Profiling Examples for Valgrind/Massif ===" << std::endl;
    std::cout << "Run with: valgrind --tool=massif ./memory_examples" << std::endl;
    std::cout << "Visualize with: ms_print massif.out.<pid>" << std::endl;

    bool run_leak_demo = false;

    if (argc > 1 && std::string(argv[1]) == "--with-leak") {
        run_leak_demo = true;
        std::cout << "\n*** Running WITH memory leak demonstration ***" << std::endl;
    } else {
        std::cout << "\n*** Running WITHOUT memory leak (use --with-leak to include) ***" << std::endl;
    }

    // Example 1: Normal memory growth pattern
    simulateMapGrowth();

    // Example 2: Memory leak (only if requested)
    if (run_leak_demo) {
        demonstrateMemoryLeak();
    }

    // Example 3: Proper memory management
    demonstrateProperMemoryManagement();

    // Example 4: Container memory patterns
    demonstrateContainerMemory();

    // Example 5: Temporary allocations
    demonstrateTemporaryAllocations();

    std::cout << "\n=== Program Complete ===" << std::endl;
    if (run_leak_demo) {
        std::cout << "Note: Memory leaks were intentionally created for demonstration." << std::endl;
        std::cout << "Run 'valgrind --leak-check=full ./memory_examples --with-leak' to see leak report." << std::endl;
    }

    return 0;
}
