# C++ Memory Profiling with Valgrind and Massif

This tutorial demonstrates how to profile memory usage and detect memory leaks in C++ using Valgrind and its Massif tool.

Memory profiling is essential for SLAM systems where:
- Maps grow continuously over time
- Large image buffers are frequently allocated
- Long-running processes need stable memory usage

## Tools Overview

### Valgrind
- **Memcheck**: Detects memory leaks, invalid accesses, uninitialized values
- **Massif**: Heap profiler that tracks memory usage over time

### When to Use Each Tool

| Tool | Use Case |
|------|----------|
| `--leak-check` | Find memory leaks and their sources |
| `--tool=massif` | Profile heap memory usage over time |
| `ms_print` | Visualize massif output as ASCII graph |

---

## How to Build

Local build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4
```

Docker build:
```bash
docker build . -t slam_zero_to_hero:1_12
```

---

## How to Run

### Option 1: Local Execution

```bash
# Run without memory leak (normal execution)
./build/memory_examples

# Run WITH intentional memory leak (for demonstration)
./build/memory_examples --with-leak
```

### Option 2: Memory Leak Detection

```bash
# Basic leak check
valgrind --leak-check=full ./build/memory_examples --with-leak

# Detailed leak check with source locations
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
    ./build/memory_examples --with-leak
```

### Option 3: Heap Profiling with Massif

```bash
# Run with massif profiler
valgrind --tool=massif ./build/memory_examples

# View the results (replace <pid> with actual process ID)
ms_print massif.out.<pid>

# Or save to file
ms_print massif.out.<pid> > memory_report.txt
```

### Option 4: Docker Execution

```bash
# Run and get profiling output
docker run -it --rm -v $(pwd)/output:/output slam_zero_to_hero:1_12

# View the generated report
cat output/massif_report.txt
```

---

## Understanding the Examples

### 1. Map Growth Simulation
Demonstrates gradual memory buildup as map points accumulate (common in SLAM):
```cpp
std::vector<std::unique_ptr<MapPoint>> map_points;
for (int frame = 0; frame < 100; frame++) {
    map_points.push_back(std::make_unique<MapPoint>(x, y, z));
}
// Memory properly freed when vector goes out of scope
```

### 2. Memory Leak (Intentional Bug)
Shows what happens when memory isn't freed:
```cpp
SensorData* data = new SensorData(1000);
// BUG: Missing delete!
// delete data;  // This causes the leak
```

### 3. Proper Memory Management
Demonstrates RAII with smart pointers:
```cpp
std::vector<std::unique_ptr<ImageBuffer>> frames;
frames.push_back(std::make_unique<ImageBuffer>(1920, 1080));
// Automatically freed when frames goes out of scope
```

### 4. Container Memory Patterns
Compares efficient vs inefficient vector usage:
```cpp
// Efficient: single allocation
std::vector<double> v;
v.reserve(10000);

// Inefficient: multiple reallocations
std::vector<double> v2;
for (int i = 0; i < 10000; i++) v2.push_back(i);
```

### 5. Temporary Allocations
Shows memory spikes from large temporary buffers:
```cpp
{
    std::vector<float> input(1920*1080);   // ~7.9 MB
    std::vector<float> filtered(1920*1080); // ~7.9 MB
    std::vector<float> output(1920*1080);   // ~7.9 MB
    // Peak: ~24 MB
}
// All freed here
```

---

## Reading Valgrind Output

### Leak Check Output
```
==12345== LEAK SUMMARY:
==12345==    definitely lost: 8,016,000 bytes in 1,000 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
==12345==    possibly lost: 0 bytes in 0 blocks
==12345==    still reachable: 0 bytes in 0 blocks
==12345==    suppressed: 0 bytes in 0 blocks
```

- **definitely lost**: Memory that was never freed (real leaks)
- **indirectly lost**: Memory reachable only through leaked memory
- **possibly lost**: Memory that might be leaked
- **still reachable**: Memory still accessible at exit (not necessarily a bug)

### Massif Output (ms_print)
```
    MB
23.45^                                                   #
     |                                                 ###
     |                                              @@@@@@
     |                                           @@@@@@@@@
     |                                        @@@@@@@@@@@@
     |                                     @@@@@@@@@@@@@@@
     |                                  @@@@@@@@@@@@@@@@@@
     |                               @@@@@@@@@@@@@@@@@@@@@
     |                            @@@@@@@@@@@@@@@@@@@@@@@@
     |                         @@@@@@@@@@@@@@@@@@@@@@@@@@@
     |                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
     |                   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
     |                :::::::::@@@@@@@@@@@@@@@@@@@@@@@@@@@@
     |             :::::::::::@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
     |          ::::::::::::::@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   0 +--------------------------------------------------------------->
```

---

## Best Practices for SLAM Memory Management

1. **Use Smart Pointers**
   ```cpp
   // Good
   std::unique_ptr<MapPoint> point = std::make_unique<MapPoint>();

   // Avoid
   MapPoint* point = new MapPoint();  // Easy to forget delete
   ```

2. **Reserve Vector Capacity**
   ```cpp
   std::vector<KeyPoint> keypoints;
   keypoints.reserve(2000);  // Avoid reallocations
   ```

3. **Use RAII Patterns**
   ```cpp
   class ImageProcessor {
       std::vector<uint8_t> buffer_;
   public:
       ImageProcessor(size_t size) : buffer_(size) {}
       // Destructor automatically frees buffer_
   };
   ```

4. **Profile Regularly**
   - Run Valgrind during development
   - Check for leaks before merging code
   - Monitor memory growth in long-running tests

---

## Common Valgrind Options

```bash
# Full leak check with source info
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./app

# Massif with time-based snapshots
valgrind --tool=massif --time-unit=ms ./app

# Massif tracking stack memory too
valgrind --tool=massif --stacks=yes ./app

# Generate callgrind output (for KCachegrind visualization)
valgrind --tool=callgrind ./app
```

---

## Troubleshooting

**Valgrind is slow**: Yes, it adds 10-50x overhead. Use smaller test cases.

**False positives**: Some libraries have known "leaks" that are intentional. Use suppression files:
```bash
valgrind --suppressions=myapp.supp ./app
```

**Missing debug symbols**: Build with `-g -O0` for best results:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```
