#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// easy_profiler headers
#include <easy/profiler.h>

// Global mutex for thread-safe output
std::mutex cout_mutex;

// Thread-safe print function
void safePrint(const std::string& msg) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << msg << std::endl;
}

// Generate a synthetic test image with features
cv::Mat generateTestImage(int width, int height, int seed) {
    EASY_FUNCTION(profiler::colors::Yellow);

    cv::Mat image(height, width, CV_8UC1);
    cv::RNG rng(seed);

    // Create gradient background
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image.at<uchar>(y, x) = static_cast<uchar>((x + y) % 256);
        }
    }

    // Add random circles and rectangles (create corner features)
    for (int i = 0; i < 50; i++) {
        cv::Point center(rng.uniform(0, width), rng.uniform(0, height));
        int radius = rng.uniform(5, 30);
        cv::circle(image, center, radius, cv::Scalar(rng.uniform(0, 255)), -1);
    }

    for (int i = 0; i < 30; i++) {
        cv::Point pt1(rng.uniform(0, width), rng.uniform(0, height));
        cv::Point pt2(pt1.x + rng.uniform(10, 50), pt1.y + rng.uniform(10, 50));
        cv::rectangle(image, pt1, pt2, cv::Scalar(rng.uniform(0, 255)), -1);
    }

    // Add Gaussian noise
    cv::Mat noise(height, width, CV_8UC1);
    rng.fill(noise, cv::RNG::NORMAL, 0, 25);
    image += noise;

    return image;
}

// FAST feature detection
std::vector<cv::KeyPoint> detectFAST(const cv::Mat& image, int threshold = 20) {
    EASY_FUNCTION(profiler::colors::Green);

    std::vector<cv::KeyPoint> keypoints;

    {
        EASY_BLOCK("FAST_Create", profiler::colors::Green100);
        cv::Ptr<cv::FastFeatureDetector> detector =
            cv::FastFeatureDetector::create(threshold, true, cv::FastFeatureDetector::TYPE_9_16);

        EASY_END_BLOCK;

        EASY_BLOCK("FAST_Detect", profiler::colors::Green200);
        detector->detect(image, keypoints);
        EASY_END_BLOCK;
    }

    return keypoints;
}

// SIFT feature detection
std::vector<cv::KeyPoint> detectSIFT(const cv::Mat& image) {
    EASY_FUNCTION(profiler::colors::Blue);

    std::vector<cv::KeyPoint> keypoints;

    {
        EASY_BLOCK("SIFT_Create", profiler::colors::Blue100);
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        EASY_END_BLOCK;

        EASY_BLOCK("SIFT_Detect", profiler::colors::Blue200);
        detector->detect(image, keypoints);
        EASY_END_BLOCK;
    }

    return keypoints;
}

// ORB feature detection (bonus - commonly used in SLAM)
std::vector<cv::KeyPoint> detectORB(const cv::Mat& image) {
    EASY_FUNCTION(profiler::colors::Red);

    std::vector<cv::KeyPoint> keypoints;

    {
        EASY_BLOCK("ORB_Create", profiler::colors::Red100);
        cv::Ptr<cv::ORB> detector = cv::ORB::create(500);
        EASY_END_BLOCK;

        EASY_BLOCK("ORB_Detect", profiler::colors::Red200);
        detector->detect(image, keypoints);
        EASY_END_BLOCK;
    }

    return keypoints;
}

// Worker thread function
void workerThread(int thread_id, const cv::Mat& image,
                  std::atomic<int>& fast_total,
                  std::atomic<int>& sift_total,
                  std::atomic<int>& orb_total) {
    // Name this thread for the profiler
    std::string thread_name = "Worker-" + std::to_string(thread_id);
    EASY_THREAD(thread_name.c_str());

    EASY_BLOCK("WorkerThread_Total", profiler::colors::Cyan);

    safePrint("[Thread " + std::to_string(thread_id) + "] Starting feature detection...");

    // Detect FAST features
    {
        EASY_BLOCK("FAST_Pipeline", profiler::colors::Green);
        auto fast_kps = detectFAST(image);
        fast_total += fast_kps.size();
        safePrint("[Thread " + std::to_string(thread_id) + "] FAST: " +
                  std::to_string(fast_kps.size()) + " keypoints");
    }

    // Detect SIFT features
    {
        EASY_BLOCK("SIFT_Pipeline", profiler::colors::Blue);
        auto sift_kps = detectSIFT(image);
        sift_total += sift_kps.size();
        safePrint("[Thread " + std::to_string(thread_id) + "] SIFT: " +
                  std::to_string(sift_kps.size()) + " keypoints");
    }

    // Detect ORB features
    {
        EASY_BLOCK("ORB_Pipeline", profiler::colors::Red);
        auto orb_kps = detectORB(image);
        orb_total += orb_kps.size();
        safePrint("[Thread " + std::to_string(thread_id) + "] ORB: " +
                  std::to_string(orb_kps.size()) + " keypoints");
    }

    EASY_END_BLOCK;

    safePrint("[Thread " + std::to_string(thread_id) + "] Done!");
}

int main(int argc, char** argv) {
    // Initialize profiler
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    std::cout << "=== easy_profiler CPU Profiling Demo ===" << std::endl;
    std::cout << "Profiling OpenCV feature detection on 3 threads\n" << std::endl;

    // Configuration
    const int NUM_THREADS = 3;
    const int IMAGE_WIDTH = 1280;
    const int IMAGE_HEIGHT = 720;

    // Atomic counters for results
    std::atomic<int> fast_total{0};
    std::atomic<int> sift_total{0};
    std::atomic<int> orb_total{0};

    // Generate test images (one per thread with different seed)
    std::vector<cv::Mat> images;
    {
        EASY_BLOCK("GenerateImages", profiler::colors::Yellow);
        std::cout << "Generating " << NUM_THREADS << " test images ("
                  << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << ")..." << std::endl;

        for (int i = 0; i < NUM_THREADS; i++) {
            images.push_back(generateTestImage(IMAGE_WIDTH, IMAGE_HEIGHT, i * 1000));
        }
        std::cout << "Images generated.\n" << std::endl;
    }

    // Create and launch worker threads
    std::vector<std::thread> threads;
    {
        EASY_BLOCK("SpawnThreads", profiler::colors::Magenta);
        std::cout << "Launching " << NUM_THREADS << " worker threads..." << std::endl;

        for (int i = 0; i < NUM_THREADS; i++) {
            threads.emplace_back(workerThread, i, std::ref(images[i]),
                                 std::ref(fast_total), std::ref(sift_total), std::ref(orb_total));
        }
    }

    // Wait for all threads to complete
    {
        EASY_BLOCK("WaitForThreads", profiler::colors::Orange);
        for (auto& t : threads) {
            t.join();
        }
    }

    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total FAST keypoints: " << fast_total << std::endl;
    std::cout << "Total SIFT keypoints: " << sift_total << std::endl;
    std::cout << "Total ORB keypoints:  " << orb_total << std::endl;

    // Save profiling data
    std::string profile_path = "profile.prof";
    {
        EASY_BLOCK("SaveProfile", profiler::colors::White);
        auto blocks_count = profiler::dumpBlocksToFile(profile_path.c_str());
        std::cout << "\nProfile saved to: " << profile_path << std::endl;
        std::cout << "Total blocks recorded: " << blocks_count << std::endl;
    }

    std::cout << "\nTo visualize, run: profiler_gui " << profile_path << std::endl;
    std::cout << "(Install easy_profiler GUI on your host machine)" << std::endl;

    return 0;
}
