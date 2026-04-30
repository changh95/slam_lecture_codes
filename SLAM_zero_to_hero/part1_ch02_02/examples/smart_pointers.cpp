#include <iostream>
#include <memory>
#include <vector>

// Simulated MapPoint class (like in ORB-SLAM)
class MapPoint {
public:
    int id;
    double x, y, z;

    MapPoint(int id_, double x_, double y_, double z_)
        : id(id_), x(x_), y(y_), z(z_) {
        std::cout << "  [MapPoint " << id << " created]" << std::endl;
    }

    ~MapPoint() {
        std::cout << "  [MapPoint " << id << " destroyed]" << std::endl;
    }

    void print() const {
        std::cout << "  MapPoint " << id << ": (" << x << ", " << y << ", " << z << ")" << std::endl;
    }
};

// Simulated KeyFrame class
class KeyFrame {
public:
    int id;
    std::vector<std::shared_ptr<MapPoint>> observed_points;

    KeyFrame(int id_) : id(id_) {
        std::cout << "  [KeyFrame " << id << " created]" << std::endl;
    }

    ~KeyFrame() {
        std::cout << "  [KeyFrame " << id << " destroyed]" << std::endl;
    }

    void addObservation(std::shared_ptr<MapPoint> point) {
        observed_points.push_back(point);
    }
};

int main() {
    std::cout << "=== Smart Pointer Examples ===" << std::endl;
    std::cout << "(Essential for SLAM memory management)" << std::endl;

    // 1. unique_ptr - Exclusive ownership
    std::cout << "\n1) std::unique_ptr (exclusive ownership):" << std::endl;
    {
        std::unique_ptr<MapPoint> point = std::make_unique<MapPoint>(1, 1.0, 2.0, 3.0);
        point->print();
        // Cannot copy unique_ptr, only move
        // std::unique_ptr<MapPoint> point2 = point; // Error!
        std::unique_ptr<MapPoint> point2 = std::move(point);
        std::cout << "  After move, point2:" << std::endl;
        point2->print();
    }  // point2 automatically deleted here
    std::cout << "  (unique_ptr automatically cleaned up)" << std::endl;

    // 2. shared_ptr - Shared ownership (very common in SLAM)
    std::cout << "\n2) std::shared_ptr (shared ownership):" << std::endl;
    {
        std::shared_ptr<MapPoint> shared_point = std::make_shared<MapPoint>(2, 4.0, 5.0, 6.0);
        std::cout << "  Reference count: " << shared_point.use_count() << std::endl;

        {
            std::shared_ptr<MapPoint> shared_point2 = shared_point;  // Copy OK!
            std::cout << "  After copy, reference count: " << shared_point.use_count() << std::endl;
        }  // shared_point2 goes out of scope

        std::cout << "  After inner scope, reference count: " << shared_point.use_count() << std::endl;
    }  // shared_point deleted when ref count reaches 0
    std::cout << "  (shared_ptr deleted when ref count = 0)" << std::endl;

    // 3. SLAM-like scenario: KeyFrames sharing MapPoints
    std::cout << "\n3) SLAM scenario - KeyFrames sharing MapPoints:" << std::endl;
    {
        // Create shared map points
        auto mp1 = std::make_shared<MapPoint>(10, 1.0, 0.0, 5.0);
        auto mp2 = std::make_shared<MapPoint>(11, 2.0, 1.0, 4.0);

        // Create keyframes
        auto kf1 = std::make_unique<KeyFrame>(0);
        auto kf2 = std::make_unique<KeyFrame>(1);

        // Both keyframes observe the same map points
        kf1->addObservation(mp1);
        kf1->addObservation(mp2);
        kf2->addObservation(mp1);  // mp1 observed by both!

        std::cout << "  mp1 ref count: " << mp1.use_count() << " (kf1, kf2, and mp1)" << std::endl;
        std::cout << "  mp2 ref count: " << mp2.use_count() << " (kf1 and mp2)" << std::endl;

        // Delete kf2
        kf2.reset();
        std::cout << "  After kf2 deleted, mp1 ref count: " << mp1.use_count() << std::endl;
    }
    std::cout << "  (All cleaned up automatically!)" << std::endl;

    // 4. weak_ptr - Break circular references
    std::cout << "\n4) std::weak_ptr (avoid circular references):" << std::endl;
    std::cout << "  - Used when you need to observe but not own" << std::endl;
    std::cout << "  - Example: MapPoint -> KeyFrame back-reference" << std::endl;
    std::cout << "  - Prevents memory leaks from circular shared_ptr" << std::endl;

    return 0;
}
