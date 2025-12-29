#include <iostream>
#include <array>

// Template class example: A simple 2D point (like Eigen::Vector2)
template <typename T>
class Point2D {
public:
    T x, y;

    // Constructor
    Point2D(T x_val = 0, T y_val = 0) : x(x_val), y(y_val) {}

    // Addition operator
    Point2D<T> operator+(const Point2D<T>& other) const {
        return Point2D<T>(x + other.x, y + other.y);
    }

    // Scalar multiplication
    Point2D<T> operator*(T scalar) const {
        return Point2D<T>(x * scalar, y * scalar);
    }

    // Squared norm
    T squaredNorm() const {
        return x * x + y * y;
    }

    // Print
    void print(const std::string& name = "") const {
        std::cout << name << "(" << x << ", " << y << ")" << std::endl;
    }
};

// Template class with fixed size (like Eigen::Matrix<T, Rows, Cols>)
template <typename T, size_t N>
class FixedVector {
private:
    std::array<T, N> data_;

public:
    FixedVector() { data_.fill(T{0}); }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    static constexpr size_t size() { return N; }

    void print() const {
        std::cout << "[";
        for (size_t i = 0; i < N; ++i) {
            std::cout << data_[i];
            if (i < N - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

int main() {
    std::cout << "=== Template Class Examples ===" << std::endl;

    // 1. Point2D with different types
    std::cout << "\n1) Point2D<T> with different types:" << std::endl;
    Point2D<int> pixel(320, 240);
    pixel.print("  Pixel: ");

    Point2D<double> keypoint(320.5, 240.7);
    keypoint.print("  Keypoint: ");

    Point2D<float> uv(0.5f, 0.5f);
    uv.print("  UV coord: ");

    // 2. Point operations
    std::cout << "\n2) Point operations:" << std::endl;
    Point2D<double> p1(1.0, 2.0);
    Point2D<double> p2(3.0, 4.0);
    auto p3 = p1 + p2;
    p3.print("  p1 + p2 = ");

    auto p4 = p1 * 2.0;
    p4.print("  p1 * 2 = ");

    std::cout << "  p1.squaredNorm() = " << p1.squaredNorm() << std::endl;

    // 3. FixedVector (like Eigen fixed-size vectors)
    std::cout << "\n3) FixedVector<T, N>:" << std::endl;
    FixedVector<double, 3> translation;
    translation[0] = 0.1;
    translation[1] = 0.2;
    translation[2] = 0.3;
    std::cout << "  Translation: ";
    translation.print();

    FixedVector<float, 4> quaternion;
    quaternion[0] = 1.0f;  // w
    std::cout << "  Quaternion: ";
    quaternion.print();

    std::cout << "\n4) Why templates matter in SLAM:" << std::endl;
    std::cout << "  - Eigen uses templates: Eigen::Matrix<double, 3, 3>" << std::endl;
    std::cout << "  - OpenCV uses templates: cv::Mat_<float>" << std::endl;
    std::cout << "  - Allows compile-time optimization for fixed sizes" << std::endl;

    return 0;
}
