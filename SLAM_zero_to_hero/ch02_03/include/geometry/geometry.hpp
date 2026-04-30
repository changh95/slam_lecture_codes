#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace geometry {

// Header-only library example (like Eigen, Sophus)
// All implementations are in the header

template <typename T>
class Point3D {
public:
    T x, y, z;

    Point3D(T x_ = 0, T y_ = 0, T z_ = 0) : x(x_), y(y_), z(z_) {}

    // Vector addition
    Point3D<T> operator+(const Point3D<T>& other) const {
        return Point3D<T>(x + other.x, y + other.y, z + other.z);
    }

    // Vector subtraction
    Point3D<T> operator-(const Point3D<T>& other) const {
        return Point3D<T>(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    Point3D<T> operator*(T scalar) const {
        return Point3D<T>(x * scalar, y * scalar, z * scalar);
    }

    // Dot product
    T dot(const Point3D<T>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Cross product
    Point3D<T> cross(const Point3D<T>& other) const {
        return Point3D<T>(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Squared norm
    T squaredNorm() const {
        return x * x + y * y + z * z;
    }

    // Norm
    T norm() const {
        return std::sqrt(squaredNorm());
    }

    // Normalize
    Point3D<T> normalized() const {
        T n = norm();
        if (n > T{1e-10}) {
            return Point3D<T>(x / n, y / n, z / n);
        }
        return *this;
    }
};

// Type aliases (like Eigen::Vector3d, Eigen::Vector3f)
using Point3d = Point3D<double>;
using Point3f = Point3D<float>;

// Utility function: compute angle between two vectors
template <typename T>
T angleBetween(const Point3D<T>& a, const Point3D<T>& b) {
    T dot_val = a.dot(b);
    T norms = a.norm() * b.norm();
    if (norms < T{1e-10}) return T{0};
    return std::acos(std::clamp(dot_val / norms, T{-1}, T{1}));
}

}  // namespace geometry
