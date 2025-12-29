/**
 * SLAM/Robotics Python Bindings using nanobind
 *
 * This example demonstrates:
 * 1. Converting Python containers (list, dict) to C++ (vector, map)
 * 2. Zero-copy numpy <-> Eigen conversion
 * 3. Exposing robotics-relevant classes and functions
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/eigen/dense.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <map>

namespace nb = nanobind;

// =============================================================================
// Part 1: Basic Container Conversions
// =============================================================================

/**
 * Sum elements of a vector (demonstrates list -> vector conversion)
 */
double sum_vector(const std::vector<double>& vec) {
    double sum = 0.0;
    for (const auto& v : vec) {
        sum += v;
    }
    return sum;
}

/**
 * Scale a vector by a factor (demonstrates vector <-> list bidirectional)
 */
std::vector<double> scale_vector(const std::vector<double>& vec, double factor) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * factor;
    }
    return result;
}

/**
 * Merge two dictionaries (demonstrates dict <-> map)
 */
std::map<std::string, double> merge_configs(
    const std::map<std::string, double>& config1,
    const std::map<std::string, double>& config2
) {
    std::map<std::string, double> merged = config1;
    for (const auto& [key, value] : config2) {
        merged[key] = value;
    }
    return merged;
}

/**
 * Filter sensor readings by threshold (demonstrates complex data flow)
 */
std::vector<std::map<std::string, double>> filter_readings(
    const std::vector<std::map<std::string, double>>& readings,
    const std::string& field,
    double threshold
) {
    std::vector<std::map<std::string, double>> filtered;
    for (const auto& reading : readings) {
        auto it = reading.find(field);
        if (it != reading.end() && it->second > threshold) {
            filtered.push_back(reading);
        }
    }
    return filtered;
}

// =============================================================================
// Part 2: Eigen/NumPy Zero-Copy Operations
// =============================================================================

/**
 * Compute transform: R * p + t (common in SLAM)
 * Uses Eigen::Ref for zero-copy numpy access
 */
Eigen::Vector3d transform_point(
    const Eigen::Ref<const Eigen::Matrix3d>& R,
    const Eigen::Ref<const Eigen::Vector3d>& t,
    const Eigen::Ref<const Eigen::Vector3d>& p
) {
    return R * p + t;
}

/**
 * Batch transform points (efficient with Eigen)
 * Input: 3xN matrix of points
 * Output: 3xN matrix of transformed points
 */
Eigen::MatrixXd transform_points_batch(
    const Eigen::Ref<const Eigen::Matrix3d>& R,
    const Eigen::Ref<const Eigen::Vector3d>& t,
    const Eigen::Ref<const Eigen::MatrixXd>& points
) {
    // R @ points + t (broadcast t to all columns)
    return (R * points).colwise() + t;
}

/**
 * Compute covariance matrix (common in state estimation)
 */
Eigen::MatrixXd compute_covariance(const Eigen::Ref<const Eigen::MatrixXd>& data) {
    // Center the data
    Eigen::VectorXd mean = data.rowwise().mean();
    Eigen::MatrixXd centered = data.colwise() - mean;

    // Compute covariance: (1/(n-1)) * X * X^T
    int n = data.cols();
    return (centered * centered.transpose()) / (n - 1);
}

/**
 * SVD decomposition (useful for point cloud alignment)
 */
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd>
compute_svd(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return std::make_tuple(svd.matrixU(), svd.singularValues(), svd.matrixV());
}

// =============================================================================
// Part 3: Robotics-Specific Classes
// =============================================================================

/**
 * SE2 pose class (2D pose: x, y, theta)
 */
class SE2 {
public:
    SE2() : x_(0), y_(0), theta_(0) {}
    SE2(double x, double y, double theta) : x_(x), y_(y), theta_(theta) {
        normalize_angle();
    }

    // Accessors
    double x() const { return x_; }
    double y() const { return y_; }
    double theta() const { return theta_; }

    // Set from values
    void set(double x, double y, double theta) {
        x_ = x;
        y_ = y;
        theta_ = theta;
        normalize_angle();
    }

    // Get as Eigen vector
    Eigen::Vector3d as_vector() const {
        return Eigen::Vector3d(x_, y_, theta_);
    }

    // Set from numpy array (zero-copy read)
    void from_vector(const Eigen::Ref<const Eigen::Vector3d>& v) {
        x_ = v(0);
        y_ = v(1);
        theta_ = v(2);
        normalize_angle();
    }

    // Get 3x3 transformation matrix
    Eigen::Matrix3d as_matrix() const {
        Eigen::Matrix3d T;
        double c = std::cos(theta_);
        double s = std::sin(theta_);
        T << c, -s, x_,
             s,  c, y_,
             0,  0, 1;
        return T;
    }

    // Compose two SE2 poses: this * other
    SE2 compose(const SE2& other) const {
        double c = std::cos(theta_);
        double s = std::sin(theta_);
        double new_x = x_ + c * other.x_ - s * other.y_;
        double new_y = y_ + s * other.x_ + c * other.y_;
        double new_theta = theta_ + other.theta_;
        return SE2(new_x, new_y, new_theta);
    }

    // Inverse of this pose
    SE2 inverse() const {
        double c = std::cos(theta_);
        double s = std::sin(theta_);
        return SE2(
            -c * x_ - s * y_,
            s * x_ - c * y_,
            -theta_
        );
    }

    // Transform a 2D point
    Eigen::Vector2d transform_point(const Eigen::Ref<const Eigen::Vector2d>& p) const {
        double c = std::cos(theta_);
        double s = std::sin(theta_);
        return Eigen::Vector2d(
            x_ + c * p(0) - s * p(1),
            y_ + s * p(0) + c * p(1)
        );
    }

    // String representation
    std::string repr() const {
        char buf[100];
        snprintf(buf, sizeof(buf), "SE2(x=%.3f, y=%.3f, theta=%.3f)", x_, y_, theta_);
        return std::string(buf);
    }

private:
    double x_, y_, theta_;

    void normalize_angle() {
        while (theta_ > M_PI) theta_ -= 2 * M_PI;
        while (theta_ < -M_PI) theta_ += 2 * M_PI;
    }
};

/**
 * Simple Kalman Filter for 1D state
 */
class KalmanFilter1D {
public:
    KalmanFilter1D(double x0, double P0, double Q, double R)
        : x_(x0), P_(P0), Q_(Q), R_(R) {}

    // Predict step
    void predict(double u = 0.0) {
        x_ = x_ + u;
        P_ = P_ + Q_;
    }

    // Update step
    void update(double z) {
        double K = P_ / (P_ + R_);  // Kalman gain
        x_ = x_ + K * (z - x_);
        P_ = (1 - K) * P_;
    }

    // Get state
    double state() const { return x_; }
    double variance() const { return P_; }

    // Get state as vector (for numpy interface)
    Eigen::Vector2d get_state_vector() const {
        return Eigen::Vector2d(x_, P_);
    }

private:
    double x_;  // State estimate
    double P_;  // Estimate uncertainty
    double Q_;  // Process noise
    double R_;  // Measurement noise
};

// =============================================================================
// Part 4: Module Definition
// =============================================================================

NB_MODULE(slam_bindings, m) {
    m.doc() = "SLAM/Robotics bindings demonstrating nanobind features";

    // --- Container conversions ---
    m.def("sum_vector", &sum_vector,
          "Sum elements of a list/vector",
          nb::arg("vec"));

    m.def("scale_vector", &scale_vector,
          "Scale a vector by a factor",
          nb::arg("vec"), nb::arg("factor"));

    m.def("merge_configs", &merge_configs,
          "Merge two configuration dictionaries",
          nb::arg("config1"), nb::arg("config2"));

    m.def("filter_readings", &filter_readings,
          "Filter sensor readings by field threshold",
          nb::arg("readings"), nb::arg("field"), nb::arg("threshold"));

    // --- Eigen/NumPy operations ---
    m.def("transform_point", &transform_point,
          "Transform point: R @ p + t",
          nb::arg("R"), nb::arg("t"), nb::arg("p"));

    m.def("transform_points_batch", &transform_points_batch,
          "Transform batch of points (3xN matrix)",
          nb::arg("R"), nb::arg("t"), nb::arg("points"));

    m.def("compute_covariance", &compute_covariance,
          "Compute covariance matrix of data (features x samples)",
          nb::arg("data"));

    m.def("compute_svd", &compute_svd,
          "Compute SVD decomposition, returns (U, S, V)",
          nb::arg("matrix"));

    // --- SE2 class ---
    nb::class_<SE2>(m, "SE2")
        .def(nb::init<>())
        .def(nb::init<double, double, double>(),
             nb::arg("x"), nb::arg("y"), nb::arg("theta"))
        .def_prop_ro("x", &SE2::x)
        .def_prop_ro("y", &SE2::y)
        .def_prop_ro("theta", &SE2::theta)
        .def("set", &SE2::set)
        .def("as_vector", &SE2::as_vector)
        .def("from_vector", &SE2::from_vector)
        .def("as_matrix", &SE2::as_matrix)
        .def("compose", &SE2::compose)
        .def("inverse", &SE2::inverse)
        .def("transform_point", &SE2::transform_point)
        .def("__repr__", &SE2::repr)
        .def("__mul__", &SE2::compose);  // Enable pose1 * pose2 syntax

    // --- Kalman Filter class ---
    nb::class_<KalmanFilter1D>(m, "KalmanFilter1D")
        .def(nb::init<double, double, double, double>(),
             nb::arg("x0"), nb::arg("P0"), nb::arg("Q"), nb::arg("R"),
             "Initialize Kalman filter with initial state, variance, process noise, measurement noise")
        .def("predict", &KalmanFilter1D::predict,
             nb::arg("u") = 0.0,
             "Predict step with optional control input")
        .def("update", &KalmanFilter1D::update,
             nb::arg("z"),
             "Update step with measurement")
        .def_prop_ro("state", &KalmanFilter1D::state)
        .def_prop_ro("variance", &KalmanFilter1D::variance)
        .def("get_state_vector", &KalmanFilter1D::get_state_vector);
}
