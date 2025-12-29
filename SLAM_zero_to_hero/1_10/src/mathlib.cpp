#include "mathlib/mathlib.hpp"
#include <stdexcept>

namespace mathlib {

double computeNorm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (const auto& elem : vec) {
        sum += elem * elem;
    }
    return std::sqrt(sum);
}

double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void normalize(std::vector<double>& vec) {
    double norm = computeNorm(vec);
    if (norm > 1e-10) {
        for (auto& elem : vec) {
            elem /= norm;
        }
    }
}

double euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2) {
    if (p1.size() != p2.size()) {
        throw std::invalid_argument("Points must have the same dimension");
    }
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

}  // namespace mathlib
