#pragma once

#include <vector>
#include <cmath>

namespace mathlib {

// Compute L2 norm of a vector
double computeNorm(const std::vector<double>& vec);

// Compute dot product of two vectors
double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

// Normalize a vector (in-place)
void normalize(std::vector<double>& vec);

// Compute Euclidean distance between two 3D points
double euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2);

}  // namespace mathlib
