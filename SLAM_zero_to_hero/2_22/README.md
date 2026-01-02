# RANSAC and USAC: Robust Estimation for Visual SLAM

This tutorial covers robust estimation methods used in Visual SLAM, specifically RANSAC (Random Sample Consensus) and its modern variants implemented in OpenCV's USAC (Universal Sample Consensus) framework.

---

## What is RANSAC?

RANSAC is an iterative algorithm for robust model estimation in the presence of outliers. It's essential in Visual SLAM for:

- **Homography estimation**: Planar motion, image stitching
- **Fundamental matrix estimation**: Epipolar geometry between two views
- **Essential matrix estimation**: Calibrated camera motion estimation
- **PnP (Perspective-n-Point)**: Camera pose from 2D-3D correspondences

### RANSAC Algorithm

```
1. Randomly select minimum sample size (e.g., 4 points for homography)
2. Fit model to this minimal sample
3. Count inliers (points within threshold distance)
4. Repeat for N iterations
5. Return model with most inliers
6. (Optional) Refine model using all inliers
```

### Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `ransacReprojThreshold` | Maximum reprojection error to consider inlier | 3.0 pixels |
| `confidence` | Probability of finding correct model | 0.99 |
| `maxIters` | Maximum number of iterations | 1000-5000 |

---

## USAC: Universal Sample Consensus

OpenCV 4.5+ introduced USAC, a unified framework implementing state-of-the-art RANSAC variants. USAC provides:

- Multiple sampling strategies
- Local optimization methods
- Different scoring functions
- Parallel execution

### USAC Methods in OpenCV

| Flag | Description |
|------|-------------|
| `USAC_DEFAULT` | Default USAC parameters |
| `USAC_PARALLEL` | Parallel RANSAC |
| `USAC_FM_8PTS` | 8-point algorithm for fundamental matrix |
| `USAC_FAST` | Fast but less accurate |
| `USAC_ACCURATE` | Slower but more accurate |
| `USAC_PROSAC` | Progressive sampling (uses match quality) |
| `USAC_MAGSAC` | Marginalizing sample consensus |

### UsacParams Configuration

```cpp
cv::UsacParams params;

// Sampling method
params.sampler = cv::SAMPLING_UNIFORM;      // Random uniform
params.sampler = cv::SAMPLING_PROSAC;       // Progressive (best first)
params.sampler = cv::SAMPLING_NAPSAC;       // N adjacent points
params.sampler = cv::SAMPLING_PROGRESSIVE_NAPSAC;

// Scoring method
params.score = cv::SCORE_METHOD_RANSAC;     // Binary inlier/outlier
params.score = cv::SCORE_METHOD_MSAC;       // Soft scoring
params.score = cv::SCORE_METHOD_MAGSAC;     // Marginalized scoring
params.score = cv::SCORE_METHOD_LMEDS;      // Least median of squares

// Local optimization
params.loMethod = cv::LOCAL_OPTIM_NULL;     // None
params.loMethod = cv::LOCAL_OPTIM_INNER_LO; // Inner RANSAC
params.loMethod = cv::LOCAL_OPTIM_INNER_AND_ITER_LO;
params.loMethod = cv::LOCAL_OPTIM_GC;       // Graph cut
params.loMethod = cv::LOCAL_OPTIM_SIGMA;    // Sigma consensus

// Final polishing
params.final_polisher = cv::NONE_POLISHER;
params.final_polisher = cv::LSQ_POLISHER;   // Least squares
params.final_polisher = cv::MAGSAC;         // MAGSAC polishing
params.final_polisher = cv::COV_POLISHER;   // Covariance

// Other parameters
params.confidence = 0.99;
params.threshold = 3.0;
params.maxIterations = 5000;
params.isParallel = true;
```

---

## Comparison of Methods

### Sampling Strategies

| Method | Description | When to Use |
|--------|-------------|-------------|
| Uniform | Random sampling | General purpose |
| PROSAC | Progressive sampling based on match quality | When matches are sorted by quality |
| NAPSAC | N-Adjacent Point sampling | Spatially coherent matches |

### Scoring Methods

| Method | Description | Robustness |
|--------|-------------|------------|
| RANSAC | Hard threshold (0/1) | Basic |
| MSAC | Soft threshold | Better |
| MAGSAC | Marginalizes over thresholds | Best |
| LMedS | Least median of squared residuals | Very robust to outliers |

---

## How to Build

Local build:
```bash
mkdir build && cd build
cmake ..
make -j4
```

Docker build:
```bash
docker build . -t slam_zero_to_hero:2_22
```

---

## Additional RANSAC Libraries

This exercise also includes examples using external RANSAC libraries that provide additional features beyond OpenCV's USAC:

### RansacLib

[RansacLib](https://github.com/tsattler/RansacLib) is a header-only C++ library implementing RANSAC and its variants with a clean, template-based design.

**Key Features:**
- Template-based design for flexibility
- Clear separation of Solver, Sampler, and Estimator
- LO-MSAC (Locally Optimized MSAC) for better accuracy
- Header-only, easy to integrate

**Example:**
```cpp
#include <RansacLib/ransac.h>

// Define custom solver implementing RansacLib interface
class FundamentalMatrixSolver {
public:
    int min_sample_size() const { return 8; }
    int MinimalSolver(const std::vector<int>& sample, ModelVector* models) const;
    double EvaluateModelOnPoint(const Model& model, int i) const;
    // ...
};

// Configure and run
ransac_lib::LORansacOptions options;
options.squared_inlier_threshold_ = threshold * threshold;

ransac_lib::LocallyOptimizedMSAC<Model, ModelVector, Solver> lomsac;
int inliers = lomsac.EstimateModel(options, solver, &model, &stats);
```

### MAGSAC / MAGSAC++

[MAGSAC](https://github.com/danini/magsac) is a state-of-the-art robust estimation algorithm that provides threshold-free estimation.

**Key Features:**
- **Threshold-free**: Marginalizes over noise scale sigma
- **Sigma-scoring**: Probabilistic scoring instead of binary inlier/outlier
- **Progressive NAPSAC sampling**: Spatially-aware sampling
- **MAGSAC++** (2020) is faster than original MAGSAC (2019)

**Example:**
```cpp
#include "magsac.h"
#include "estimators.h"
#include "samplers/progressive_napsac_sampler.h"

// Create estimator and sampler
magsac::utils::DefaultFundamentalMatrixEstimator estimator(maxThreshold);
gcransac::sampler::ProgressiveNapsacSampler<4> sampler(&points, {16, 8, 4, 2},
    estimator.sampleSize(), {imgW, imgH, imgW, imgH}, 0.5);

// Create MAGSAC++ instance
MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator> magsac(
    MAGSAC<...>::MAGSAC_PLUS_PLUS);
magsac.setMaximumThreshold(maxThreshold);

// Run estimation
magsac.run(points, confidence, estimator, sampler, model, iterations, score);
```

### Library Comparison

| Library | Type | Key Feature | Use Case |
|---------|------|-------------|----------|
| OpenCV USAC | Built-in | All-in-one flags | Quick prototyping |
| RansacLib | Educational | Template design | Learning RANSAC |
| MAGSAC/MAGSAC++ | SOTA 2020 | Threshold-free | Production |

---

## Examples

### 1. Homography Estimation with RANSAC

Estimate homography between two images using different RANSAC methods:

```cpp
cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0);
cv::Mat H_usac = cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, 3.0);
```

### 2. Fundamental Matrix with USAC

Estimate fundamental matrix with full USAC configuration:

```cpp
cv::UsacParams params;
params.sampler = cv::SAMPLING_PROSAC;
params.score = cv::SCORE_METHOD_MAGSAC;
params.threshold = 1.0;

cv::Mat F = cv::findFundamentalMat(pts1, pts2, mask, params);
```

### 3. Custom RANSAC Implementation

Learn RANSAC internals by implementing it from scratch for line fitting and homography estimation.

---

## Running Examples

```bash
# OpenCV USAC examples
./ransac_homography
./ransac_fundamental
./ransac_custom

# RansacLib example (template-based RANSAC)
./ransac_ransaclib

# MAGSAC++ example (threshold-free estimation)
./ransac_magsac

# With Docker
docker run -it --rm slam_zero_to_hero:2_22
```

---

## Tips for Visual SLAM

1. **Use PROSAC** when matches are sorted by descriptor distance
2. **Use MAGSAC** for automatic threshold estimation
3. **Enable local optimization** for better accuracy
4. **Set appropriate threshold** based on image resolution:
   - VGA (640x480): 1.0-2.0 pixels
   - HD (1920x1080): 2.0-4.0 pixels
5. **Increase iterations** for high outlier ratios (>50%)

### Outlier Ratio vs Iterations

| Outlier % | Required Iterations (99% confidence) |
|-----------|-------------------------------------|
| 25% | ~72 |
| 50% | ~293 |
| 75% | ~4603 |
| 90% | ~113600 |

Formula: `N = log(1 - p) / log(1 - (1 - e)^s)`
- p: desired confidence
- e: outlier ratio
- s: sample size

---

## SLAM Applications

### Feature Matching Pipeline

```
1. Detect keypoints (ORB, SIFT, SuperPoint)
2. Extract descriptors
3. Match descriptors (brute force / FLANN)
4. Apply ratio test (Lowe's ratio)
5. RANSAC to remove outliers  <-- This tutorial
6. Estimate motion (R, t)
```

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Real-time SLAM | USAC_FAST or RANSAC |
| Offline processing | USAC_ACCURATE or MAGSAC |
| Loop closure | USAC_MAGSAC with high confidence |
| Low inlier ratio | USAC_PROSAC with sorted matches |

---

## References

- [OpenCV RANSAC Tutorial](https://docs.opencv.org/4.x/d1/de0/tutorial_homography.html)
- [USAC Paper](https://cmp.felk.cvut.cz/~mishMDES/papers/Mishkin-IJCV2022.pdf)
- [MAGSAC Paper (CVPR 2019)](https://arxiv.org/abs/1803.07469)
- [MAGSAC++ Paper (CVPR 2020)](https://arxiv.org/abs/1912.05909)
- [PROSAC Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2005/papers/Chum_Matching_with_PROSAC_2005_CVPR_paper.pdf)
- [OpenCV UsacParams Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [RansacLib GitHub](https://github.com/tsattler/RansacLib)
- [MAGSAC GitHub](https://github.com/danini/magsac)
