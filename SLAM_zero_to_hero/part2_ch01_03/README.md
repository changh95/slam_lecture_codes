# Local Feature Extraction and Matching using OpenCV

This tutorial demonstrates local feature detection and matching techniques using OpenCV - fundamental operations in visual SLAM systems.

## Overview

Local features are distinct, repeatable patterns in images that can be detected and matched across different viewpoints. In SLAM, they are used for:

- **Visual Odometry**: Tracking camera motion between consecutive frames
- **Loop Closure Detection**: Recognizing previously visited locations
- **Place Recognition**: Matching images to a pre-built map
- **Stereo Matching**: Finding correspondences between stereo image pairs

---

## Feature Detection Algorithms

### 1. FAST (Features from Accelerated Segment Test)

**Type**: Corner detector (no descriptor)

FAST is an extremely efficient corner detector designed for real-time applications.

**How it works**:
1. For each pixel p, examine 16 pixels in a circle of radius 3
2. If N contiguous pixels are all brighter (or darker) than p by threshold t, p is a corner
3. Common variants: FAST-9 (N=9), FAST-12 (N=12)

**Characteristics**:
- Very fast (suitable for real-time SLAM)
- High repeatability
- No orientation or scale information
- Typically combined with other descriptors (e.g., ORB uses FAST + BRIEF)

**SLAM Usage**: ORB-SLAM, PTAM, LSD-SLAM (for initialization)

### 2. ORB (Oriented FAST and Rotated BRIEF)

**Type**: Detector + Binary descriptor

ORB combines FAST keypoint detection with BRIEF descriptors, adding rotation invariance.

**How it works**:
1. Detect keypoints using FAST at multiple scales (image pyramid)
2. Compute orientation using intensity centroid
3. Compute rotation-invariant BRIEF descriptor (256-bit binary)

**Characteristics**:
- Fast computation (real-time capable)
- Rotation invariant
- Scale invariant (via pyramid)
- Binary descriptor - fast matching with Hamming distance
- Free and patent-unencumbered

**SLAM Usage**: ORB-SLAM, ORB-SLAM2, ORB-SLAM3 (primary feature)

### 3. SIFT (Scale-Invariant Feature Transform)

**Type**: Detector + Float descriptor

SIFT detects blob-like features and computes a highly distinctive 128-dimensional descriptor.

**How it works**:
1. Build scale-space using Difference of Gaussians (DoG)
2. Detect extrema in scale-space
3. Refine keypoint location and reject low-contrast/edge points
4. Assign orientation based on gradient histogram
5. Compute 128-dim descriptor from gradient orientations

**Characteristics**:
- Highly distinctive descriptors
- Scale and rotation invariant
- Robust to illumination changes
- Slower than binary descriptors
- Higher memory usage (128 floats per keypoint)

**SLAM Usage**: Structure-from-Motion, offline 3D reconstruction, loop closure

### 4. TEBLID (Trained Binary Local Image Descriptor)

**Type**: Binary descriptor (requires separate detector like FAST)

TEBLID uses machine learning to select optimal binary tests, achieving better accuracy than hand-crafted binary descriptors like BRIEF/ORB.

**How it works**:
1. Use FAST or other detector for keypoints
2. Extract patch around each keypoint
3. Apply learned binary tests (boosting-selected)
4. Output 256-bit or 512-bit binary descriptor

**Characteristics**:
- Binary descriptor - fast Hamming distance matching
- Better accuracy than ORB's BRIEF descriptor
- Approaches SIFT-like matching quality
- Requires OpenCV contrib modules (xfeatures2d)

**SLAM Usage**: Real-time SLAM where ORB accuracy is insufficient but SIFT is too slow

---

## Feature Matching

### Brute-Force Matcher (BFMatcher)

The simplest matching approach - compares each descriptor in set A with every descriptor in set B.

**Distance Metrics**:
- `NORM_L2`: Euclidean distance (for float descriptors like SIFT)
- `NORM_HAMMING`: Hamming distance (for binary descriptors like ORB, BRIEF)

**Complexity**: O(N * M) where N, M are the number of descriptors

### FLANN (Fast Library for Approximate Nearest Neighbors)

FLANN uses spatial data structures for efficient approximate matching.

**For float descriptors (SIFT)**:
- Uses KD-trees for fast nearest neighbor search
- Configurable accuracy/speed trade-off

**For binary descriptors (ORB)**:
- Uses LSH (Locality Sensitive Hashing)
- Parameters: table_number, key_size, multi_probe_level

**Complexity**: O(N * log(M)) average case

### Matching Strategies

#### 1. Simple Matching
Match each descriptor to its nearest neighbor. Fast but prone to false matches.

#### 2. Ratio Test (Lowe's Ratio Test)
Compare distance to best match vs second-best match:
```
if (best_distance < ratio * second_best_distance):
    accept match
```
Typical ratio: 0.7-0.8. Rejects ambiguous matches.

#### 3. Cross-Check Matching
A matches B AND B matches A. Ensures mutual consistency.

---

## Keypoint Selection: ANMS-SSC

Most detectors (FAST especially) produce thousands of candidate keypoints
clustered in high-contrast regions of the image. Taking the **top-K by
response** picks the strongest corners but leaves large portions of the
frame uncovered — bad for pose estimation, because the resulting point
distribution makes the camera-pose covariance ill-conditioned.

**ANMS-SSC** (Adaptive Non-Maximal Suppression via Square Covering, Bailo
et al. 2018) is the variant used by **Kimera-VIO**'s feature tracker. Given
N detected keypoints, it returns ~K **spatially well-distributed** keypoints
by binary-searching a suppression radius and accepting keypoints in
response order — but only if no stronger keypoint already occupies the
surrounding square.

This tutorial applies ANMS-SSC to ORB, SIFT, and FAST+TEBLID keypoints
before description and matching.

**Reference**: Bailo et al., *"Efficient adaptive non-maximal suppression
algorithms for homogeneous spatial keypoint distribution"*, PRL 2018.
Reference C++ implementation: <https://github.com/BAILOOL/ANMS-Codes>.

---

## Geometric Verification: RANSAC

The ratio test removes descriptor-level ambiguity but cannot reject
geometrically inconsistent matches (e.g. matches on repeated structures
like windows or bricks that survive the descriptor test). RANSAC with the
fundamental matrix (`cv::findFundamentalMat` with `cv::FM_RANSAC`) keeps
only matches consistent with a single epipolar geometry — the canonical
SLAM outlier filter, applied after ratio test in this tutorial.

For planar / pure-rotation scenes, use `cv::findHomography` instead.

---

## Examples in This Tutorial

### 1. `feature_detection.cpp`
Demonstrates FAST, ORB, and SIFT feature detection on synthetic images. Compares:
- Number of keypoints detected
- Detection time
- Keypoint distribution

### 2. `feature_matching.cpp`
Full matching pipeline on two KITTI sequence-00 frames (frames 0 and 3) for
ORB, SIFT, and FAST+TEBLID. For each detector the demo:

1. Detects a **large candidate pool** of keypoints.
2. Selects ~K well-distributed keypoints with **ANMS-SSC**
   (Kimera-VIO-style — see "Keypoint Selection" above).
3. Computes descriptors and runs BF k-NN matching.
4. Applies **Lowe's ratio test** (0.75 for float / 0.80 for binary).
5. Applies **RANSAC geometric verification** with the fundamental matrix.

It opens one resizable window per detector showing a 3-row pipeline
figure — *raw NN* (red), *after ratio test* (yellow), *after RANSAC* (green)
— and a final cross-method **comparison window** stacking the RANSAC-
verified inliers from ORB, SIFT, and FAST+TEBLID side-by-side.

A separate loop-closure demo uses SIFT with a stricter ratio (0.7) followed
by RANSAC for higher-precision matching.

### 3. `feature_profiling.cpp`
Comprehensive profiling comparing ORB, SIFT, and FAST+TEBLID using **easy_profiler**:
- Detailed timing breakdown (Create, Detect, Compute, Match, Visualize)
- Matching quality comparison
- Saves profiling data to `feature_profiling.prof` (view with `profiler_gui`)
- Generates visualization images for each method

---

## How to Build

**Dependencies**: OpenCV 4.x with contrib, easy_profiler (included in base image)

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
# Build base image first (from SLAM_zero_to_hero root)
docker build . -t slam:base

# Build this tutorial
cd part2_ch01_03
docker build . -t slam_zero_to_hero:part2_ch01_03
```

---

## How to Run

### Local

```bash
# Feature detection demo
./build/feature_detection

# Feature matching demo (uses synthetic images)
./build/feature_matching

# Feature profiling (ORB vs SIFT vs FAST+TEBLID)
./build/feature_profiling

# With custom images (optional)
./build/feature_matching /path/to/image1.png /path/to/image2.png
./build/feature_profiling /path/to/image1.png /path/to/image2.png
```

### Docker

```bash
# Run with X11 forwarding for visualization
docker run -it --rm \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    slam_zero_to_hero:part2_ch01_03

# Inside container
./feature_detection
./feature_matching
./feature_profiling
```

---

## Key OpenCV Classes

```cpp
// Feature detectors
cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(threshold);
cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures);
cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

// Detection
detector->detect(image, keypoints);
detector->detectAndCompute(image, mask, keypoints, descriptors);

// Matchers
cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING);  // For ORB
cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_L2);       // For SIFT

// FLANN for ORB (LSH)
cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

// FLANN for SIFT (KD-Tree)
cv::FlannBasedMatcher flann(cv::makePtr<cv::flann::KDTreeIndexParams>(5));

// Matching
matcher->match(desc1, desc2, matches);                    // Simple match
matcher->knnMatch(desc1, desc2, knn_matches, 2);         // KNN for ratio test
```

---

## SLAM-Specific Considerations

### 1. Real-time Requirements
- Prefer ORB/FAST for real-time visual odometry (30+ FPS)
- SIFT acceptable for offline processing or loop closure

### 2. Feature Distribution
- Use grid-based detection or adaptive thresholds
- Ensure features cover entire image (not clustered in one region)
- This tutorial uses **ANMS-SSC** (as in Kimera-VIO) to enforce spatial
  coverage — see the "Keypoint Selection" section above

### 3. Outlier Rejection
After matching, apply geometric verification (wired in this tutorial after
the ratio test):
- RANSAC with the fundamental/essential matrix (`cv::findFundamentalMat`)
- Homography RANSAC for planar / pure-rotation scenes (`cv::findHomography`)
- Epipolar constraint checking

### 4. Descriptor Storage
- Binary descriptors (ORB): 32 bytes per keypoint
- Float descriptors (SIFT): 512 bytes per keypoint
- Consider memory for large-scale mapping

---

## Performance Comparison

| Feature      | Detection Speed | Descriptor Size | Matching Speed | Distinctiveness |
|--------------|-----------------|-----------------|----------------|-----------------|
| FAST         | Very Fast       | N/A             | N/A            | Low             |
| ORB          | Fast            | 32 bytes        | Fast (Hamming) | Medium          |
| FAST+TEBLID  | Fast            | 32 bytes        | Fast (Hamming) | Medium-High     |
| SIFT         | Slow            | 512 bytes       | Slower (L2)    | High            |

**Note**: FAST+TEBLID combines the speed of binary descriptors with improved matching accuracy, making it an excellent choice for real-time SLAM applications where ORB's accuracy is insufficient.

---

## Further Reading

- [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)
- [SIFT Paper (Lowe, 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [FAST Paper](https://www.edwardrosten.com/work/fast.html)
- [TEBLID Paper](https://arxiv.org/abs/2002.06271) - Boosted Local Image Descriptors
- [ANMS-SSC Paper (Bailo et al., 2018)](https://www.sciencedirect.com/science/article/abs/pii/S016786551830062X) - Efficient adaptive non-maximal suppression
- [ANMS-Codes (reference implementation)](https://github.com/BAILOOL/ANMS-Codes)
- [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO) - VIO front-end that uses ANMS-SSC for feature selection
- [OpenCV Feature Detection Tutorial](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [OpenCV xfeatures2d (contrib)](https://docs.opencv.org/4.x/d2/dca/group__xfeatures2d.html)
