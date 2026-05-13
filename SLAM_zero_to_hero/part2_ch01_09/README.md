# Global Feature Extraction and Bag of Visual Words using DBoW2

This tutorial covers global feature extraction and the Bag of Visual Words (BoVW) approach for place recognition and loop closure detection in Visual SLAM, using the [DBoW2](https://github.com/dorian3d/DBoW2) library.

---

## Overview

While local features (like ORB or SIFT) describe individual keypoints, **global features** describe an entire image in a single compact representation. This is crucial for:

- **Loop Closure Detection**: Recognizing previously visited places
- **Place Recognition**: Image retrieval from a database
- **Relocalization**: Finding camera pose when tracking is lost

---

## What is Bag of Visual Words (BoVW)?

Bag of Visual Words is a technique borrowed from natural language processing, adapted for image retrieval. The core idea is to represent images as histograms of visual "words" instead of raw pixels.

### The Text Analogy

Just as a text document can be summarized by the frequency of words it contains (ignoring grammar and word order), an image can be summarized by the frequency of visual patterns (local features) it contains.

```
Text Document:              Image:
--------------              ------
Words: ["cat", "sat", ...]  Visual Words: [corner_pattern_1, blob_pattern_2, ...]
Word Histogram              Visual Word Histogram
```

---

## Key Concepts

### 1. Visual Vocabulary

A visual vocabulary is a set of representative visual patterns (cluster centers) learned from a large set of training images. Each pattern is called a "visual word."

```
Step 1: Extract features from many images
        Image1 -> [f1, f2, f3, ...]
        Image2 -> [f4, f5, f6, ...]
        ...

Step 2: Cluster all features (e.g., k-means)
        Clusters -> Visual Words (Vocabulary)

Step 3: Organize into hierarchical tree for fast lookup
        (This is what makes DBoW2 efficient)
```

### 2. Vocabulary Tree Structure

DBoW2 uses a hierarchical k-means tree for efficient word lookup:

```
Parameters:
  k = branching factor (children per node)
  L = depth levels

Example: k=10, L=6 gives 10^6 = 1,000,000 possible words

Tree Structure:
                    [Root]
         /    /   |   \    \
       [.]  [.]  [.]  [.]  [.] ... (k nodes)
       / \  / \  / \  / \  / \
      ...  ...  ...  ...  ...     (L levels)

      [w1][w2][w3]...[wN]         (Leaf nodes = Visual Words)
```

### 3. Image Representation (Bag of Words Vector)

Once we have a vocabulary, each image is converted to a sparse vector:

```cpp
// For each feature in the image:
//   1. Traverse tree to find closest visual word
//   2. Increment that word's count

Image -> BowVector: {word_id: weight, word_id: weight, ...}

Example: {42: 0.15, 156: 0.08, 789: 0.12, ...}
```

### 4. TF-IDF Weighting

To improve discrimination, words are weighted using TF-IDF:

- **TF (Term Frequency)**: How often a word appears in this image
- **IDF (Inverse Document Frequency)**: Penalize common words (appear in many images)

```
weight = TF * IDF

IDF(word) = log(N / n_word)
  where N = total images in database
        n_word = images containing this word
```

### 5. Image Similarity Scoring

Similarity between two images is computed by comparing their BoW vectors:

```cpp
// L1-norm scoring (default in DBoW2)
score = 1 - 0.5 * |v1 - v2|_L1

// Score ranges from 0 (completely different) to 1 (identical)
```

---

## DBoW2 Architecture

### Main Components

| Component | Description |
|-----------|-------------|
| `Vocabulary` | Hierarchical tree of visual words |
| `Database` | Stores BoW vectors with inverted index |
| `BowVector` | Sparse histogram of visual words |
| `FeatureVector` | Groups features by tree nodes (for geometric verification) |

### Supported Descriptors

| Type | Class | Description |
|------|-------|-------------|
| ORB | `OrbVocabulary`, `OrbDatabase` | Binary descriptor, fast |
| BRIEF | `BriefVocabulary`, `BriefDatabase` | Binary descriptor |
| SURF | `SurfVocabulary`, `SurfDatabase` | Floating-point descriptor |

---

## Loop Closure Detection in SLAM

Loop closure detection is critical for:
1. Correcting accumulated drift
2. Building globally consistent maps
3. Enabling revisiting known areas

### The Pipeline

```
1. Extract Features
   Current Frame -> ORB Keypoints + Descriptors

2. Query Database
   Descriptors -> BowVector -> Query(database) -> Candidate Matches

3. Geometric Verification
   For each candidate:
     - Match features using FeatureVector
     - Estimate fundamental/essential matrix
     - Check inlier ratio

4. Accept Loop Closure
   If geometric verification passes:
     - Add loop constraint to pose graph
     - Trigger global optimization
```

### Direct Index for Geometric Verification

DBoW2 provides a `FeatureVector` that groups features by vocabulary tree nodes. This enables efficient feature matching between loop candidates:

```cpp
// FeatureVector structure: {node_id: [feature_indices...], ...}
// Only compare features that share the same node

void matchFeatures(const FeatureVector& fv1, const FeatureVector& fv2) {
    // Iterate through shared nodes
    for (auto& node : shared_nodes) {
        // Only compare features within the same node
        // Much faster than brute-force matching!
    }
}
```

---

## Project Structure

```
part2_ch01_09/
├── README.md                           # This file
├── CMakeLists.txt                      # Build configuration
├── Dockerfile                          # Docker environment
├── data/                               # Small KITTI-style sample images
└── examples/
    ├── vocabulary_training.cpp         # Create vocabulary from images
    └── loop_closure_detection.cpp      # Loop detection demo + viz
```

---

## How to Build

### Dependencies
- OpenCV 4.x
- DBoW2 library

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:part2_ch01_09
```

---

## How to Run

### `vocabulary_training`

Trains a 100k-word ORB vocabulary (`k=10, L=5`) and writes
`orb_vocabulary.yml.gz` in the current working directory.

```bash
# Synthetic random patterns (default when no directory is given)
./build/vocabulary_training

# Real images from a directory (PNG/JPG/JPEG/BMP)
./build/vocabulary_training /path/to/training/images
```

### `loop_closure_detection`

Simulates online SLAM: builds a per-sequence vocabulary, transforms every
frame into a BoW vector, queries the database, and runs RANSAC geometric
verification on each candidate. Pops up OpenCV windows showing matched
inliers for both **accepted** and **rejected** candidates, and ends with a
JET-colormap heatmap of the full pairwise BoW similarity matrix.

```
Flags:
  --data <dir>             load real grayscale images from a directory
                           (default: synthetic 16-frame sequence)
  --stride <N>             use every Nth image (default 1)
  --max <N>                cap loaded frame count (default unlimited)
  --min-inliers <N>        RANSAC inliers required for a LOOP (default 80)
  --score-threshold <X>    minimum BoW score for a candidate (default 0.1)
  --temporal-gap <N>       minimum keyframe distance for a candidate (default 10)
  --no-vis / --headless    disable OpenCV windows
```

```bash
# Quick synthetic demo (16 frames, with windows)
./build/loop_closure_detection

# Real EuRoC MH_01_easy frames, ~93 keyframes, tighter threshold
./build/loop_closure_detection --data /data --stride 40 --max 100 \
    --min-inliers 120 --score-threshold 0.45 --temporal-gap 15

# Headless (text-only, useful for CI / SSH without X)
./build/loop_closure_detection --no-vis
```

Window controls: click on the OpenCV window and press **any key** to advance
to the next candidate, **ESC** to skip remaining previews.

### Docker (with X11 forwarding for visualization)

```bash
# Build (uses slam:base as parent image)
docker build . -t slam_zero_to_hero:part2_ch01_09

# Allow the container to talk to the host X server
xhost +SI:localuser:root

# Synthetic loop closure demo with windows
docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch01_09 \
    ./loop_closure_detection

# Real EuRoC MH_01_easy cam0 frames mounted into /data
docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/data/euroc_mav/MH_01_easy/mav0/cam0/data:/data:ro \
    slam_zero_to_hero:part2_ch01_09 \
    ./loop_closure_detection --data /data --stride 40 --max 100

# Train a vocabulary from real images (output goes to /output on the host)
mkdir -p /tmp/dbow2_out
docker run --rm \
    -v ~/data/euroc_mav/MH_01_easy/mav0/cam0/data:/data:ro \
    -v /tmp/dbow2_out:/output \
    -w /output \
    slam_zero_to_hero:part2_ch01_09 \
    /workspace/part2_ch01_09/build/vocabulary_training /data
```

### Visualization legend

| Panel element              | Meaning                                              |
|----------------------------|------------------------------------------------------|
| Green "LOOP FOUND!" banner | RANSAC inliers ≥ `--min-inliers` -> verified loop    |
| Red "REJECTED" banner      | BoW candidate failed geometric verification          |
| Green / red lines          | Inlier matches between left (keyframe) & right (current frame) |
| Cyan caption               | `Keyframe N <-> Current M  score=...  inliers=A/B  threshold=T` |
| JET heatmap (final window) | Pairwise BoW similarity (bright = high)              |

---

## Code Examples

### 1. Creating a Visual Vocabulary

```cpp
#include "DBoW2/DBoW2.h"
#include <opencv2/features2d.hpp>

// Extract ORB features from training images
cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
std::vector<std::vector<cv::Mat>> features;

for (const auto& image : training_images) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    // Convert to vector of single-row matrices
    std::vector<cv::Mat> desc_vec;
    for (int i = 0; i < descriptors.rows; i++) {
        desc_vec.push_back(descriptors.row(i));
    }
    features.push_back(desc_vec);
}

// Create vocabulary with k=10, L=5
OrbVocabulary voc(10, 5, DBoW2::TF_IDF, DBoW2::L1_NORM);
voc.create(features);
voc.save("vocabulary.yml.gz");
```

### 2. Querying for Loop Closure

```cpp
// Load vocabulary
OrbVocabulary voc("vocabulary.yml.gz");

// Create database with direct index (level 4 for efficient matching)
OrbDatabase db(voc, true, 4);

// Add keyframes to database
for (size_t i = 0; i < keyframes.size(); ++i) {
    db.add(keyframes[i].descriptors);
}

// Query for loop candidates
DBoW2::QueryResults results;
db.query(current_frame.descriptors, results, 5);  // Top 5 matches

// Filter and verify results
for (const auto& result : results) {
    if (result.Score > 0.3 &&
        std::abs((int)result.Id - current_id) > 30) {
        // Potential loop closure candidate
        // Proceed with geometric verification
        std::cout << "Loop candidate: Frame " << result.Id
                  << " (score: " << result.Score << ")" << std::endl;
    }
}
```

### 3. Converting Image to BoW Vector

```cpp
DBoW2::BowVector bow_vec;
DBoW2::FeatureVector feat_vec;

// Transform descriptors to BoW representation
voc.transform(descriptors, bow_vec, feat_vec, 4);

// bow_vec contains {word_id: tf-idf_weight, ...}
// feat_vec contains {node_id: [local_feature_indices], ...}
```

---

## Vocabulary Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `k` | Branching factor | 9-10 |
| `L` | Tree depth | 3-6 |
| `weighting` | TF-IDF, TF, IDF, BINARY | TF_IDF |
| `scoring` | L1_NORM, L2_NORM, CHI_SQUARE, etc. | L1_NORM |

### Parameter Trade-offs

```
Higher k, L:
  + More discriminative vocabulary
  + Better precision
  - Slower training
  - Larger vocabulary file

Lower k, L:
  + Faster training and lookup
  + Smaller file
  - Less discriminative
  - May have more false positives
```

---

## Integration with SLAM Systems

### ORB-SLAM Integration

ORB-SLAM2 and ORB-SLAM3 use DBoW2 for:
1. **Loop Closure Detection**: Query database for similar keyframes
2. **Relocalization**: Find camera pose when tracking is lost
3. **Fast Feature Matching**: Use FeatureVector for efficient matching

```cpp
// Typical ORB-SLAM loop closure workflow
class LoopClosing {
public:
    bool DetectLoop(KeyFrame* current_kf) {
        // Query database (excluding recent keyframes)
        db_.query(current_kf->bow_vec, results, 10);

        // Find consistent candidates
        for (const auto& result : results) {
            if (CheckTemporalConsistency(result)) {
                // Geometric verification with Sim3
                if (ComputeSim3(current_kf, candidate_kf)) {
                    return true;
                }
            }
        }
        return false;
    }
};
```

---

## Best Practices

1. **Vocabulary Size**: Use k=10, L=5 or L=6 for good balance (100K to 1M words)

2. **Training Data**: Train on images similar to your target environment
   - Indoor vocabulary for indoor SLAM
   - Outdoor vocabulary for outdoor SLAM

3. **Direct Index Level**: Set to 3-4 for efficient feature matching

4. **Temporal Consistency**: Require consistent matches across multiple frames

5. **Geometric Verification**: Always verify loop candidates with RANSAC

---

## Performance Considerations

| Operation | Complexity | Time (typical) |
|-----------|------------|----------------|
| Vocabulary Training | O(n * k * L * iterations) | Minutes to hours |
| Transform to BoW | O(n_features * L) | < 5 ms |
| Database Query | O(n_words * n_images) | < 10 ms |
| Add to Database | O(n_words) | < 1 ms |

---

## Common Issues

### False Positives
- Images with repetitive patterns (windows, tiles)
- Solution: Use geometric verification and temporal consistency

### Missed Loops
- Vocabulary not trained on representative data
- Solution: Train on similar environment or use larger vocabulary

### Memory Usage
- Large databases consume memory
- Solution: Use direct index level 0 if not doing geometric verification

---

## References

- [DBoW2 GitHub](https://github.com/dorian3d/DBoW2)
- [DBoW2 Paper](http://doriangalvez.com/papers/GalvezTRO12.pdf): "Bags of Binary Words for Fast Place Recognition in Image Sequences"
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2): Uses DBoW2 for loop closure
- [Vocabulary Tree Paper](https://www.robots.ox.ac.uk/~vgg/publications/papers/nister06.pdf): "Scalable Recognition with a Vocabulary Tree"
