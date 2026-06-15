# Global Features & Bag of Visual Words (DBoW2)

Place recognition and loop closure detection in Visual SLAM using the
[DBoW2](https://github.com/dorian3d/DBoW2) library.

---

## Basic Concepts

- **Bag of Visual Words (BoVW)** — represent an image as a histogram of visual
  "words" (recurring local feature patterns), the same way a text document can
  be summarized by word counts. This gives each image one compact descriptor.

- **Visual vocabulary** — a tree of representative feature patterns learned by
  k-means over many descriptors. DBoW2 stores it as a `k`-branch, `L`-deep tree
  (`k^L` possible words) for fast lookup. Words are weighted with **TF-IDF** so
  common, non-distinctive words count for less.

- **Loop closure detection** — recognizing a place the robot has already
  visited. Pipeline:
  1. Extract ORB features from the current frame.
  2. Convert to a BoW vector and query the database for similar frames.
  3. Geometrically verify the best candidate (RANSAC on feature matches).
  4. If enough inliers survive, accept it as a loop.

This is the technique ORB-SLAM uses for loop closing and relocalization.

---

## Project Structure

```
part2_ch01_09/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/                               # Sample KITTI-style images (real)
└── examples/
    ├── vocabulary_training.cpp         # Create an ORB vocabulary
    └── loop_closure_detection.cpp      # Loop detection demo + visualization
```

---

## Build

Dependencies: OpenCV 4.x and DBoW2.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker (uses slam:base as parent image)
docker build . -t slam_zero_to_hero:part2_ch01_09
```

---

## Run

### `vocabulary_training`

Trains a 100k-word ORB vocabulary (`k=10, L=5`) and writes
`orb_vocabulary.yml.gz` in the current directory.

```bash
./build/vocabulary_training                 # synthetic patterns (default)
./build/vocabulary_training /path/to/images # real images (PNG/JPG/JPEG/BMP)
```

### `loop_closure_detection`

Builds a vocabulary from the sequence, queries each frame against the database,
runs RANSAC geometric verification, and shows OpenCV windows for accepted /
rejected candidates plus a final BoW similarity heatmap.

Images are always read from a directory. With no `--data` flag it uses the
sample frames bundled in `data/` (resolved as `../data` when run from `build/`).

```bash
# Default: bundled data/ frames, with windows
./build/loop_closure_detection

# A different image directory, headless (text only)
./build/loop_closure_detection --data /path/to/images --no-vis
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--data <dir>` | image directory to load | bundled `data/` |
| `--stride <N>` | use every Nth image | 1 |
| `--max <N>` | cap loaded frame count | unlimited |
| `--min-inliers <N>` | RANSAC inliers required for a LOOP | 80 |
| `--score-threshold <X>` | minimum BoW score for a candidate | 0.1 |
| `--temporal-gap <N>` | minimum keyframe distance for a candidate | 10 |
| `--no-vis` / `--headless` | disable OpenCV windows | off |

Window controls: press **any key** to advance to the next candidate, **ESC** to
skip remaining previews.

| Visualization | Meaning |
|---------------|---------|
| Green "LOOP FOUND!" banner | inliers ≥ `--min-inliers` → verified loop |
| Red "REJECTED" banner | candidate failed geometric verification |
| Green / red lines | inlier matches between keyframe (left) and current frame (right) |
| JET heatmap (final window) | pairwise BoW similarity (bright = high) |

---

## References

- [DBoW2 GitHub](https://github.com/dorian3d/DBoW2)
- [DBoW2 Paper](http://doriangalvez.com/papers/GalvezTRO12.pdf) — "Bags of Binary Words for Fast Place Recognition in Image Sequences"
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) — uses DBoW2 for loop closure
