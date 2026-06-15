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

### Docker (with X11 visualization)

The demos open OpenCV windows, so the container needs access to the host X
server. On the host, allow local containers to connect once:

```bash
xhost +local:root
```

Then launch the container with the display forwarded. The image already
contains the built binaries and the bundled `data/` frames:

```bash
docker run -it --rm \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    slam_zero_to_hero:part2_ch01_09
```

The working directory inside the container is already `build/`, so run the
demos from there, pointing `--data` at the bundled frames one level up:

```bash
# 1. Train a vocabulary once -> writes orb_vocabulary.yml.gz in the build dir
./vocabulary_training --data ../data

# 2. Loop closure reusing that vocabulary, with OpenCV windows
./loop_closure_detection --data ../data --vocab orb_vocabulary.yml.gz

# Or build a small vocabulary on the fly (no step 1), headless
./loop_closure_detection --data ../data --no-vis
```

If no X server is reachable (e.g. plain SSH) or you skip the `xhost`/`DISPLAY`
setup, add `--no-vis` / `--headless` to run text-only.

### `vocabulary_training`

Trains a 100k-word ORB vocabulary (`k=10, L=5`) from a directory of images
(required via `--data`, PNG/JPG/JPEG/BMP) and writes `orb_vocabulary.yml.gz`
in the current directory. Feed this file to `loop_closure_detection --vocab`
to reuse it instead of building one on the fly.

```bash
# Bundled sample frames in data/
./build/vocabulary_training --data data

# A different image directory
./build/vocabulary_training --data /path/to/images
```

### `loop_closure_detection`

Loads (or builds) a vocabulary, queries each frame against the database, matches
features against the best candidate, runs RANSAC geometric verification, and
shows OpenCV windows for accepted / rejected candidates plus a final BoW
similarity heatmap.

**Vocabulary.** Pass `--vocab orb_vocabulary.yml.gz` to reuse the vocabulary
trained by `vocabulary_training` — this is how real systems work (ORB-SLAM ships
a fixed vocabulary trained offline on a large corpus). Without `--vocab`, a
small vocabulary is built on the fly from the sequence so the demo runs
standalone.

**Feature matching.** Correspondences between the current frame and a candidate
use the DBoW2 direct index (FeatureVector) to restrict comparisons to shared
vocabulary nodes — ORB-SLAM's `SearchByBoW` trick — then keep only confident
matches via Lowe's ratio test, an absolute Hamming cap, and one-to-one
consistency. This is far cleaner than thresholding raw distance, so geometric
verification mostly sees true correspondences.

Images are read from a directory passed with `--data` (required). Sample
frames are bundled in `data/`.

```bash
# Reuse a trained vocabulary (run vocabulary_training first), with windows
./build/loop_closure_detection --data data --vocab orb_vocabulary.yml.gz

# Build a vocabulary on the fly, headless (text only)
./build/loop_closure_detection --data data --no-vis
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--data <dir>` | image directory to load | **required** |
| `--vocab <file>` | reuse a pretrained vocabulary (`orb_vocabulary.yml.gz`) | build on the fly |
| `--stride <N>` | use every Nth image | 1 |
| `--max <N>` | cap loaded frame count | unlimited |
| `--min-inliers <N>` | RANSAC inliers required for a LOOP | 50 |
| `--match-ratio <X>` | Lowe ratio test threshold (lower = stricter) | 0.75 |
| `--score-threshold <X>` | minimum BoW score for a candidate | auto: 0.1 / 0.03 |
| `--temporal-gap <N>` | minimum keyframe distance for a candidate | 10 |
| `--no-vis` / `--headless` | disable OpenCV windows | off |

> **Score scale & `--score-threshold`.** A larger vocabulary spreads features
> over more words, so cross-image BoW scores shrink. The bundled frames score
> ~**0.4** off-diagonal with the on-the-fly vocabulary but only ~**0.02–0.07**
> with the pretrained 100k-word one — the same fixed gate would reject every
> real loop. The default therefore adapts to the vocabulary (`0.1` on the fly,
> `0.03` pretrained); override it with `--score-threshold` for your own
> data/vocabulary. On the bundled frames both modes confirm the same four loops
> (frames 19, 24, 25, 28).

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
