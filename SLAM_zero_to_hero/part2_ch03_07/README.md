# Advanced ICP Methods on KITTI: GICP, NDT, and TEASER++

Code exercise for point cloud registration with GICP, NDT, and TEASER++.

Every demo in this chapter runs on **KITTI odometry** data, and every demo is
scored against the **KITTI ground-truth poses**. That is the point: KITTI ships a
pose per frame, so the transform between two scans is known and each method can be
graded on how close it got, not just on its own fitness score. TEASER++ is linked
as a real library rather than stood in for, so nothing here produces numbers that
could be mistaken for the published algorithm's.

---

## Project Structure

```
part2_ch03_07/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_sequences/        # KITTI seq 04, frames 0-1, with calib + GT poses
│       ├── 000000.bin
│       ├── 000001.bin
│       ├── calib.txt
│       └── poses.txt
└── examples/
    ├── demo_common.hpp         # KITTI loading, ground truth, rerun streaming
    ├── gicp_demo.cpp           # Generalized ICP (plane-to-plane) vs standard ICP
    ├── ndt_demo.cpp            # Normal Distributions Transform + parameter studies
    ├── method_comparison.cpp   # ICP vs GICP vs NDT across four experiments
    └── teaser_demo.cpp         # TEASER++ global registration (real library)
```

---

## Data

### The bundled pair

`data/sample_sequences/` holds **KITTI odometry sequence 04, frames 0 and 1** plus
that sequence's `calib.txt` and the two matching lines of `poses/04.txt`. Every
C++ demo defaults to this pair, so `./gicp_demo` with no arguments runs on real
KITTI data with real ground truth and no download:

```
Ground truth (source -> target): t = [-1.311, 0.004, -0.012] m, |t| = 1.311 m, rotation = 0.093 deg
```

### The full dataset

The more interesting registration cases - the scan-separation sweep and the loop
closures - need whole sequences. Download with `../download_kitti.py`, then extract
`data_odometry_velodyne.zip`, `data_odometry_calib.zip` and
`data_odometry_poses.zip` into one tree:

```
dataset/
├── poses/               # 00.txt .. 10.txt (ground truth; 11-21 ship none)
└── sequences/
    ├── 00/{velodyne/,calib.txt,times.txt}
    └── 04/{velodyne/,calib.txt,times.txt}
```

### How the ground truth is resolved

Only the scan paths go on the command line. From a scan the demos work out the
sequence directory, its `calib.txt`, and `poses/<NN>.txt` next to `sequences/`,
and take the frame index from the filename. KITTI poses are `T_cam0_cami`, so they
are mapped into the velodyne frame with the calib `Tr` matrix -
`T_velo0_veloi = Tr⁻¹ · T_cam0_cami · Tr` - before anything is compared. Point a
demo at sequences 11-21 (or at scans with no poses file) and it says so, then skips
the parts that need ground truth rather than quietly reporting errors against the
identity.

---

## Build

Dependencies:

| Dependency | Required? | Used for |
|---|---|---|
| **PCL 1.10+** (`common`, `io`, `filters`, `features`, `registration`, `visualization`) and **MPI** | yes | ICP, GICP, NDT, FPFH |
| **TEASER++** | for `teaser_demo` | global registration. No fallback exists: without the library the target is simply not built |
| **rerun** 0.33.0 (C++ SDK) | optional | live 3D streaming to a viewer on the host |

```bash
# Local
mkdir build && cd build
cmake ..            # USE_TEASER defaults to ON; teaser_demo is skipped if not found
make -j4

# TEASER++, if you do not have it yet
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build build -j8 --target install

# Docker: builds and installs everything above, TEASER++ included
docker build . -t slam_zero_to_hero:part2_ch03_07
```

---

## Run

### Pairwise registration (C++)

All four take two KITTI scans, or none at all to use the bundled pair.

```bash
# GICP vs standard ICP, plus a correspondence-randomness study
./build/gicp_demo
./build/gicp_demo <kitti>/sequences/04/velodyne/000000.bin \
                  <kitti>/sequences/04/velodyne/000001.bin

# NDT, plus resolution / step size / initial guess studies
./build/ndt_demo                       # optional third argument: cell size in meters
./build/ndt_demo source.bin target.bin 1.0

# ICP vs GICP vs NDT: accuracy, initial-guess robustness, density, scan separation
./build/method_comparison <kitti>/sequences/04/velodyne/000000.bin \
                          <kitti>/sequences/04/velodyne/000001.bin

# TEASER++ global registration, no initial guess
./build/teaser_demo
```

`method_comparison`'s fourth experiment registers the source frame against frames
further ahead in the same sequence, so point it at a full sequence rather than the
two-scan sample to get that table.

### Global registration on a real loop closure

Consecutive KITTI scans are ~1.5 m apart, which any motion model handles - not
where global registration earns its cost. Sequence 00 revisits streets, so pick two
frames that are metres apart in space and thousands of frames apart in time. These
pairs come back down the same street facing the other way:

```bash
# frames 1539 and 4540: 7.3 m apart, 179.7 degrees of heading difference
./build/teaser_demo <kitti>/sequences/00/velodyne/001539.bin \
                    <kitti>/sequences/00/velodyne/004540.bin

# a second revisit: frames 580 and 1407, 7.9 m apart, 169.7 degrees
./build/teaser_demo <kitti>/sequences/00/velodyne/000580.bin \
                    <kitti>/sequences/00/velodyne/001407.bin
```

The second pair has the smaller heading difference but ends up slightly harder -
2.28 m and 0.48° after refinement, against 1.76 m and 0.08° for the first. Heading
difference alone does not predict difficulty; how much of the same surface the two
scans actually see does.

### Docker

```bash
docker run -it --rm -v $(pwd)/data:/data slam_zero_to_hero:part2_ch03_07

# On a real sequence, streaming live to a rerun viewer on the host (start it
# first: rerun &). --network=host lets the demo reach the viewer at
# 127.0.0.1:9876; without a viewer the demos run the same and skip streaming.
docker run --rm --network=host \
    -v ~/data/kitti_vo_slam/extracted/dataset:/kitti:ro \
    slam_zero_to_hero:part2_ch03_07 \
    ./method_comparison /kitti/sequences/04/velodyne/000000.bin \
                        /kitti/sequences/04/velodyne/000001.bin
```

Mount the dataset root (the directory holding `sequences/` and `poses/`) so the
ground truth is reachable, not just the `velodyne` folder.

### In the rerun viewer

The pairwise demos overlay a height-coloured target, the red un-registered source,
and each method's aligned result, and log every optimization step on an
`iteration` timeline - scrub it to play back the ICP/GICP/NDT convergence.
`teaser_demo` additionally draws its FPFH correspondences as line segments.

Note that with a viewer connected the reported registration times include the
streaming overhead. Override the address with `RERUN_URL` if the viewer is not at
`rerun+http://127.0.0.1:9876/proxy`. The rerun C++ SDK and the desktop viewer have
to be the same version; the image pins the SDK to 0.33.0.

---

## Measured results

Numbers from a run of this chapter's image; timings are indicative of one machine,
the errors are not.

### One consecutive scan pair

Bundled pair, both scans voxelized to 0.3 m (124k → 29k points), identity initial
guess, so all 1.311 m of vehicle motion is left for registration to find:

| Method | Translation error | Rotation error | Time |
|---|---|---|---|
| ICP | 0.0568 m | 0.0478° | ~440 ms |
| GICP | **0.0125 m** | **0.0290°** | ~320 ms |
| NDT | **0.0103 m** | **0.0252°** | ~290 ms |

GICP's advantage over point-to-point ICP is a factor of ~4 in translation, which
is what modelling the road and wall surfaces as distributions buys.

### How far apart two scans can be (`method_comparison`, sequence 04)

| Frame gap | GT displacement | ICP | GICP | NDT |
|---|---|---|---|---|
| 1 | 1.31 m | 0.057 m | 0.012 m | 0.010 m |
| 2 | 2.63 m | 0.020 m | 0.021 m | 2.459 m |
| 5 | 6.58 m | 4.931 m | 4.994 m | 6.530 m |
| 10 | 13.24 m | 13.079 m | 13.455 m | 13.243 m |
| 20 | 26.82 m | 27.147 m | 26.530 m | 26.502 m |

All three are local methods. Past a couple of metres the identity initial guess
falls outside the basin of convergence and the error becomes the displacement
itself - the estimate has not moved. NDT gives up first, at a 2.6 m gap, because
its cell grid has the narrowest basin of the three.

### Outlier robustness, measured (`teaser_demo`)

The ground truth splits the FPFH matches into confirmed inliers and the rest, then
each row is a fresh set of the same size with a prescribed fraction replaced by
random pairs. Same data, same size, only the inlier ratio changes:

| Injected outliers | TEASER++ | | RANSAC + SVD | |
|---|---|---|---|---|
| | translation | rotation | translation | rotation |
| 50% | 0.051 m | 0.199° | 0.105 m | 0.525° |
| 70% | 0.136 m | 0.207° | 0.202 m | 0.486° |
| 90% | 0.055 m | 0.352° | 0.057 m | 0.719° |
| 95% | 0.225 m | 0.297° | 0.764 m | 2.653° |
| 99% | 0.083 m | 0.852° | **17.357 m** | **162.893°** |

RANSAC keeps up to about 90% and then falls off a cliff; TEASER++ is still within
a few centimetres at 99%, in 41 ms. The max-clique pass on the invariant (TIM)
graph is what does it - it prunes the outliers before the estimator ever sees them,
instead of hoping to sample a clean minimal set.

### A real loop closure (sequence 00, frames 1539 and 4540)

Ground truth: 7.324 m apart, **179.700°** of rotation. With no initial guess at
all, from FPFH correspondences that are mostly wrong:

| Stage | Translation error | Rotation error | Time |
|---|---|---|---|
| TEASER++ | 1.870 m | 0.321° | 31 ms |
| + GICP refinement | 1.755 m | **0.079°** | 130 ms |

Recovering a 180° reversal from scratch is the headline. The residual translation
is worth reading by direction, which the demo prints:

```
along  track (x): -1.270 m
across track (y): -0.078 m
vertical     (z):  1.209 m
```

Across-track is 8 cm - the estimate has the vehicle in the right part of the road.
The along-track component is the weakly observable one: a street is close to
translation-invariant along its own axis, so sliding a scan a metre down the road
barely changes how well it fits, while sliding it sideways into a wall does. The
1.2 m vertical component is the interesting one, because the road surface is
physically the same on both visits. The likelier explanation is height drift in
KITTI's own ground truth over the ~3 km loop rather than a registration error -
this is worth checking before treating a number like this as the algorithm's fault.
The same effect is why `teaser_demo` skips its outlier sweep on this pair and says
so: with the ground truth itself 1.75 m out, hardly any correspondence can be
confirmed as an inlier no matter how good the features are.

---

## Two parameter traps worth knowing

**NDT's step size has to match the displacement.** It is the *maximum* length of
the More-Thuente line search step. At the 0.1 m that indoor examples use, NDT's
first step on a KITTI pair comes out shorter than `TransformationEpsilon`, so it
stops after one iteration having barely moved - and still reports
`hasConverged() == true`, because PCL's flag only means the update fell below the
epsilon. `ndt_demo`'s step size study shows the cliff:

| Step size | Iterations | Translation error |
|---|---|---|
| 0.01 | 1 | 1.301 m |
| 0.05 | 1 | 1.261 m |
| 0.10 | 1 | 1.211 m |
| **0.50** | **6** | **0.010 m** |
| 1.00 | 5 | 0.015 m |

Read `Converged` next to the iteration count, never on its own.

**Set NDT's resolution before its target cloud.** `setInputTarget()` builds the
voxel-covariance grid immediately, using whatever resolution is set at that moment;
a later `setResolution()` throws that grid away and builds another. Besides the
wasted work, the discarded grid is built at PCL's 1 m default, which on a sparse
cloud has too few points per cell and prints a `Grid will not be searchable`
warning that has nothing to do with the run.

---

## References

- [PCL `registration` module](https://pointclouds.org/documentation/group__registration.html) (`GeneralizedIterativeClosestPoint`, `NormalDistributionsTransform`)
- Segal et al., "Generalized-ICP", RSS 2009
- Biber & Strasser, "The Normal Distributions Transform", IROS 2003
- Yang et al., "TEASER: Fast and Certifiable Point Cloud Registration", T-RO 2020 — [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus)
