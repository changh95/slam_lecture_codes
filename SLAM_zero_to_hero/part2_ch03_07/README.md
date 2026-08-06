# Advanced Registration on KITTI: GICP, NDT, and TEASER++

Code exercise for point cloud registration, run as three experiments on KITTI
odometry scans and scored against the KITTI ground-truth poses — so every method
is graded on how close it got, not only on its own fitness score.

| | Experiment | Compares |
|---|---|---|
| 1 | **GICP** | PCL ICP · PCL GICP · small_gicp GICP (CPU) · fast_gicp VGICP (CUDA) |
| 2 | **NDT** | PCL NDT · fast_gicp NDTCuda, in both D2D and P2D modes |
| 3 | **TEASER++** | global registration where all of the above have no chance |

Experiment 1 is deliberately four rows rather than three. The first two differ
only in the cost function, so they answer *does GICP beat ICP*. The last three
are broadly the same algorithm in three implementations, so they answer *does the
implementation matter*. Collapsed into one comparison you cannot tell which
effect you are looking at — and it turns out the second is much the larger.

Two things are worth stating plainly up front, because both are easy to get wrong:

- **small_gicp has no NDT.** It ships exactly `{ICP, PLANE_ICP, GICP, VGICP}`, and
  its VGICP is `GICPFactor` evaluated against a Gaussian voxel map, not NDT under
  another name. That is why the NDT experiment uses fast_gicp, whose `NDTCuda` is
  the only NDT either library has — and it is CUDA-only.
- **fast_gicp's CUDA GICP is VGICP**, i.e. voxelized. It is labelled that way
  throughout rather than being passed off as the same algorithm as PCL's GICP.

---

## Project Structure

```
part2_ch03_07/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── images/                      # Demo output, shown under Output below
├── data/
│   └── sample_sequences/        # KITTI seq 04, frames 0-1, with calib + GT poses
└── examples/
    ├── demo_common.hpp          # KITTI loading, ground truth, timing, iteration tracing
    ├── gicp_demo.cpp            # Experiment 1
    ├── ndt_demo.cpp             # Experiment 2
    └── teaser_demo.cpp          # Experiment 3
```

---

## Data

`data/sample_sequences/` ships KITTI sequence 04 frames 0 and 1 together with that
sequence's `calib.txt` and the two matching ground-truth pose lines, so every demo
runs with no download when given no arguments.

The loop-closure pair needs a whole sequence. Download with `../download_kitti.py`,
then extract `data_odometry_velodyne.zip`, `data_odometry_calib.zip` and
`data_odometry_poses.zip` into one tree:

```
dataset/
├── poses/               # 00.txt .. 10.txt (sequences 11-21 ship none)
└── sequences/
    └── 04/{velodyne/,calib.txt,times.txt}
```

Only the scan paths go on the command line — the demos find the sequence's
`calib.txt` and `poses/NN.txt` themselves.

---

## Build

| Dependency | Required? | Used for |
|---|---|---|
| **PCL 1.10+** and **MPI** | yes | ICP, GICP, NDT, FPFH |
| **small_gicp** | optional | the CPU GICP column in experiment 1 |
| **fast_gicp** + **CUDA 12.8** | optional | the GPU columns in experiments 1 and 2 |
| **TEASER++** | for `teaser_demo` | without it that one target is not built |
| **rerun** 0.33.0 C++ SDK | optional | live 3D streaming to a viewer on the host |

Everything optional degrades the same way TEASER++ already did: the demos still
build and run, they just lose that column. `-DUSE_CUDA=OFF` skips the GPU
backends outright, which is what you want without an NVIDIA card.

```bash
# Docker: installs CUDA, small_gicp, fast_gicp, TEASER++ and the rerun SDK
docker build . -t slam_zero_to_hero:part2_ch03_07

# for a non-Blackwell GPU, pass its compute capability (default is 120, RTX 50xx)
docker build . --build-arg CUDA_ARCH=86 -t slam_zero_to_hero:part2_ch03_07
```

### Two traps if you build fast_gicp yourself

Upstream `fast_gicp` does not build against CUDA 12.x unpatched, and does not
build for the right GPU even once it compiles. Both fixes are in the Dockerfile,
but they are worth knowing about:

1. **It will not compile.** `fast_vgicp_cuda.cuh` and `ndt_cuda.cuh`
   hand-forward-declare `thrust::pair`, `thrust::device_vector` and
   `thrust::device_allocator` inside `namespace thrust {`. Since CCCL 2.x those
   live in an inline ABI namespace, so the redeclarations collide and you get
   ~130 `error: "thrust::pair" is ambiguous`. The errors point into
   `/usr/local/cuda/include/thrust/pair.h`, so it reads like a broken CUDA
   install rather than a fast_gicp bug.

2. **It compiles for the wrong GPU, silently.** The CUDA library uses the legacy
   `cuda_add_library()`, which ignores `CMAKE_CUDA_ARCHITECTURES`, and the project
   hard-sets `CUDA_NVCC_FLAGS` so `-DCUDA_NVCC_FLAGS` is shadowed too. The build
   then *succeeds* and emits nvcc's default `sm_52`. That still runs on a modern
   card, because the fatbin carries `compute_52` PTX the driver JITs at load —
   it just makes the first `align()` take **3300 ms instead of 12 ms**. Pass the
   arch through `CUDA_NVCC_FLAGS_RELEASE`, the one variable FindCUDA appends
   rather than replaces, and check the result with `cuobjdump --list-elf`.

Also note CUDA **13.x will not work at all**: CCCL 3.0 removed `thrust::async`,
which `find_voxel_correspondences.cu` needs on the required path for both
`NDTCuda` and `FastVGICPCuda`. 12.8 is simultaneously the oldest toolkit that can
emit `sm_120` and within the range fast_gicp still builds against.

---

## Run

Each demo takes two KITTI scans, or none at all to use the bundled pair.

```bash
./build/gicp_demo                             # experiment 1
./build/ndt_demo                              # experiment 2, optional 3rd arg: cell size (m)
./build/teaser_demo                           # experiment 3
```

Global registration is worth its cost on pairs no motion model can bridge.
Sequence 00 revisits streets, so these two frames see the same place from opposite
directions (7.3 m and 179.7 deg apart):

```bash
./build/teaser_demo <kitti>/sequences/00/velodyne/001539.bin \
                    <kitti>/sequences/00/velodyne/004540.bin
```

### Docker, with the GPU

```bash
docker run --rm \
    --runtime=/usr/bin/nvidia-container-runtime --security-opt=label=disable \
    -v ~/data/kitti_vo_slam/extracted/dataset:/kitti:ro \
    slam_zero_to_hero:part2_ch03_07 \
    ./gicp_demo /kitti/sequences/04/velodyne/000000.bin \
                /kitti/sequences/04/velodyne/000001.bin
```

Mount the dataset root — the directory holding `sequences/` and `poses/` — so the
ground truth is reachable, not just the `velodyne` folder.

On Docker proper, `--gpus all` is the usual flag. Under podman it is a **silent
no-op** — the container starts, and CUDA simply reports no devices — so the
`--runtime=` form above is used here and by the other GPU chapters in this repo.
Add `--network=host` to stream to a rerun viewer on the host.

---

## Results

Everything below is measured, not quoted: KITTI sequence 04 frames 0 → 1
(1.311 m of vehicle motion, 0.093 deg of rotation), voxel-downsampled to 0.3 m
(29064 / 28952 points), identity initial guess unless stated. Machine is a Ryzen
9 7950X (16 cores) plus an RTX 5090, inside the chapter's own container. Single
run each, so read the times as ratios rather than absolutes.

**On the timings.** Preprocessing and alignment are reported separately because
the backends divide the work differently: PCL computes GICP's covariances inside
`align()`, whereas small_gicp and fast_gicp build their trees, covariances and
voxel maps when the clouds are handed over. Quoting `align()` alone would flatter
them by roughly a factor of two. Compare on the total. (`NDTCuda` is the one
exception in the other direction — it builds its voxel maps at the top of its own
`align()`, so that cost sits in the align column.) GPU timings exclude CUDA
context creation, which the demos pay in an explicit warm-up.

### Experiment 1 — GICP

| Method | Trans err | Rot err | Prep | Align | **Total** |
|---|---|---|---|---|---|
| PCL ICP | 0.0568 m | 0.0478° | 0.0 ms | 405.6 ms | **405.6 ms** |
| PCL GICP | 0.0125 m | 0.0290° | 0.0 ms | 325.2 ms | **325.3 ms** |
| small_gicp GICP (CPU, 8 threads) | 0.0124 m | 0.0290° | 13.7 ms | 12.9 ms | **26.6 ms** |
| fast_gicp VGICP (CUDA) | 0.0098 m | 0.0268° | 14.3 ms | 4.2 ms | **18.5 ms** |

Read the first two rows for the algorithm and the last three for the
implementation, and the two effects separate cleanly:

- **The algorithm buys accuracy.** GICP is 4.5× closer than ICP — 1.3 cm against
  5.7 cm, on 1.31 m of motion — for slightly *less* time.
- **The implementation buys speed, and nothing else.** PCL GICP and small_gicp
  GICP agree to 0.0001 m: the same answer, computed 12× faster. That is the whole
  point of the fourth row. Without PCL GICP in the table, small_gicp's 12× would
  look like a property of GICP rather than of the code.

ICP's 5.7 cm is a **bias, not a convergence failure** — and the initial-guess
sweep is where that becomes visible. Translation error, in metres:

| Initial guess | PCL ICP | PCL GICP | small_gicp | CUDA VGICP |
|---|---|---|---|---|
| Exact | **0.0563** | 0.0124 | 0.0123 | 0.0098 |
| 0.2 m / 1° | 0.0568 | 0.0124 | 0.0125 | 0.0097 |
| 0.5 m / 3° | 0.0568 | 0.0124 | 0.0125 | 0.0097 |
| 1.0 m / 6° | 0.0568 | 0.0125 | 0.0125 | 0.0097 |
| Identity | 0.0568 | 0.0125 | 0.0124 | 0.0098 |

Hand ICP the ground truth itself and it does not stay there — it walks back out
to 0.0563 m, essentially where it lands from identity. Point-to-point matching
pairs points across the Velodyne's scan rings, and on a road surface those
pairings are systematically offset; the fixed point they pull ICP toward is
stable, repeatable and wrong. More iterations do not help. GICP's surface model
is exactly what removes it.

Every GICP variant is flat to four decimal places across the whole sweep, so on
consecutive KITTI scans the initial guess is simply not the binding constraint —
which is worth contrasting with NDT below, where it very much is.

#### Where the speed comes from

| Threads | Trans err | Prep | Align | Total |
|---|---|---|---|---|
| 1 | 0.0124 m | 69.1 ms | 38.2 ms | 107.3 ms |
| 2 | 0.0124 m | 36.1 ms | 20.8 ms | 56.9 ms |
| 4 | 0.0124 m | 20.4 ms | 17.7 ms | 38.1 ms |
| 8 | 0.0124 m | 11.6 ms | 10.3 ms | 21.9 ms |
| 16 | 0.0124 m | 7.3 ms | 6.9 ms | 14.2 ms |

The error column does not move at all — threading changes how fast the same
answer arrives, not what it is. Note that even single-threaded, small_gicp is
3× faster than PCL GICP (107 ms vs 325 ms), so threading is only part of the
story; the rest is the KdTree and the data structures.

#### When the GPU is worth it

| Voxel leaf | Points | CPU total | GPU total | Speedup | CPU err | GPU err |
|---|---|---|---|---|---|---|
| 1.6 m | 3573 | 2.6 ms | 11.9 ms | **0.22×** | 0.0789 m | 0.0709 m |
| 0.8 m | 9486 | 7.0 ms | 6.7 ms | 1.04× | 0.0484 m | 0.0439 m |
| 0.4 m | 21568 | 17.7 ms | 11.5 ms | 1.54× | 0.0109 m | 0.0100 m |
| 0.2 m | 41844 | 36.5 ms | 22.6 ms | 1.61× | 0.0135 m | 0.0118 m |

A GPU has fixed overheads to earn back, and on a small cloud it does not: at 3573
points the RTX 5090 is **4.5× slower** than the CPU. The crossover is around 9500
points, and even at 42k the win is only 1.6×. Against a well-threaded CPU
implementation, the GPU is a modest improvement on this workload — not the order
of magnitude the raw `align()` numbers alone would suggest.

### Experiment 2 — NDT

| Method | Trans err | Rot err | Iters | Prep | Align | **Total** |
|---|---|---|---|---|---|---|
| PCL NDT | 0.0103 m | 0.0252° | 6 | 2.2 ms | 283.5 ms | **285.7 ms** |
| NDTCuda (D2D) | 0.0676 m | 0.0457° | 12 | 0.2 ms | 5.3 ms | **5.5 ms** |
| NDTCuda (P2D) | 0.0344 m | 0.0465° | 12 | 0.2 ms | 5.2 ms | **5.3 ms** |

At a shared 1.0 m resolution the GPU is **52× faster and 3–7× less accurate**.
That is the honest headline, and it is more useful than "the GPU is faster"
— but it is also not the whole picture, for two reasons.

**The two implementations want different cell sizes.** Sweeping resolution shows
they do not even peak in the same place:

| Resolution | PCL NDT err | NDTCuda D2D err |
|---|---|---|
| 0.5 m | 1.3115 m *(failed, 1 iteration)* | **0.0154 m** |
| 1.0 m | 0.0103 m | 0.0675 m |
| 2.0 m | **0.0076 m** | 0.1540 m |
| 3.0 m | 0.0154 m | 0.3746 m |
| 5.0 m | 0.0122 m | 0.7222 m |

PCL NDT is best at 2.0 m and fails outright at 0.5 m; NDTCuda is best at 0.5 m
and degrades monotonically from there. So comparing both at the 1.0 m default
handicaps the GPU. Tuned against tuned it is 0.0154 m in 10.0 ms versus 0.0076 m
in 303 ms — still half the accuracy, but now **30× faster** rather than 52×.

**And the GPU version is far more robust.** Perturbing the initial guess:

| Initial guess | PCL NDT | NDTCuda (D2D) |
|---|---|---|
| Exact | 0.0099 m / 0.027° | 0.0427 m / 0.042° |
| 0.2 m / 1° | 0.0085 m / 0.024° | 0.0677 m / 0.047° |
| 0.5 m / 3° | 0.0710 m / 0.505° | 0.0672 m / 0.046° |
| 1.0 m / 6° | **1.0699 m / 5.72°** | 0.0675 m / 0.046° |
| Identity | 0.0103 m / 0.025° | 0.0670 m / 0.046° |

PCL NDT is the more accurate method right up until it isn't: at 6 degrees of
heading error it loses the alignment completely. NDTCuda barely moves across the
whole sweep. So the choice is not simply speed against accuracy — it is a precise
method with a narrow basin against a coarser one with a wide basin.

Note also that PCL NDT's basin is narrow **in rotation, not translation**: the
identity guess is wrong by the full 1.311 m and is handled fine, while the
"1.0 m / 6°" guess is wrong by *less* translation and fails. A motion model that
gets the heading wrong is worse than having no motion model at all.

Every row above reports `Converged: YES`, including the 1.07 m failure. PCL's
flag only means the update fell below the epsilon. Read it next to the iteration
count — and the step-size study is the cleanest example of why:

| Step size | Iters | Trans err |
|---|---|---|
| 0.01 | 1 | 1.3009 m |
| 0.05 | 1 | 1.2609 m |
| 0.10 | 1 | 1.2109 m |
| 0.50 | 6 | **0.0103 m** |
| 1.00 | 5 | 0.0153 m |

The step size is the *maximum* line-search step, so it has to be set against the
displacement being recovered. At KITTI's 1.3–1.5 m scan spacing the small values
stall: the first step comes out shorter than `TransformationEpsilon`, NDT stops
after one iteration having barely moved, and reports success. 0.1 m is the value
indoor NDT tutorials use.

### Experiment 3 — TEASER++

Sequence 00 frames 1539 and 4540: the same street driven the other way, **7.324 m
and 179.700 deg apart**, no initial guess. Voxelized to 0.5 m, 1039 FPFH
correspondences.

| Stage | Trans err | Rot err | Time |
|---|---|---|---|
| TEASER++ (167 / 1039 max-clique inliers) | 1.8699 m | 0.3211° | 28 ms |
| + GICP refinement (coarse to fine) | 1.7552 m | **0.0788°** | 132 ms |

A 179.7 deg reversal recovered to within 0.08 deg, from correspondences that are
mostly wrong, in 28 ms, with nothing to start from. Every local method in
experiments 1 and 2 needs the initial overlap to be good enough that
nearest-neighbour correspondences are mostly right; at 7.3 m and half a turn none
of them are close.

**The 1.76 m of residual translation is not all solver error.** The demo breaks it
down as −1.270 m along-track, −0.078 m across-track and +1.209 m vertical. A
1.2 m vertical offset between two passes of the same flat street is ground-truth
drift accumulated over the 3000 frames between them, not a misregistration — and
the same run can only confirm 4 of 1039 correspondences against the ground truth
within 1 m, which says the same thing from the other side. Take the **rotation**
as the honest accuracy figure on this pair and read the translation as an upper
bound. The along-track component being the largest of the three is also expected:
a street is nearly translation-invariant along its own axis, so that direction is
the weakly observable one.

---

## Output

The views below are the rerun viewer. Override its address with `RERUN_URL` if it
is not at `rerun+http://127.0.0.1:9876/proxy`.

Alongside the 3D views there are two graphs, `translation_error` and
`rotation_error`, scoring every optimization step against the ground truth. All
methods are drawn in the same graph on a shared axis, so the convergence rates
compare directly. Point 0 is the initial guess, and the graphs share the
`iteration` timeline with the 3D playback, so scrubbing moves the clouds and the
curves together.

Getting those curves out of the newer backends takes some care, and the failure
mode is silent. Both small_gicp and fast_gicp derive from `pcl::Registration`, so
`registerVisualizationCallback()` compiles and returns true — but neither library
ever invokes `update_visualizer_`, so the curve would come back holding only the
seeded initial guess: one point, no error, no warning. The demos instead override
fast_gicp's protected virtual `linearize()`, which both `step_gn()` and
`step_lm()` call exactly once per outer iteration, and inject a recording
optimizer into small_gicp's `Registration<>` template. See `demo_common.hpp`.

In the 3D views the target is coloured by height, the un-registered source is
red, and each method's result sits on top of it. The red offset is the 1.31 m the
vehicle moved between the two scans, which registration has to recover from an
identity initial guess.

### TEASER++ on a KITTI loop closure

Sequence 00 frames 1539 and 4540, the same street driven in the opposite
direction. The grey web is the FPFH correspondences handed to TEASER++, most of
them wrong; the solver recovers the 179.7 deg reversal from them with no initial
guess, and GICP refines the result (orange).

![](./images/teaser_loop_closure.png)

---

## References

- [PCL `registration` module](https://pointclouds.org/documentation/group__registration.html)
- Segal et al., "Generalized-ICP", RSS 2009
- Biber & Strasser, "The Normal Distributions Transform", IROS 2003
- Yang et al., "TEASER: Fast and Certifiable Point Cloud Registration", T-RO 2020 — [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus)
- Koide et al., "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration", ICRA 2021 — [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)
- [small_gicp](https://github.com/koide3/small_gicp) — Koide's newer, faster CPU rewrite
