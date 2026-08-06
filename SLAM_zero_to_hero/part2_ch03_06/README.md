# ICP Point Cloud Registration using PCL

Code exercise for point-to-point and point-to-plane ICP registration using PCL.

Both demos run on the **Stanford bunny**, registered against a copy of itself
displaced by a known transform, so every result is scored against the exact
answer. Both open an interactive viewer and step their ICP one iteration per
keystroke.

---

## Project Structure

```
part2_ch03_06/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   ├── bun_zipper_res3.ply    # Stanford bunny - the input to both demos
│   └── scene.pcd              # not used by either demo
├── images/                     # Demo output, shown under Output below
└── examples/
    ├── demo_common.hpp           # Cloud loading (.ply/.pcd), scale helpers, pose error
    ├── demo_viz.hpp              # Interactive viewer and per-keystroke ICP steppers
    ├── icp_basic.cpp             # Point-to-point ICP registration
    └── icp_point_to_plane.cpp    # Point-to-plane vs point-to-point, side by side
```

---

## Build

Dependencies:
- **PCL 1.10+** (`common`, `io`, `filters`, `registration`, `features`, `visualization`, `kdtree`, `search`) — required.
- **Eigen3 3.3+** — required.
- **MPI** — required (used by VTK/PCL visualization).

Both executables are always built; there are no optional targets.

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker
docker build . -t slam_zero_to_hero:part2_ch03_06
```

---

## Run

Neither demo takes arguments: they load `data/bun_zipper_res3.ply`, centre it,
and build the source cloud by applying a known transform, so the estimate is
compared against the exact answer.

### ICP on the Stanford bunny

```bash
# Point-to-point ICP
./build/icp_basic

# Point-to-plane ICP, stepped side by side with point-to-point
./build/icp_point_to_plane
```

### Docker

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    slam_zero_to_hero:part2_ch03_06 ./icp_point_to_plane
```

---

## Output

### Point-to-point ICP on the bunny

The source (yellow) starts displaced from the target (blue) by a known 15 deg
rotation and 8 % of the model size — large enough that the misalignment is
actually visible, and still well inside ICP's basin of attraction. Every
keystroke runs one iteration.

![](./images/icp_basic_start.png)

After 15 iterations the two clouds interleave everywhere and the overlay reports
`CONVERGED at 15`. The recovered transform matches ground truth to 0.0000 deg and
0.000000 m.

![](./images/icp_basic_converged.png)

### Point-to-plane vs point-to-point, stepped together

Both methods start from the same displaced source and advance together, one
iteration each per keystroke — point-to-point in yellow, point-to-plane in
magenta. This frame is five keystrokes in: magenta has already settled onto the
blue target (`CONVERGED at 4`), while yellow is still visibly short of it along
the right flank and the base.

![](./images/icp_point_to_plane.png)

That is the whole argument for point-to-plane in one picture — same data, same
correspondence distance, roughly a quarter of the iterations.

---

## References

- [PCL `registration` module](https://pointclouds.org/documentation/group__registration.html) (`IterativeClosestPoint`, `IterativeClosestPointWithNormals`)
- [PCL ICP tutorial](https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html)
- [PCL `visualization` module](https://pointclouds.org/documentation/group__visualization.html)
- [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/) (source of the bunny)
