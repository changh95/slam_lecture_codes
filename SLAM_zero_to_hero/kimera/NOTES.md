# Kimera — notes

## Why uHumans2 and not `kimera_semantics_demo.bag`

The Kimera-Semantics README's only semantic demo is `kimera_semantics_demo.bag` on
Google Drive. Both ids ever published for it are gone:

| Source | Drive id | Result (2026-09-27) |
|---|---|---|
| MIT-SPARK/Kimera-Semantics README (and issue #52) | `1SG8cfJ6JEfY2PGXcxDPAMYzCcGBEh4Qq` | HTTP 404, "file not found" page |
| ToniRV/Kimera-Semantics-1 README (older fork) | `1jpuE6tMDoJyNq2Wu2EsVAc1r3e7qteUf` | HTTP 404 |

No mirror turned up in GitHub code search, GitHub issues, or a web search.
`download_kimera_semantics.py` still tries the README id first, so the demo becomes
the default again if the file ever comes back.

The replacement is **uHumans2**, from the same lab, rendered by the same TESSE Unity
simulator, and still downloadable (all 12 bags answered HTTP 206 with a
`#ROSBAG V2.0` header). Upstream already ships `kimera_semantics_uHumans2.launch` and
`kimera_semantics_uHumans2.rviz` for it, so this is a dataset Kimera-Semantics
officially supports, not a port.

## Why the office is the default

The first version of this folder used `apartment_s1_00h`, the smallest bag (2.7 GB,
138 s): a two-storey flat, 23 × 18 m. `office_s1_00h` (16.8 GB, 506 s, no humans) is a
whole office floor instead: three open-plan rooms, a meeting room, a lobby and the
corridors between them, 54.6 × 52.8 m. The camera path is 264 m against the
apartment's 49 m, and the mesh has 3.6× the vertices. The apartment stays as the
quick run.

`rosbag info` on the office bag: 506 s (sim time 11.4–517.6 s), bz2, 15.7 GB on disk
and 40.4 GB uncompressed, 8307 frames each of depth (`32FC1`), segmentation (`rgb8`),
left/right RGB and mono, and every camera_info; `/tesse/odom` at 200 Hz; one
`/tf_static` message (`base_link_gt → left_cam/right_cam/front_lidar/rear_lidar`) and
`/tf` with `world → base_link_gt`. These are the topics and frames the launch file
already used for the apartment, so nothing in the launch file changed except the
default label csv.

## Label mapping for the office

Kimera needs a csv that maps each segmentation colour to a class id. Kimera-Semantics
ships two for the TESSE office: upstream's `kimera_semantics_uHumans2.launch` defaults
to `tesse_multiscene_office2_segmentation_mapping.csv`, and `kimera_semantics.launch`
to `…office1…`. Both list the same object names and colours; they differ in the ids.
On 10 segmentation frames sampled across the bag:

| csv | pixels matched | ids |
|---|---|---|
| `office1` | 100 % (except one colour, below) | the 21 Kimera classes, one id per colour: floor 3, wall 19, chair 5, table 16, screen 17, … |
| `office2` | 100 % | 0–7 only, and the same colour gets different ids in different rows (grey floor = 7, 6 and 2) |
| `archviz1` (apartment) | 0 % | — |

So `office1` is the csv that means something with the class names `mesh_stats.py`
prints, and `office2` is a coarser, inconsistent remap.

The first full run with `office1` spammed the node log with
`Caught an unknown color: 125 218 3` (60 MB of log in 100 s). That colour appears
around t = 13–20 s, as the big wall of the stairwell beside the start. It is missing
from `office1`; `office2` names it `Stairs_RailingInside_Loop`, and Kimera's
`sofas_segmentation_mapping.csv` gives the same object id 15 (stairs).
[`config/uhumans2_office_segmentation_mapping.csv`](config/uhumans2_office_segmentation_mapping.csv)
is `office1` plus that one row. The next full run logged no unknown colour.

Two things in the office labels that show up in the mesh statistics:

- **The ceiling is labelled floor.** TESSE builds the office ceiling out of the same
  `Floor_*` tiles as the floor, with the same grey colour, so they get id 3.
  49 % of the "floor" vertices are above 3.2 m (the floor is at z ≈ 1.0 m, the ceiling
  at ≈ 4.2 m): about a quarter of the whole mesh is the ceiling. "ceiling" (id 4, 1.2 %)
  is only the recessed light panels, and 85 % of "appliance" is the air vents, which
  are in the ceiling too.
- **White is label 0.** `color.cpp` always paints the unknown label 0 white, whatever
  the csv says. In the office csv, id 0 covers stairs, window blinds and misc props
  (1.2 % of the mesh). `scripts/mesh_stats.py` counts white as "unknown".

RViz does not need a cutaway to show the floor plan from above: the ceiling's
triangles face down, and RViz culls their back faces. A render with and without
removing everything above 3.2 m looked the same, except for a few sprinkler pipes.

## Label mapping for the apartment

The apartment is TESSE's `archviz1` scene:
`tesse_multiscene_archviz1_segmentation_mapping.csv` is byte-identical to Hydra-ROS's
`hydra_ros/config/color/uhumans2_apartment.csv`, and on 9 segmentation frames sampled
across the bag, 100 % of pixels matched a csv row.

Two csv quirks that show up in the output:

- Several rows share an id. Kimera keeps the **last** row per id as that class's
  display colour (`kimera_semantics/src/color.cpp`), which is why the mesh colours
  differ from the segmentation image's colours. `scripts/mesh_stats.py` inverts that
  last-row map to count classes.
- Ids 12 (painting) and 17 (screen) have no row in this csv at all. The white
  vertices (1.3 % of the mesh) are label 0, which `color.cpp` paints white; this csv
  has 122 rows with id 0. (An earlier version of these notes put the white down to
  ids 12 and 17.)

## Office parameters

The office ran with the apartment's parameters, and none of them needed changing:

| Parameter | Value | Why it stays |
|---|---|---|
| `tsdf_voxel_size` | 0.05 m | Peak RAM was 8.6–9.9 GB for the whole container (Kimera, RViz, rosbag), which fits easily; 10 cm would have blurred the desks and chairs, which are what makes the office readable. |
| `max_ray_length_m` | 10 | Depth is below 3.9 m for half the pixels and below 10 m for 91 %. The corridors are longer than 10 m, but the camera drives down them, so they are complete anyway. Longer rays mostly add integration time. |
| `truncation_distance` | voxblox default, 4 × voxel = 0.2 m (`voxblox_ros/ros_params.h`) | Simulator depth has no noise, so a wider band would only thicken thin objects. |
| `min_time_between_msgs_sec` | 0.2 | At most 5 of the 16.4 labelled clouds per second are integrated; neighbouring frames overlap almost entirely, and this is what keeps Kimera far below the playback bottleneck. |
| `update_mesh_every_n_sec` | 1.0 | Only changed blocks are sent (at most 137 blocks per update during playback), so RViz keeps up too. |

The time budget was set by `rosbag play` on the bz2 bag, not by these parameters (next section); from LZ4 the office plays in real time.

## Build fix

`catkin build kimera_semantics_ros` on Noetic fails with
`pcl_config.h:7:4: error: #error PCL requires C++14 or above`.
`eigen_checks-extras.cmake` appends `-std=c++11` to `CMAKE_CXX_FLAGS` for every
package that depends on it. The Dockerfile adds `add_definitions(-std=c++14)` after
`catkin_simple()` in both Kimera packages; definitions come after `CMAKE_CXX_FLAGS`
on the compiler command line, so C++14 wins (checked in `flags.make`).

Also: `minkindr_python` is `CATKIN_IGNORE`d (it needs numpy_eigen, and nothing here
uses it), and `DISABLE_ROS1_EOL_WARNINGS=1` stops Noetic's RViz from opening a modal
end-of-life dialog over the 3D view (it showed up in the first screenshot).

## Throughput

As downloaded, the uHumans2 bags are bz2-compressed. `rosbag play` then runs one core
at 100 % just decompressing, and that set the wall clock of every early run to a third
of real time. RViz on software GL (Mesa llvmpipe) took another 150–370 % CPU on top.

Fixes:

- **LZ4.** `rosbag compress --lz4` once (command in the README): 42 min for the
  office (15.7 GB bz2 → 26.0 GB LZ4; 40.4 GB uncompressed), 8–11 min for the apartment
  (2.7 → 4.5 GB). `rosbag play` on the LZ4 bag uses 10–15 % CPU and keeps 1.0×.
- **GPU RViz.** With `--runtime=/usr/bin/nvidia-container-runtime` and
  `NVIDIA_DRIVER_CAPABILITIES=graphics,…`, the container gets `libGLX_nvidia`, and
  `run_uhumans2.sh` then leaves `LIBGL_ALWAYS_SOFTWARE` unset; `glxinfo -B` reports
  `NVIDIA GeForce RTX 5090/PCIe/SSE2` and `nvidia-smi` lists `rviz` as a graphics
  process. RViz dropped to 11–15 % CPU at its 15 fps cap. Headless runs stay on
  software GL, since Xvfb has no GPU.

Office runs, whole bag (506 s), all giving the same mesh (2,695,4xx–2,695,7xx vertices):

| Bag | RViz | Wall clock (playback) | Real-time factor | Peak RAM | Load average (1 min) start → end |
|---|---|---|---|---|---|
| bz2 | Xvfb, software GL | 1557 s | 0.33× | 9.7 GB | 18.8 → 18.0 |
| bz2 | Xvfb, software GL, alongside the next run | 1487 s | 0.34× | 8.8 GB | 5.6 → 16.3 |
| bz2 | on-screen, software GL | 1525 s | 0.33× | 8.6 GB | 5.6 → 16.3 |
| LZ4 | Xvfb, software GL | 507 s | 1.0× | 8.8 GB | 4.6 → 4.7 |
| LZ4 | on-screen, NVIDIA GL (the only Kimera run on the host) | 508 s | 1.0× | 9.9 GB | 2.5 → 0.8 (max 5.7) |

Apartment: 374–401 s (0.34–0.37×) from bz2, 139 s (1.0×) from LZ4 with NVIDIA GL.

At 1.0× Kimera itself averaged ~420 % CPU (multi-threaded integration of one cloud per
0.2 s) and kept up: all 8307 labelled clouds arrived and the mesh matches the slow runs.
Sim time (`--clock`) makes the result independent of playback speed; two apartment
runs gave 754,957 and 755,017 vertices with identical class shares to 0.1 %.

## Kimera-VIO

The folder used to hold only a thin, unverified Kimera-VIO (EuRoC) Dockerfile.
Kimera-VIO needs GTSAM + OpenGV + DBoW2 + Kimera-RPGO and a separate ROS wrapper
(Kimera-VIO-ROS) to feed poses to Kimera-Semantics. This demo uses the simulator's
ground-truth TF instead (`sensor_frame:=left_cam`), so it covers the semantic-mapping
half of Kimera only. VIO was not rebuilt or run here.
