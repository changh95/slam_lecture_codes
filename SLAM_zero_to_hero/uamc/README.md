# UAMC — handheld LiDAR-inertial-visual scanner

A complete handheld 3D scanner, hardware and software, from [U-AMC](https://github.com/U-AMC):
a 3D-printable rig that carries a Livox LiDAR, an Oak-D Pro camera and a Jetson Orin,
plus a ROS 2 Humble port of FAST-LIVO2 that runs the mapping on that Orin. Between them
they cover the part of SLAM the papers leave out — what the sensor rig physically is, how
its clocks are synchronised, and what has to change in an algorithm to make it run on
embedded hardware rather than a workstation.

The source of both repositories is vendored here verbatim, so everything the students
build is in the folder they clone. Neither is a git submodule.

| Directory | Upstream | Vendored at | License |
|---|---|---|---|
| [`UAMHD-Mapping/`](UAMHD-Mapping/) | [U-AMC/UAMHD-Mapping](https://github.com/U-AMC/UAMHD-Mapping) | `361dabd` (2026-05-31) | CC BY-NC-SA 4.0 |
| [`FAST-LIVO2-ROS2/`](FAST-LIVO2-ROS2/) | [U-AMC/FAST-LIVO2-ROS2](https://github.com/U-AMC/FAST-LIVO2-ROS2) | `a09599a` (2026-05-16) | GPLv2 |
| [`rpg_vikit_rational_polynomial/`](rpg_vikit_rational_polynomial/) | [U-AMC/rpg_vikit_rational_polynomial](https://github.com/U-AMC/rpg_vikit_rational_polynomial) | `6f213c7` (2026-03-21) | GPLv3 per `package.xml`; no `LICENSE` file — see below |

Copied 2026-08-06 with `git archive HEAD`, so each tree is exactly its upstream tracked
files at that commit — no `.git`, nothing added, nothing removed. To refresh one, re-run
the archive from a fresh clone and record the new SHA in the table above.

## Credit

Both repositories are the work of **U-AMC (Jason Kim)** and are used in this course with
his permission, including the NonCommercial term of `UAMHD-Mapping`'s CC BY-NC-SA 4.0
licence. Each tree keeps its own `LICENSE` file; those terms, not this course's, govern
the code inside it.

`rpg_vikit_rational_polynomial` declares **GPLv3** in `vikit_common/package.xml` and
`vikit_ros/package.xml`, and nine of its source files carry GNU GPL v3-or-later headers,
but the repository ships **no `LICENSE` file** and its README adds "shall not be used for
any commercial purposes" — a restriction GPLv3 does not permit a redistributor to add.
The permission U-AMC granted for this course covers it, so this is not a blocker here;
it is worth asking him to add a `LICENSE` file stating what he intends, since the repo
currently says two contradictory things. `vikit_py/package.xml` still reads
`TODO: License declaration`.

`FAST-LIVO2-ROS2` is itself a fork. The original FAST-LIVO2 is by
[Chunran Zheng](https://github.com/xuankuzcr) and HKU MARS Lab — see
[hku-mars/FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) and
[FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/abs/2408.14035)
(IEEE T-RO 2024). The unmodified ROS 1 version is a separate demo in this repo, under
[`../fast_livo2`](../fast_livo2). What U-AMC's fork adds: ROS 2 Humble, Livox Mid-360
alongside Avia, a LiDAR-inertial / LiDAR-visual-inertial pipeline split, rolling-shutter
per-row timing in the VIO patch alignment, and online image-IMU time-delay estimation.

`rpg_vikit_rational_polynomial` is a fork of the same vision toolkit, credited upstream to
[uzh-rpg/rpg_vikit](https://github.com/uzh-rpg/rpg_vikit),
[xuankuzcr/rpg_vikit](https://github.com/xuankuzcr/rpg_vikit),
[uavfly/vikit](https://github.com/uavfly/vikit) and
[Robotic-Developer-Road/rpg_vikit](https://github.com/Robotic-Developer-Road/rpg_vikit).

## UAMHD-Mapping — the hardware

Six printable parts under
[`UAMHD-Mapping/release/`](UAMHD-Mapping/release/) (LiDAR mount, left and right grips,
Orin enclosure base and lid, cable holder), all FDM-printable STL meshes — import into a
slicer and print, no CAD needed. Editable CAD source is not distributed.
[`BOM.md`](UAMHD-Mapping/BOM.md) lists them with the off-the-shelf parts, and
[`pics/`](UAMHD-Mapping/pics/) carries the assembly and wiring diagrams.

[`UAMHD-Mapping/setup/`](UAMHD-Mapping/setup/) holds the data-acquisition side, for
Ubuntu 22.04 + ROS 2 Humble on the onboard computer: `setup_depthai.sh` and
`setup_fastdds.sh` once per machine, `build_ws.sh` for the driver workspace, then
`sensor_bringup.sh` (PTP master, Fast DDS, LiDAR + camera drivers) and `bag_record.sh`
per run. `IFACE` in `init_ptp_master.sh` and the `LIDAR_A` / `LIDAR_B` topic IDs in
`bag_record.sh` are per-unit and need editing — see
[`setup/README.md`](UAMHD-Mapping/setup/README.md).

## FAST-LIVO2-ROS2 — the software

Its own Dockerfile builds from this vendored tree (`COPY . ${WS}/src/fast_livo`), so no
clone is involved:

```bash
podman build -t slam_zero_to_hero:uamc FAST-LIVO2-ROS2
```

The build still fetches Livox-SDK2 and `livox_ros_driver2` from the network. It also
still *clones* vikit at `FAST-LIVO2-ROS2/Dockerfile:71`, even though
`rpg_vikit_rational_polynomial/` is now vendored beside it — that Dockerfile's build
context is `FAST-LIVO2-ROS2/`, so a `COPY` cannot reach a sibling directory. Until the
course-level Dockerfile lands (see Status), the Docker image builds vikit from GitHub
`main` while the tree in this folder is pinned at `6f213c7`; if `main` moves, the two
diverge. The native colcon path below has no such gap. The resulting image is fully
pre-built; `ros2 launch fast_livo ...` works inside it without a `colcon build`.

Four pipelines ship, one per LiDAR × per sensor set. `img_en` in the YAML is the only
real difference between a LiDAR-inertial and a LiDAR-visual-inertial run:

| LiDAR | LiDAR-inertial | LiDAR-visual-inertial |
|---|---|---|
| Livox Avia | `mapping_aviz.launch.py` → `config/avia_only.yaml` | `mapping_aviz_lvi.launch.py` → `config/avia_lvi.yaml` |
| Livox Mid-360 | `mapping_mid360.launch.py` → `config/mid360_only.yaml` | `mapping_mid360_lvi.launch.py` → `config/mid360_lvi.yaml` |

```bash
ros2 launch fast_livo mapping_aviz_lvi.launch.py use_rviz:=True
ros2 bag play -p Retail_Street      # in another terminal
```

The FAST-LIVO2 reference bags are ROS 1, so they need `rosbags-convert` and a
`metadata.yaml` edit pointing `/livox/lidar` at `livox_ros_driver2/msg/CustomMsg`.
Full instructions are in
[`FAST-LIVO2-ROS2/README.md`](FAST-LIVO2-ROS2/README.md) §4 and
[`FAST-LIVO2-ROS2/docs/docker.md`](FAST-LIVO2-ROS2/docs/docker.md).
`../download_fast_livo2.py` already fetches `Retail_Street`.

## rpg_vikit_rational_polynomial — the vision toolkit

FAST-LIVO2's VIO half depends on vikit for camera models, patch scores and image
alignment, and this fork is where most of U-AMC's embedded-platform work lives:

- **`RationalPolynomialCamera`**, implementing OpenCV's `CALIB_RATIONAL_MODEL` — radial
  distortion as a ratio of two 6th-order polynomials (`k1..k3` over `k4..k6`) plus
  tangential `p1, p2`, for wide-angle rectilinear lenses that Brown-Conrady cannot fit.
  Registered as `cam_model: "RationalPolynomial"`. Unprojection is Newton iteration with
  an analytic 2×2 Jacobian; upstream reports ~7.3e-15 px round-trip.
- **ROS 2 parameter handling.** ROS 2 has no global parameter server, so
  `params_helper.hpp` falls back in three tiers: the node's own parameters, then a
  `SyncParametersClient` for another node's, then shelling out to `ros2 param get`.
- **aarch64 support.** CMake switches on `CMAKE_SYSTEM_PROCESSOR`: `-march=armv8-a` on
  Jetson, `-march=native -msse*` on x86_64. NEON covers `halfSample` in `vision.cpp`;
  everything else is scalar C++ on ARM.

It wants **Sophus 1.22.10** — the modern templated API, in contrast to the pre-templating
`a621ff` that the ROS 1 [`../fast_livo2`](../fast_livo2) image has to pin.

`vikit_common` is a plain CMake package meant to be installed globally; `vikit_ros` builds
under colcon. To build from the vendored trees rather than from GitHub, run this from this
folder (the local-tree equivalent of upstream's §3.1 — not yet executed on this host):

```bash
cmake -S rpg_vikit_rational_polynomial/vikit_common -B /tmp/vikit_common_build
cmake --build /tmp/vikit_common_build -j"$(nproc)"
sudo cmake --install /tmp/vikit_common_build

mkdir -p ~/fast_ws/src
ln -s "$PWD/FAST-LIVO2-ROS2" ~/fast_ws/src/fast_livo
ln -s "$PWD/rpg_vikit_rational_polynomial/vikit_ros" ~/fast_ws/src/vikit_ros
cd ~/fast_ws && colcon build --symlink-install --continue-on-error
```

## The dataset

Four ROS 2 bags recorded with the UAMHD rig itself, at two locations in Seoul named in the
bags by abbreviation — `ghm` for Gwanghwamun and `coex` for the COEX centre in Gangnam.
Each is a zstd-compressed tar of one rosbag2 directory (`metadata.yaml` + one sqlite3
`.db3`):

```bash
python3 ../download_gwanghwamun_coex.py --list                   # all four, with sizes
python3 ../download_gwanghwamun_coex.py --extract --patch-type   # default sequence, 10.4 GB
```

Archives land in `~/data/gwanghwamun_coex/` and bags unpack into its `extracted/`
subdirectory.

| Sequence | Venue | Duration | Messages | Camera frames | Archive | Extracted |
|---|---|---|---|---|---|---|
| `lvi_ghm_set` | Gwanghwamun | **841.8 s** | 326,594 | 17,976 @ 21.4 Hz | 25.13 GB | 54.31 GB |
| `lvi_set_2_restamped` | COEX | 334.7 s | 129,083 | 7,157 @ 21.4 Hz | 10.38 GB | 21.61 GB |
| `lvi_coex_set_2` | COEX | 334.7 s | 129,083 | 7,157 @ 21.4 Hz | 10.38 GB | 21.61 GB |
| `multi_lidar_coex_set` | COEX | 364.8 s | 125,167 | **none** | 2.23 GB | 3.44 GB |

**`lvi_ghm_set` is the flagship** — 14 minutes, the full sensor set, recorded 2026-03-27,
six weeks before the COEX sequences and 2.5× longer than any of them. It has **no public
download link**; ask U-AMC for it. The script handles it anyway if the archive is already
in the destination directory, so `--extract lvi_ghm_set` works offline.

**`lvi_coex_set_2` and `lvi_set_2_restamped` are the same recording.** Same duration, same
message counts per topic, same start time (2026-05-13 14:50 UTC), and `.db3` files of
byte-identical length; the `_restamped` tar was built three days later. Use
`lvi_set_2_restamped` and take the other only to compare the two — having both costs
21.6 GB of duplicate data. All four archives plus all four extracted bags need ~125 GB.

Both LiDARs record simultaneously, and they arrive as *different message types*, one Livox
CustomMsg and one plain `PointCloud2`:

| Topic | Type | Rate |
|---|---|---|
| `/livox/lidar_3JEDM180010C211` | `livox_interfaces/msg/CustomMsg` | ~10 Hz |
| `/livox/imu_3JEDM180010C211` | `sensor_msgs/msg/Imu` | ~125 Hz |
| `/livox/lidar_192_168_1_150` | `sensor_msgs/msg/PointCloud2` | ~10 Hz |
| `/livox/imu_192_168_1_150` | `sensor_msgs/msg/Imu` | ~200 Hz |
| `/oak/rgb/image_raw` | `sensor_msgs/msg/Image` | ~21 Hz |
| `/oak/rgb/camera_info` | `sensor_msgs/msg/CameraInfo` | ~21 Hz |

Two things will silently cost you a run:

- **The Avia topic's type is wrong for this build.** The bags say
  `livox_interfaces/msg/CustomMsg`; FAST-LIVO2-ROS2 builds against `livox_ros_driver2`,
  whose `CustomMsg` has a different type hash. Play the bag unpatched and the node
  subscribes to nothing — no error, just no LiDAR. Rewrite the type in the extracted
  `metadata.yaml`, which is what `--patch-type` does (it keeps a `.yaml.orig` backup).
  This is upstream README §4.2, but with a different source type than documented there.
- **`multi_lidar_coex_set` has no camera data.** Both `/oak` topics were advertised and
  recorded empty, so it can only drive the LiDAR-inertial launches, never the `_lvi`
  ones.

Unlike the HKU-MARS reference bags, these are already ROS 2, so no `rosbags-convert`
step is involved:

```bash
ros2 launch fast_livo mapping_aviz_lvi.launch.py use_rviz:=True
ros2 bag play -p ~/data/gwanghwamun_coex/extracted/lvi_ghm_set
```

The `_192_168_1_150` LiDAR publishes `PointCloud2` against a 200 Hz IMU while the
serial-named one publishes CustomMsg against a ~125 Hz IMU, so which config you pick is not
just a topic rename — `preprocess`'s LiDAR type and the IMU rate both differ between the
two.

## Status

The three source trees are in place; the course-side work is not started. Still to do: a
top-level `Dockerfile` and `scripts/` in this folder following the pattern of the other
demos — which is also what closes the vikit gap, by building from `uamc/` as the context
so the vendored vikit can be `COPY`d instead of cloned — then a verified run, a `results/`
directory with that run's trajectory and timings, and the row in `../LIST.md`.

None of the build or run commands above have been executed on this host.

The dataset is on disk: all four archives are in `~/data/gwanghwamun_coex/`, and
`lvi_ghm_set` is the one extracted, at `extracted/lvi_ghm_set/` (54.31 GB). Every duration,
message count and topic rate in this file was read from the bags' own `metadata.yaml`, and
every archive size checked against the file on disk. **`extracted/lvi_ghm_set/metadata.yaml`
is not type-patched yet** — it still says `livox_interfaces/msg/CustomMsg`, so a run against
it today would see no Avia LiDAR. Fix it with:

```bash
python3 ../download_gwanghwamun_coex.py lvi_ghm_set --extract --patch-type
```

The three COEX archives are downloaded but not unpacked, which needs another 46.6 GB.

Two spots upstream are still templated and will read oddly to a student: the
`<PURPOSE — ...>` placeholder in `UAMHD-Mapping/README.md` §1, and the
`<COMPONENT> / <SPECS>` placeholder row in its `BOM.md` off-the-shelf table.
