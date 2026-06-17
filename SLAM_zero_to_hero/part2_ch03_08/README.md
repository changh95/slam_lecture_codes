# Octree, OctoMap, and Bonxai: Spatial Data Structures for 3D Mapping

Code exercise demonstrating PCL Octree spatial search, OctoMap probabilistic occupancy mapping, and Bonxai hash-based voxel mapping.

---

## Project Structure

```
part2_ch03_08/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── 3rdparty/
│   └── bonxai/                 # Header-only Bonxai (cloned at build time)
├── data/
│   └── sample_clouds/
└── examples/
    ├── octree_demo.cpp         # PCL Octree voxel/KNN/radius search
    ├── octomap_demo.cpp        # OctoMap probabilistic occupancy mapping
    ├── octomap_navigation.cpp  # Ray-cast path check with OctoMap
    ├── bonxai_demo.cpp         # Bonxai voxel grid (built only if Bonxai headers present)
    └── comparison.cpp          # Compare all three methods (built only if Bonxai headers present)
```

---

## Build

Dependencies:
- **PCL 1.12+** (`common`, `io`, `octree`, `visualization`) and **MPI** — required.
- **OctoMap 1.9+** — required.
- **Bonxai** — header-only, cloned automatically in Docker; for local builds, clone to `3rdparty/bonxai/`. `bonxai_demo` and `comparison` are built only when the headers are present.

### Local

```bash
# Optional: clone Bonxai headers first
git clone https://github.com/facontidavide/Bonxai.git /tmp/bonxai
mkdir -p 3rdparty/bonxai
cp -r /tmp/bonxai/bonxai_core/include/bonxai/* 3rdparty/bonxai/

mkdir build && cd build
cmake ..
make -j4
```

### Docker

```bash
docker build . -t slam_zero_to_hero:part2_ch03_08
```

---

## Run

### Local

```bash
# PCL Octree spatial search demo
./build/octree_demo cloud.pcd

# OctoMap probabilistic mapping
./build/octomap_demo /path/to/lidar/sequence/

# OctoMap ray-cast navigation check
./build/octomap_navigation map.bt start_x start_y start_z end_x end_y end_z

# Bonxai voxel grid demo (if built)
./build/bonxai_demo cloud.pcd

# Compare all three methods (if built)
./build/comparison cloud.pcd
```

### Docker

```bash
docker run -it --rm \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:part2_ch03_08
```

---

## References

- [OctoMap](https://github.com/OctoMap/octomap)
- [Bonxai](https://github.com/facontidavide/Bonxai)
- [PCL Octree API](https://pcl.readthedocs.io/projects/tutorials/en/latest/octree.html)
