# KISS-SLAM

LiDAR SLAM based on KISS-ICP. Simple, robust, and effective.

- **Repo**: https://github.com/PRBonn/kiss-slam
- **Sensors**: LiDAR
- **GPU**: Not required

## Build

```bash
docker build -t slam:kiss_slam .
```

## Run

```bash
docker run -it --rm \
    -v /path/to/kitti:/data \
    slam:kiss_slam bash
```

Inside the container:
```bash
kiss_slam_pipeline --dataloader kitti --sequence /data/00
```

## Dataset

- **KITTI** (already available)
