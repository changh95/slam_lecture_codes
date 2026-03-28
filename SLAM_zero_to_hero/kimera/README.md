# Kimera VIO

Visual-Inertial Odometry from MIT SPARK Lab. Produces metric-semantic 3D meshes.

- **Repo**: https://github.com/MIT-SPARK/Kimera-VIO
- **Sensors**: Stereo + IMU
- **GPU**: Not required

## Build

```bash
docker build -t slam:kimera .
```

## Run

```bash
docker run -it --rm \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/euroc:/data \
    slam:kimera bash
```

## Dataset

- **EuRoC MAV** (recommended)
