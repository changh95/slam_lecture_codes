# Kalibr: Camera-IMU Calibration

This tutorial demonstrates how to use [Kalibr](https://github.com/ethz-asl/kalibr) for camera and camera-IMU calibration, along with [Allan Variance ROS](https://github.com/ori-drs/allan_variance_ros) for IMU noise characterization.

Kalibr is widely used in visual-inertial SLAM systems for obtaining accurate intrinsic and extrinsic calibration parameters.

---

## What is Kalibr?

Kalibr is a calibration toolbox developed by ETH Zurich that provides:

1. **Camera Intrinsic Calibration**: Focal length, principal point, distortion
2. **Camera-Camera Extrinsic Calibration**: Multi-camera systems
3. **Camera-IMU Extrinsic Calibration**: Transformation between camera and IMU
4. **IMU Intrinsic Calibration**: Noise parameters (with Allan Variance)

---

## How to Build

```bash
docker build . -t kalibr
```

---

## How to Run

### Start Container with GUI Support

```bash
# Allow X11 connections
xhost +local:root

# Run container
docker run -it \
  --net=host \
  --ipc=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --volume=$(pwd)/data:/data \
  --privileged \
  --env="XAUTHORITY=/root/.Xauthority" \
  kalibr

# Inside container
source /catkin_ws/devel/setup.bash
```

---

## Calibration Workflow

### 1. Prepare Calibration Target

Kalibr supports several calibration targets:

| Target | Description |
|--------|-------------|
| Checkerboard | Traditional grid pattern |
| AprilGrid | AprilTag-based grid (recommended) |
| Circlegrid | Circular pattern |

Download AprilGrid PDF from: [Kalibr Wiki](https://github.com/ethz-asl/kalibr/wiki/downloads)

Create a target YAML file (`april_6x6.yaml`):
```yaml
target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.024        # Tag size in meters
tagSpacing: 0.3       # Ratio of spacing to tag size
```

### 2. Record Calibration Data

Record a rosbag with camera images:

```bash
rosbag record /camera/image_raw
```

**Tips for good calibration data:**
- Move target slowly through the entire field of view
- Include various angles and distances
- Ensure good lighting with no motion blur
- Record for 60-120 seconds

### 3. Camera Intrinsic Calibration

```bash
rosrun kalibr kalibr_calibrate_cameras \
  --bag /data/camera_calib.bag \
  --topics /camera/image_raw \
  --models pinhole-radtan \
  --target /data/april_6x6.yaml
```

Camera models available:
| Model | Description |
|-------|-------------|
| `pinhole-radtan` | Pinhole + radial-tangential distortion |
| `pinhole-equi` | Pinhole + equidistant distortion |
| `omni-radtan` | Omnidirectional + radial-tangential |
| `ds-none` | Double sphere (fisheye) |

### 4. Camera-IMU Calibration

First, create an IMU configuration file (`imu.yaml`):
```yaml
rostopic: /imu/data
update_rate: 200.0   # Hz

accelerometer_noise_density: 0.01     # m/s^2/sqrt(Hz)
accelerometer_random_walk: 0.0002     # m/s^3/sqrt(Hz)
gyroscope_noise_density: 0.0001       # rad/s/sqrt(Hz)
gyroscope_random_walk: 0.00002        # rad/s^2/sqrt(Hz)
```

Run calibration:
```bash
rosrun kalibr kalibr_calibrate_imu_camera \
  --bag /data/imu_camera_calib.bag \
  --cam /data/camchain.yaml \
  --imu /data/imu.yaml \
  --target /data/april_6x6.yaml
```

### 5. IMU Noise Characterization (Allan Variance)

For accurate IMU parameters, use Allan Variance analysis:

```bash
# Record static IMU data (keep IMU stationary for ~3 hours)
rosbag record /imu/data -O imu_static.bag

# Run Allan Variance analysis
rosrun allan_variance_ros allan_variance /data/imu_static.bag /imu/data
```

---

## Output Files

Kalibr produces several output files:

| File | Description |
|------|-------------|
| `camchain.yaml` | Camera intrinsics and extrinsics |
| `imu_camera.yaml` | IMU-camera transformation |
| `results-cam.pdf` | Calibration report with reprojection errors |
| `results-imu-cam.pdf` | IMU-camera calibration report |

### Example camchain.yaml

```yaml
cam0:
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [-0.2834, 0.0846, 0.0001, -0.0002]
  distortion_model: radtan
  intrinsics: [461.63, 460.15, 362.68, 246.35]
  resolution: [752, 480]
  rostopic: /camera/image_raw

cam1:
  T_cn_cnm1:
    - [0.99997, 0.00012, 0.00763, -0.11003]
    - [-0.00011, 0.99999, -0.00048, 0.00016]
    - [-0.00763, 0.00047, 0.99997, 0.00052]
    - [0.0, 0.0, 0.0, 1.0]
  # ... similar to cam0
```

---

## Sample Datasets

### RPNG AR Table Dataset

Download from: [RPNG Datasets](https://udel.edu/~rpng/)

Contains:
- Camera images for calibration
- Static IMU data for Allan Variance
- AprilGrid YAML configuration

---

## Common Issues

**Problem**: Calibration fails with "Not enough corners detected"
**Solution**: Improve lighting, reduce motion blur, ensure target is fully visible

**Problem**: High reprojection error (>1 pixel)
**Solution**: Use more calibration data, check for lens damage, ensure target is flat

**Problem**: IMU-camera calibration diverges
**Solution**: Check IMU noise parameters, ensure time synchronization

---

## Integration with SLAM

Use the calibration results in your SLAM system:

```yaml
# Example for ORB-SLAM3
Camera.fx: 461.63
Camera.fy: 460.15
Camera.cx: 362.68
Camera.cy: 246.35

Camera.k1: -0.2834
Camera.k2: 0.0846
Camera.p1: 0.0001
Camera.p2: -0.0002

# IMU-Camera transformation
Tbc: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [0.0148, -0.9999, 0.0016, 0.0652,
         0.9999, 0.0148, 0.0013, -0.0207,
        -0.0012, 0.0017, 0.9999, -0.0080,
         0, 0, 0, 1]
```

---

## References

- [Kalibr Wiki](https://github.com/ethz-asl/kalibr/wiki)
- [Kalibr Paper](https://ieeexplore.ieee.org/document/6906596)
- [Allan Variance](https://en.wikipedia.org/wiki/Allan_variance)
