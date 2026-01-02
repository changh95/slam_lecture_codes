# Camera Calibration with OpenCV

This tutorial covers camera calibration techniques using OpenCV-Python.

## Topics Covered

- Checkerboard corner detection
- Monocular camera calibration
- Distortion correction (undistortion)
- Stereo camera calibration
- Stereo rectification
- Fisheye camera calibration

## Installation

### Using Virtual Environment

```bash
cd 1_29
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Examples

```bash
cd examples
python3 01_checkerboard_detection.py
python3 02_camera_calibration.py
python3 03_undistortion.py
python3 04_stereo_calibration.py
python3 05_stereo_rectification.py
python3 06_fisheye_calibration.py
```

## Calibration Data

- Place your calibration images in `data/checkerboard_images/`
- A sample checkerboard pattern can be printed from any OpenCV tutorial
- Recommended: 9x6 inner corners checkerboard with 25mm squares

## Examples

| File | Description |
|------|-------------|
| `01_checkerboard_detection.py` | Detect corners in calibration pattern |
| `02_camera_calibration.py` | Estimate camera intrinsics and distortion |
| `03_undistortion.py` | Remove lens distortion from images |
| `04_stereo_calibration.py` | Calibrate stereo camera pair |
| `05_stereo_rectification.py` | Rectify stereo images for disparity |
| `06_fisheye_calibration.py` | Calibrate fisheye/wide-angle lenses |

## Why Calibration for SLAM?

Accurate camera calibration is essential for SLAM because:

1. **Feature Localization**: Distortion causes features to be in wrong positions
2. **Epipolar Geometry**: Essential/Fundamental matrix estimation requires calibrated cameras
3. **Depth Estimation**: Stereo SLAM needs accurate baseline and intrinsics
4. **Scale Recovery**: Camera-IMU calibration enables metric scale

## Calibration Tips

1. **Use 10-20 calibration images minimum**
2. **Cover the entire field of view**
3. **Vary distance and angle**
4. **Check reprojection error (< 0.5 pixel is good)**
5. **Use a flat, rigid calibration board**

## References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Fisheye Camera Model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [LearnOpenCV Camera Calibration](https://learnopencv.com/camera-calibration-using-opencv/)
