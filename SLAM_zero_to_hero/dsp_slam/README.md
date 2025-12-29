# DSP-SLAM

## Dataset & weights

Download the dataset [here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabjw4_ucl_ac_uk/Eh3nHv6D-LZHkuny4iNOexQBGdDVxloM_nwbEZdxeRfStw?e=sYO1Ot).

Extract them in the same folder as Dockerfile.

## How to run
```
docker build -t slam:dsp_slam
docker run -it -e "DISPLAY=$DISPLAY" -e "QT_X11_NO_MITSHM=1" -e "XAUTHORITY=$XAUTH" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v ${PWD}/:/data --runtime=nvidia --net=host --ipc=host --privileged slam:dsp_slam

# Build additional dependencies inside docker container (It's good to commit the container as image once this stage is done)
./build_cuda113.sh --build-dependencies --create-conda-env

# Run
mkdir map
vim configs/config_kitti.json (See below)

./dsp_slam Vocabulary/ORBvoc.bin ./configs/KITTI04-12.yaml /data/07 map

```
## config_kitti.json

```
{
  "data_type": "KITTI",
  "detect_online": true,
  "path_label_3d": "/data/07/labels/pointpillars_labels",
  "path_label_2d": "/data/07/labels/maskrcnn_labels",
  "Detector3D": {
    "config_path": "configs/config_pointpillars.py",
    "weight_path": "/data/weights/pointpillars/model.pth"
  },
  "Detector2D": {
    "config_path": "configs/config_maskrcnn.py",
    "weight_path": "/data/weights/maskrcnn/model.pth"
  },
  "min_bb_area": 1600,
  "min_mask_area": 1000,
  "downsample_ratio": 4.0,
  "num_lidar_max": 250,
  "num_lidar_min": 10,
  "DeepSDF_DIR": "/data/weights/deepsdf/cars_64",
  "voxels_dim": 32,
  "optimizer": {
    "code_len": 64,
    "num_depth_samples": 50,
    "cut_off_threshold": 0.01,
    "joint_optim": {
      "k1": 1.0,
      "k2": 100.0,
      "k3": 0.25,
      "k4": 1e7,
      "b1": 0.20,
      "b2": 0.025,
      "num_iterations": 10,
      "learning_rate": 1.0,
      "scale_damping": 1.0
    },
    "pose_only_optim": {
      "num_iterations": 5,
      "learning_rate": 1.0
    }
  },
  "viewer": {
    "distance": 150.0,
    "tilt": 45.0,
    "frame_size": 10.0
  }
}

```

## Output

![](./output.gif)
