# Basalt VIO

Visual-Inertial Odometry and Mapping from TUM.

- **Repo**: https://gitlab.com/VladyslavUsenko/basalt
- **Sensors**: Stereo + IMU, Monocular
- **GPU**: Not required

## Build

```bash
docker build -t slam:basalt_vio .
```

## Run

```bash
docker run -it --rm \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/euroc:/data \
    slam:basalt_vio bash
```

Inside the container:
```bash
cd /basalt/build
./basalt_vio --dataset-path /data/MH_01_easy --cam-calib /basalt/data/euroc_ds_calib.json --dataset-type euroc --config-path /basalt/data/euroc_config.json
```

## Dataset

- **EuRoC MAV** (recommended)
- **TUM-VI**
