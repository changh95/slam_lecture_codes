# Kalibr and Allan Variance ROS

## How to build

```
docker build . -t kalibr
```

## How to run

Kalibr
```
xhost +local:root

docker run -it \
  --net=host \
  --ipc=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --volume=/home/$USER/vio_data:/data \
  --privileged \
  --env="XAUTHORITY=/root/.Xauthority" \
  kalibr
  
# Inside docker container
source ./devel/setup.bash
```

## Dataset

We use the [TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset).
It is served over plain HTTP by TUM, ships as native rosbags (no conversion
needed), and covers both halves of this exercise from the *same* sensor head:
Kalibr camera / camera-IMU calibration, and a long static IMU recording for
Allan Variance ROS.

> The AR Table dataset from RPNG that this chapter used to point at is gone --
> all three Google Drive links returned HTTP 404. The entire Kalibr wiki
> "Downloads" set (sample bags, target yamls, IMU yaml) is dead the same way, so
> the target and IMU configs live in `config/` here instead. See
> [Dead links](#dead-links-for-reference) below.

Sensor: stereo fisheye at 1024x1024 / 20 Hz, IMU at 200 Hz, hardware
time-synchronised. Topics are `/cam0/image_raw`, `/cam1/image_raw`, `/imu0`.
The 512x512 variant is used below -- same calibration, much faster to process.

Download into `~/vio_data` (the host directory mounted at `/data` in the
`docker run` above), and copy this chapter's configs in alongside them so they
are visible inside the container:

```bash
mkdir -p ~/vio_data
cp -r config ~/vio_data/          # from this chapter's directory
cd ~/vio_data

# Camera calibration: slow motion, well-exposed target (~467 MB)
wget https://vision.in.tum.de/tumvi/calibrated/512_16/dataset-calib-cam1_512_16.bag

# Camera-IMU calibration: rapid motion exciting all 6 DoF (~1.1 GB)
wget https://vision.in.tum.de/tumvi/calibrated/512_16/dataset-calib-imu1_512_16.bag
```

Sequences `calib-cam1..8` and `calib-imu1..4` are all available at
[that directory index](https://vision.in.tum.de/tumvi/calibrated/512_16/) if you
want a second run to compare against. Each bag has a `.md5` alongside it.

## Kalibr

The cameras are fisheye, so use the equidistant (Kanpp-Brandt) model
`pinhole-equi`, not `pinhole-radtan`.

Camera intrinsics + stereo extrinsics:

```bash
rosrun kalibr kalibr_calibrate_cameras \
  --bag /data/dataset-calib-cam1_512_16.bag \
  --topics /cam0/image_raw /cam1/image_raw \
  --models pinhole-equi pinhole-equi \
  --target /data/config/aprilgrid_6x6_80x80cm.yaml
```

Camera-IMU extrinsics, feeding in the camchain produced above:

```bash
rosrun kalibr kalibr_calibrate_imu_camera \
  --bag /data/dataset-calib-imu1_512_16.bag \
  --cam /data/dataset-calib-cam1_512_16-camchain.yaml \
  --imu /data/config/imu_tumvi.yaml \
  --target /data/config/aprilgrid_6x6_80x80cm.yaml
```

Sanity check: the cam0-cam1 baseline should come out around 10 cm. If it is off
by a constant factor, `tagSize` in the target yaml does not match the physical
target.

## Allan Variance ROS

`dataset-calib-imu-static2.bag` is the sensor sitting still for **111 hours**
(IMU only, no images). Note this is a **29.6 GB** download:

```bash
cd ~/vio_data
wget https://vision.in.tum.de/tumvi/imu_static/dataset-calib-imu-static2.bag
```

```bash
# Reorder messages by timestamp first
rosrun allan_variance_ros cookbag.py \
  --input /data/dataset-calib-imu-static2.bag \
  --output /data/static_cooked.bag

rosrun allan_variance_ros allan_variance \
  /data /data/config/allan_variance_tumvi.yaml

rosrun allan_variance_ros analysis.py \
  --data allan_variance.csv \
  --config /data/config/allan_variance_tumvi.yaml
```

This is the useful part of the exercise: `analysis.py` writes an `imu.yaml`, and
TUM publishes what they got off the same bag, so you have a reference answer to
check your own numbers against:

| Parameter | TUM published value |
| --- | --- |
| `accelerometer_noise_density` | 0.0014 m/s^1.5 |
| `accelerometer_random_walk` | 0.000086 m/s^2.5 |
| `gyroscope_noise_density` | 0.000080 rad/s^0.5 |
| `gyroscope_random_walk` | 0.0000022 rad/s^1.5 |

`config/imu_tumvi.yaml` carries these inflated (white noise x2, bias random walk
x10) because that is what actually converges well in Kalibr -- see the comments
in that file.

## Configs in this repo

| File | Purpose |
| --- | --- |
| [`config/aprilgrid_6x6_80x80cm.yaml`](config/aprilgrid_6x6_80x80cm.yaml) | Kalibr target: standard 6x6 A0 Aprilgrid |
| [`config/imu_tumvi.yaml`](config/imu_tumvi.yaml) | Kalibr IMU noise model for the TUM VI sensor |
| [`config/allan_variance_tumvi.yaml`](config/allan_variance_tumvi.yaml) | allan_variance_ros config for the static bag |

## Alternative dataset

If you want a smaller static-IMU download, or want everything from one host, the
[Monado SLAM Datasets](https://huggingface.co/datasets/collabora/monado-slam-datasets)
are on Hugging Face (resumable, least likely to rot). Under
`M_monado_datasets/MO_odyssey_plus/MOC_calibration/`:

- `MOC01_camcalib_1.zip` (432 MB), `MOC02_imucalib_1.zip` (344 MB)
- `MOC13_imustatic.zip` (3.3 GB) -- 48-hour static recording
- `aprilgrid_6x6.json` -- note `tagSize: 0.03`, a much smaller target than TUM's

Caveat: these are in EuRoC ASL format (zipped `mav0/` folders), not rosbags, so
they need `kalibr_bagcreater` before Kalibr will read them.

## Dead links (for reference)

Verified 404 as of 2026-07, kept so nobody re-adds them:

- AR Table dataset (RPNG, U. Delaware) -- images, static IMU, aprilgrid yaml.
  Reported upstream; no fix yet.
- The whole Kalibr wiki [Downloads](https://github.com/ethz-asl/kalibr/wiki/downloads)
  set -- sample bags, all target yamls, ADIS16448 IMU yaml. The Google Drive
  account behind them appears to be gone, not just individual files.
- The `allan_variance_ros` "additional sensor rosbags" Drive folder.
  Its [Realsense D435i 3-hour log](https://drive.google.com/file/d/1ovI2NvYR52Axt-KuRs5HjVk7-57ky72H/view?usp=sharing)
  is still alive, though.

EuRoC MAV calibration bags are not dead exactly, but `robotics.ethz.ch` hangs
without responding and the ASL dataset landing pages redirect in a loop, so
treat that source as unavailable.
