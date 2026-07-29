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

[TUM Visual-Inertial Dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
- stereo fisheye + 200 Hz IMU, native rosbags. Topics are `/cam0/image_raw`,
`/cam1/image_raw`, `/imu0`.

```bash
mkdir -p ~/vio_data && cp -r config ~/vio_data/ && cd ~/vio_data

# Camera calibration (467 MB)
wget https://vision.in.tum.de/tumvi/calibrated/512_16/dataset-calib-cam1_512_16.bag

# Camera-IMU calibration (1.1 GB)
wget https://vision.in.tum.de/tumvi/calibrated/512_16/dataset-calib-imu1_512_16.bag

# Static IMU, 111 hours, for Allan variance (29.6 GB)
wget https://vision.in.tum.de/tumvi/imu_static/dataset-calib-imu-static2.bag
```

The AR Table (RPNG) links this chapter used before are dead, as is the whole
Kalibr wiki Downloads set, so the target and IMU configs live in `config/` here.

## Kalibr

Use `ds-none` (double sphere), **not** `pinhole-equi` - these are 195 degree
fisheye lenses and equidistant diverges without converging.

```bash
rosrun kalibr kalibr_calibrate_cameras \
  --bag /data/dataset-calib-cam1_512_16.bag \
  --topics /cam0/image_raw /cam1/image_raw \
  --models ds-none ds-none \
  --target /data/config/aprilgrid_6x6_80x80cm.yaml

rosrun kalibr kalibr_calibrate_imu_camera \
  --bag /data/dataset-calib-imu1_512_16.bag \
  --cam /data/dataset-calib-cam1_512_16-camchain.yaml \
  --imu /data/config/imu_tumvi.yaml \
  --target /data/config/aprilgrid_6x6_80x80cm.yaml
```

Sanity check: baseline should be ~0.101 m (TUM publish ~10.1 cm) and gravity
~9.807 m/s^2. A baseline off by a constant factor means `tagSize` does not match
the physical target.

## Allan Variance ROS

```bash
roscore &

rosrun allan_variance_ros cookbag.py \
  --input /data/avr/static.bag \
  --output /data/avr_cooked/static_cooked.bag

rosrun allan_variance_ros allan_variance \
  /data/avr_cooked /data/config/allan_variance_tumvi.yaml

MPLBACKEND=Agg rosrun allan_variance_ros analysis.py \
  --data /data/avr_cooked/allan_variance.csv \
  --config /data/config/allan_variance_tumvi.yaml
```

Compare the generated `imu.yaml` against what TUM published for this bag:

| Parameter | TUM (111 h) |
| --- | --- |
| `accelerometer_noise_density` | 0.0014 m/s^1.5 |
| `accelerometer_random_walk` | 0.000086 m/s^2.5 |
| `gyroscope_noise_density` | 0.000080 rad/s^0.5 |
| `gyroscope_random_walk` | 0.0000022 rad/s^1.5 |

3 hours of data gets within ~1.5x of these. Note `analysis.py` reports gyro in
degrees while TUM publishes radians.

## Gotchas

- Kalibr writes results **next to the bag**, not the working directory, and
  **exits 0 even when it fails**. Check the `-camchain.yaml` was created.
- `allan_variance` takes a **directory**, not a bag path, and picks up every bag
  it finds there. Keep the static bag in its own directory.
- To skip the 29.6 GB download: fetch a 1.3 GB range
  (`curl -r 0-1300000000`) and run `rosbag reindex static.bag`. That recovers
  4h46m, past the 3 hour minimum. Reindex keeps a `.orig.bag` backup, so budget
  2x disk.
- The Dockerfile's shell-form `ENTRYPOINT` ignores appended commands - use
  `--entrypoint /bin/bash` to run one directly.

## Configs

| File | Purpose |
| --- | --- |
| [`config/aprilgrid_6x6_80x80cm.yaml`](config/aprilgrid_6x6_80x80cm.yaml) | Kalibr target, same spec as the old RPNG one |
| [`config/imu_tumvi.yaml`](config/imu_tumvi.yaml) | Kalibr IMU noise model for this sensor |
| [`config/allan_variance_tumvi.yaml`](config/allan_variance_tumvi.yaml) | allan_variance_ros config |
