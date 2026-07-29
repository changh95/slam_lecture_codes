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

Sensor: stereo 195-degree fisheye at 1024x1024 / 20 Hz, IMU at 200 Hz, hardware
time-synchronised. Topics are `/cam0/image_raw`, `/cam1/image_raw`, `/imu0`.
The 512x512 variant is used below -- same calibration, much faster to process.

The *calibration* sequences were deliberately recorded at a lower frame rate than
the 20 Hz main sequences: `calib-cam1` holds 436 stereo pairs at roughly 4 Hz.
Worth knowing before you reach for `--bag-freq`.

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

The Dockerfile uses a shell-form `ENTRYPOINT`, so a command appended to
`docker run` is silently ignored. Override it to get a usable shell:

```bash
docker run --rm -it -v $HOME/vio_data:/data --entrypoint /bin/bash kalibr
# then, inside the container:
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
source /catkin_ws/devel/setup.bash
```

### Camera model: use `ds` (double sphere), not `pinhole-equi`

These are 195-degree fisheye lenses. `pinhole-equi` (equidistant /
Kannala-Brandt) **does not converge on this dataset** -- it initialises fine,
then diverges through all three attempts and gives up:

```
[ERROR] Did not converge in maxIterations... restarting...
[ WARN] Optimization diverged possibly due to a bad initialization.
[ERROR] Max. attemps reached... Giving up...
```

Use `ds-none` instead. Double sphere was designed by the same group for exactly
these cameras, and it converges cleanly. Note `--bag-freq` is not the fix here:
these calibration bags are already low-frame-rate, so subsampling to 4 Hz drops
only 9 of 436 frames.

> Kalibr **exits 0 even when it gives up**, and writes no files. Don't trust the
> exit code -- check that the `-camchain.yaml` was actually produced.

Camera intrinsics + stereo extrinsics:

```bash
rosrun kalibr kalibr_calibrate_cameras \
  --bag /data/dataset-calib-cam1_512_16.bag \
  --topics /cam0/image_raw /cam1/image_raw \
  --models ds-none ds-none \
  --target /data/config/aprilgrid_6x6_80x80cm.yaml \
  --dont-show-report
```

Results are written **next to the bag file**, not to the current working
directory -- so the above produces `/data/dataset-calib-cam1_512_16-camchain.yaml`
regardless of where you `cd`. Two runs on the same bag overwrite each other; copy
results aside if you want to compare models.

Drop `--dont-show-report` if you have working X11 forwarding; without a display
Kalibr prints harmless `Gdk-CRITICAL` warnings.

Camera-IMU extrinsics, feeding in the camchain produced above:

```bash
rosrun kalibr kalibr_calibrate_imu_camera \
  --bag /data/dataset-calib-imu1_512_16.bag \
  --cam /data/dataset-calib-cam1_512_16-camchain.yaml \
  --imu /data/config/imu_tumvi.yaml \
  --target /data/config/aprilgrid_6x6_80x80cm.yaml \
  --dont-show-report
```

### Expected results

`ds-none` on `dataset-calib-cam1_512_16.bag` (427 images, 79 selected by the
mutual-information gate) should reproduce roughly:

| Quantity | Value |
| --- | --- |
| cam0 reprojection error | +- 0.065 / 0.063 px |
| cam1 reprojection error | +- 0.067 / 0.070 px |
| cam0-cam1 baseline | 0.1010 m |

The baseline is the sanity check that matters: TUM publishes ~10.1 cm for this
sensor, so hitting 0.1010 m confirms both the camera model and that `tagSize` in
the target yaml matches the physical target. If the baseline is off by a constant
factor, `tagSize` is wrong -- intrinsics will still look fine, which is what
makes that failure mode easy to miss.

The camera-IMU stage on `dataset-calib-imu1_512_16.bag` should land near:

| Quantity | Value | Why it's a good check |
| --- | --- | --- |
| Reprojection error, cam0 / cam1 | 0.093 / 0.092 px mean | sub-0.1 px |
| Gyroscope error | 0.0012 rad/s | |
| Accelerometer error | 0.027 m/s^2 | |
| Normalized gyro / accel residual | 0.53 / 0.67 | both **< 1**, so the noise model isn't over-tight |
| Gravity vector norm | 9.807 m/s^2 | must come out as real gravity |
| `timeshift cam0 to imu0` | 0.00012 s | ~0.1 ms, matching TUM's hardware sync |
| Baseline norm | 0.10106 m | agrees with the camera-only stage |

The normalized residuals are the ones to read first. They are the errors divided
by the noise densities from `config/imu_tumvi.yaml`, so values below 1 mean that
IMU noise model is honest for this data. That is the concrete reason the chapter
uses TUM's inflated values rather than the raw Allan-plot numbers -- feeding in
the raw (10x smaller random walk) numbers pushes these residuals well above 1.

Gravity norm and the sub-millisecond time shift are free physical checks: neither
is constrained to be correct by the optimiser, so getting 9.807 m/s^2 and ~0.1 ms
out is good evidence the whole chain is right.

## Allan Variance ROS

`dataset-calib-imu-static2.bag` is the sensor sitting still for **111 hours**
(IMU only, no images). The catch is that it is a **29.6 GB** download.

### Getting only as much as you need

`allan_variance_ros` wants at least 3 hours. This bag runs about 267 MB/hour, so
a 1.3 GB prefix is enough. A truncated rosbag has no index, but `rosbag reindex`
rebuilds one from the intact chunks:

```bash
mkdir -p ~/vio_data/avr && cd ~/vio_data/avr
curl -L -r 0-1300000000 \
  https://vision.in.tum.de/tumvi/imu_static/dataset-calib-imu-static2.bag \
  -o static.bag

# inside the container
rosbag reindex static.bag
rosbag info static.bag     # -> duration: 4hr 46:55s, 3432400 msgs on /imu0
```

That yields 4h46m of usable static data for 1.3 GB instead of 29.6 GB. Grab the
whole bag if you want to match TUM's numbers more tightly -- see the accuracy
note below.

Budget 2x the disk: `rosbag reindex` writes the repaired bag in place and keeps
the original as `static.orig.bag`, so the 1.3 GB download occupies ~2.6 GB until
you delete the backup.

### Running it

```bash
roscore &                  # the C++ node wants a master

rosrun allan_variance_ros cookbag.py \
  --input /data/avr/static.bag \
  --output /data/avr_cooked/static_cooked.bag

rosrun allan_variance_ros allan_variance \
  /data/avr_cooked /data/config/allan_variance_tumvi.yaml

MPLBACKEND=Agg rosrun allan_variance_ros analysis.py \
  --data /data/avr_cooked/allan_variance.csv \
  --config /data/config/allan_variance_tumvi.yaml
```

Three things that will trip you up:

- `allan_variance` takes a **directory**, not a bag path, and picks up bags it
  finds there. Keep the static bag in a directory of its own -- point it at a
  folder holding the `calib-cam`/`calib-imu` bags and it will chew on those too.
- `analysis.py` plots, so set `MPLBACKEND=Agg` when running headless, or it dies
  on display init.
- `cookbag.py` rewrites 1.2 GB down to ~201 MB while keeping all 3432400
  messages. That shrinkage is normal, not data loss.

### Results, checked against TUM

Consuming 3 hours (`sequence_time: 10800` x `measure_rate: 100` = 1080000
measurements), computation takes ~106 s and the values land within 1.1-1.5x of
what TUM published from the full 111 hours:

| Parameter | From 3 h | TUM, 111 h | Ratio |
| --- | --- | --- | --- |
| `accelerometer_noise_density` | 0.00188 m/s^1.5 | 0.0014 | 1.34 |
| `accelerometer_random_walk` | 9.67e-05 m/s^2.5 | 0.000086 | 1.12 |
| `gyroscope_noise_density` | 0.000102 rad/s^0.5 | 0.000080 | 1.28 |
| `gyroscope_random_walk` | 3.2e-06 rad/s^1.5 | 0.0000022 | 1.45 |

Reproducing TUM's published numbers to within ~1.3x is the point of the
exercise -- it is a real check that your pipeline is correct, which a dataset
without published reference values cannot give you.

The residual overestimate is expected and worth discussing: the random-walk terms
are read off the **large**-tau end of the Allan curve, which is exactly where a
short record has the fewest independent bins. Hence `gyroscope_random_walk`
(1.45x) being the worst and `accelerometer_random_walk` (1.12x) the best. Use the
full 29.6 GB bag to close the gap.

Two conversion gotchas when comparing:

- `analysis.py` reports gyro terms in **degrees**; TUM publishes **radians**.
  Multiply by pi/180 before comparing, or the ratio looks like ~57x.
- The `imu.yaml` it writes takes the **worst axis**, not the mean -- it emits
  `accelerometer_noise_density: 0.00233` (the Z axis) where the 3-axis mean is
  0.00188. Deliberately conservative, but it makes hand-comparison confusing.

`config/imu_tumvi.yaml` carries TUM's values inflated (white noise x2, bias
random walk x10) because that is what actually converges well in Kalibr -- see
the comments in that file, and the normalized-residual check above.

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
