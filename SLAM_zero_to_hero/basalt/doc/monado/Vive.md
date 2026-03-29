# Valve Index, HTC Vive, and other lighthouse-tracked headsets

When running Monado set the environment variables `VIVE_SLAM=1` and
`XRT_DEBUG_GUI=1`. You will need to manually press the "Submit to SLAM"
checkbox in the "SLAM Tracker" box on the debug gui.
It's not enabled by default due to some timestamps initialization problem I've
not managed to fix.

The camera orientation provided by the Index factory calibration seems to be a
bit off (or we just can't get to parse it correctly). So you'll need to manually
calibrate the headset if you want a better experience. Right now there are some
placeholder values (that come from a calibration I did on a specific headset)
that should work, but they will likely be subpar for your setup.

Also, remember we have community-made drivers, we are not sure whether the
timestamps we are using are correct, so that might be another source of
inaccuracy.

All in all, it is still usable. Try it out, it should work out of the box, and
only if you want a better experience, proceed with the calibration procedure
explained below.

# Calibration

The most annoying bit is that you will need to manually calibrate the headset to
get a json file for Basalt like the [index_calib.json](/data/index_calib.json)
example. You can use that example first to see how it works for you by setting
`SLAM_CONFIG=$bsltdeps/basalt/data/monado/index.toml`, but that will likely not
work very well.

The general calibration procedure is as follows:

1. Setup the calibration target:
    1. Download [the one from kalibr](https://drive.google.com/file/d/1DqKWgePodCpAKJCd_Bz-hfiEQOSnn_k0/view)
    2. Open the pdf file and measure with a ruler the black square side (e.g., 24 inch 1080p screen, 34% zoom in Okular viewer, gives 3cm)
    3. Update aprlgrid_6x6.json "tagSize" property in meters (e.g., 3cm would be "0.03")
2. Record camera calibration sequence with the euroc recorder in Monado:
    - See calib-cam3 example from TUM-VI https://vision.in.tum.de/data/datasets/visual-inertial-dataset
    - Or this example from ORB-SLAM3: https://www.youtube.com/watch?v=R_K9-O4ool8
    - Note that for stereo calibration you want to cover as much as possible of
      both cameras images but always trying to keep >90% of the calibration target
      visible in both views. This can be a little tricky so practice it beforehand.
3. Record camera-imu calibration sequence: (faster motions)
    - Similar recommendations as in the previous sequence but this time we want
      faster motions that excite all IMU axes, see these examples:
    - calib-imu1 from TUM-VI https://vision.in.tum.de/data/datasets/visual-inertial-dataset
    - Or this from ORB-SLAM3 https://www.youtube.com/watch?v=4XkivVLw5k4
4. Now run camera calibration as explained [here](https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/doc/Calibration.md#camera-calibration)
5. Then camera+imu (no mocap) calibration as explained [here](https://gitlab.freedesktop.org/mateosss/basalt/-/blob/main/doc/Calibration.md#camera-imu-mocap-calibration)
6. If the datasets are not good, the calibration will likely not be very good as
   well. If you want to be sure you calibrated things properly, a good idea is
   to repeat this process until you get similar results.
