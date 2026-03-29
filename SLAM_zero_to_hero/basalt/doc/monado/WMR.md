# Windows Mixed Reality Headsets

Monado should work out of the box with WMR devices and Basalt without any input
on your part. So if you successfully followed the guide in the main README and
Monado detects your WMR headset, then tracking should already be working for you.

If it does not work, double check that you read everything in the guide, from
top to bottom.

If it still doesn't work, triple check it.

Now if you are still experiencing issues, crashes or would like to debug the
pipeline for whatever reason, the rest of this document should help you with
that.

## Making a custom Basalt config file

It's a good idea to make a Basalt config file for your headset so that you can
easily tweak it if needed. Let's say we are trying to make a config file for a
Reverb G2.

First, let's get your WMR device json config block. To get that json, set the
environment variable `WMR_LOG=debug` and run Monado with your WMR headset connected.
The headset json is printed on start after the line `DEBUG [wmr_read_config] JSON config:`.
Copy that to a file called `reverbg2_wmrcalib.json`.

Now let's convert this WMR json to a Basalt calibration file with:

```bash
$bsltdeps/basalt/data/monado/wmr-tools/wmr2bslt_calib.py reverbg2_wmrcalib.json > $bsltdeps/basalt/data/reverbg2_calib.json
```

Finally, we'll need to create the main config file for Basalt that references
this calibration file we just created. For that let's copy the config that is
already present for the Odyssey+:

```bash
cp $bsltdeps/basalt/data/monado/odysseyplus_rt8.toml $bsltdeps/basalt/data/monado/reverbg2.toml
```

And edit the `cam-calib` field in the `reverbg2.toml` file to point to your
`reverbg2_calib.json` file.

That's it! now you have a Basalt config file that you can use for your headset.

## Set Monado options

Let's set a couple environment variables in Monado that will help us debug the
SLAM pipeline.

- `SLAM_CONFIG=$bsltdeps/basalt/data/monado/reverbg2.toml`: Tell Monado where
  the Basalt `toml` config you just created is. Notice that the `show-gui`
  property is enabled in this `toml` file so you will start seeing the Basalt
  visualizer when opening Monado. Furthermore the `config-path` key points to a
  Basalt specific config file for tweaking the VIO pipeline.

- `XRT_DEBUG_GUI=on`: Enable Monado's own debug GUI.

- `SLAM_SUBMIT_FROM_START=off`: Do not send frames to Basalt from the start,
  rather wait until we check the checkbox in the Monado GUI box called "SLAM
  Tracker".

- `WMR_AUTOEXPOSURE=off`: Disable autoexposure to have one less moving part, we
  will manually adjust it instead on the "WMR Camera" box, by moving the
  "Brightness" slider on the "Auto exposure and gain control" section.

## Controling auto exposure

By default, the UI box `SLAM Tracker` has the option `Submit data to SLAM`
disabled so that you first manually configure the exposure and gain values in
the `WMR Camera` box. You can enable it yourself in the UI or enable it at start
by setting the environment variable `SLAM_SUBMIT_FROM_START=true`.

## Recalibrating my device (TODO)

It's not a bad idea to recalibrate your headset manually with the tools Basalt provides.

TODO: Specify better the steps, but roughly they would be:

1. Get calibration target from Kalibr: https://github.com/ethz-asl/kalibr/wiki/downloads
2. Open the pdf in a flat monitor, measure dimensions with a ruler and put them on aprilgrid_6x6.json
3. Record an EuRoC dataset from Monado in which you move the headset around the target (link an example sequence)
4. Run
   [basalt_calibrate](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md#camera-calibration)
   on
   [euroc](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md#euroc-dataset).
5. Run
   [basalt_calibrate_vio](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md#camera-imu-mocap-calibration)
   on
   [euroc](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md#camera-imu-calibration).

# Video Walkthrough (DEPRECATED)

_This video is not up to date anymore but might be useful to see how things
worked before. Now, `view_offset` is automatically computed, exposure and gain
are automatically set too, so in general there is no manual input needed from
the user._

~~Here is a 15 minute walkthrough with some tips for using a WMR headset with Monado and Basalt that should help complement the guide found in the [README.md](README.md) file: <https://www.youtube.com/watch?v=jyQKjyRVMS4>~~
