# Unified RViz mouse controls

Every RViz / RViz2 demo in this course uses the same mouse scheme:

| Mouse                  | Action                          | Stock RViz         |
|------------------------|---------------------------------|--------------------|
| Left drag              | Rotate the view (orbit)         | same               |
| Wheel                  | Zoom in / out                   | same               |
| Right drag             | Pan the camera                  | zoom               |
| Middle drag            | Pan the camera                  | same               |
| Shift + left drag      | Pan the camera                  | same               |

RViz has no setting for mouse bindings, so this folder provides view-controller
plugins. Each one is the stock controller with one change: the right button is
treated as the middle button before the stock `handleMouseEvent` runs. The
properties, the saved config keys, the focal-point marker and the wheel all
stay upstream code. The one stock gesture that goes away is Shift + right drag
(move along Z in Orbit).

## Plugins

The class names are the same for ROS 1 and ROS 2:

| Class (put this in the `.rviz` file)           | Stock controller it replaces                                            |
|------------------------------------------------|--------------------------------------------------------------------------|
| `slam_zero_to_hero/UnifiedOrbit`               | `rviz/Orbit`, `rviz_default_plugins/Orbit`                               |
| `slam_zero_to_hero/UnifiedThirdPersonFollower` | `rviz/ThirdPersonFollower`, `rviz_default_plugins/ThirdPersonFollower`   |
| `slam_zero_to_hero/UnifiedTopDownOrtho`        | `rviz/TopDownOrtho`, `rviz_default_plugins/TopDownOrtho`                 |

```
rviz_unified_controls/
├── ros1/            catkin package (noetic, rviz)
├── ros2/            ament package (humble, jazzy; rviz2)
├── install_ros1.sh  build + install into /opt/ros/$ROS_DISTRO
├── install_ros2.sh
├── Dockerfile       source-only image that demo Dockerfiles copy from
└── test/            test images and the Xvfb + xdotool mouse test
```

## Selecting the controller in a `.rviz` file

The controller comes from `Visualization Manager > Views > Current > Class`.
Change only that line. The other keys (`Distance`, `Focal Point`, `Yaw`,
`Pitch`, or `Scale`/`X`/`Y`/`Angle` for TopDownOrtho) keep their meaning:

```yaml
Visualization Manager:
  Views:
    Current:
      Class: slam_zero_to_hero/UnifiedOrbit   # was rviz/Orbit or rviz_default_plugins/Orbit
      Distance: 40
      Focal Point: {X: 0, Y: 0, Z: 0}
      ...
```

The image that runs the `.rviz` file must have the plugin installed (see
below). Otherwise rviz cannot load the class.

## Adding it to a demo Dockerfile

Demo images build with their own folder as the build context, so the plugin
comes from a small source-only image rather than a `COPY ../`. Build it once:

```bash
cd SLAM_zero_to_hero
podman build -t slam_zero_to_hero:rviz_unified_controls rviz_unified_controls
```

Then, in the demo Dockerfile, after ROS and rviz are installed:

```dockerfile
# Course-wide RViz mouse scheme: left drag rotates, right/middle drag pans, wheel zooms.
COPY --from=localhost/slam_zero_to_hero:rviz_unified_controls /rviz_unified_controls /opt/rviz_unified_controls
RUN /opt/rviz_unified_controls/install_ros1.sh /opt/rviz_unified_controls
```

Rebuild the source image after you change anything in this folder. The demo
builds copy whatever the image holds at that moment.

For ROS 2 images, use `install_ros2.sh` instead. The script finds the distro
from `$ROS_DISTRO`, or from the only entry in `/opt/ros`.

The scripts install into `/opt/ros/$ROS_DISTRO` itself, next to rviz's own
plugins. Nothing extra has to be sourced, and the demo's workspace overlay and
entrypoint stay as they are. The build needs `cmake` and a C++ compiler, which
every ROS image here already has.

## Testing

`test/` holds a derived test image per distro, plus `mouse_test.sh`. The test
starts rviz on a private Xvfb, then makes real mouse input with xdotool: a
right drag, three wheel steps, a left drag and a middle drag. After each one it
saves the config with Ctrl+S and reads back `Views/Current`. The check is on
the controller's own numbers, not on pixels:

```bash
cd SLAM_zero_to_hero/rviz_unified_controls
podman build -t slam_zero_to_hero:rviz_unified_controls .                         # source image (the test images use the snippet above)
podman build -f test/Dockerfile.noetic -t localhost/rviz_controls_test:noetic .   # FROM slam_zero_to_hero:cartographer
podman build -f test/Dockerfile.humble -t localhost/rviz_controls_test:humble .   # FROM fast-livo2:humble
podman build -f test/Dockerfile.jazzy  -t localhost/rviz_controls_test:jazzy  .   # FROM slam_zero_to_hero:glim (+ rviz2)

for c in slam_zero_to_hero/UnifiedOrbit slam_zero_to_hero/UnifiedThirdPersonFollower \
         slam_zero_to_hero/UnifiedTopDownOrtho rviz/Orbit; do
  podman run --rm -v $PWD/test:/test:ro localhost/rviz_controls_test:noetic /test/mouse_test.sh $c
done
```

For a `slam_zero_to_hero/*` class, the test expects the right drag to move only
the focal point. For a stock class, it expects the right drag to move only the
distance. That second case is the control run: it shows the test can tell the
two schemes apart.
