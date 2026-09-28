"""Print a minimal .rviz config (grid + axes, docks hidden) using a given view class.

    make_config.py <1|2> <ViewClass>
"""
import sys

import yaml

ros, view_class = sys.argv[1], sys.argv[2]
ns = "rviz" if ros == "1" else "rviz_default_plugins"

if view_class.endswith("TopDownOrtho"):
    view = {"Angle": 0.0, "Scale": 40.0, "X": 0.0, "Y": 0.0}
else:
    view = {"Distance": 15.0, "Focal Point": {"X": 0.0, "Y": 0.0, "Z": 0.0},
            "Pitch": 0.6, "Yaw": 0.8}
view.update({"Class": view_class, "Name": "Current View", "Near Clip Distance": 0.01,
             "Target Frame": "base_link", "Value": view_class})

cfg = {
    "Panels": [],
    "Visualization Manager": {
        "Class": "",
        "Displays": [
            {"Class": f"{ns}/Grid", "Name": "Grid", "Enabled": True, "Value": True,
             "Plane Cell Count": 20, "Cell Size": 1, "Reference Frame": "<Fixed Frame>"},
            {"Class": f"{ns}/Axes", "Name": "Axes", "Enabled": True, "Value": True,
             "Length": 3, "Radius": 0.1, "Reference Frame": "base_link"},
        ],
        "Enabled": True,
        "Global Options": {"Background Color": "48; 48; 48", "Fixed Frame": "map",
                           "Frame Rate": 30},
        "Name": "root",
        "Tools": [{"Class": f"{ns}/MoveCamera"}],
        "Value": True,
        "Views": {"Current": view, "Saved": None},
    },
    "Window Geometry": {"Height": 800, "Width": 1280, "X": 0, "Y": 0,
                        "Hide Left Dock": True, "Hide Right Dock": True,
                        "Displays": {"collapsed": True}, "Views": {"collapsed": True}},
}
yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
