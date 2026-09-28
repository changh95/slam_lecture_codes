"""Print Views/Current of a saved .rviz file as one JSON line tagged with a step name.

    read_view.py <config.rviz> <step>
"""
import json
import sys

import yaml

view = yaml.safe_load(open(sys.argv[1]))["Visualization Manager"]["Views"]["Current"]
fp = view.get("Focal Point", {})
out = {"step": sys.argv[2], "class": view.get("Class")}
for k in ("Distance", "Yaw", "Pitch", "Scale", "X", "Y", "Angle"):
    if k in view:
        out[k] = float(view[k])
if fp:
    out["FX"], out["FY"], out["FZ"] = (float(fp[a]) for a in "XYZ")
print(json.dumps(out))
