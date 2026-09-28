"""Check what each gesture changed, against the unified or the stock mouse scheme.

    check_views.py <ViewClass> <views.jsonl> <rviz_errors.log>

Each gesture moves one of three quantities:
    rotate = Yaw/Pitch (Angle for TopDownOrtho)
    pan    = Focal Point (X/Y for TopDownOrtho)
    zoom   = Distance (Scale for TopDownOrtho)
"""
import json
import sys

view_class, views_file, err_file = sys.argv[1:4]
steps = [json.loads(line) for line in open(views_file)]
ortho = view_class.endswith("TopDownOrtho")
groups = ({"rotate": ["Angle"], "pan": ["X", "Y"], "zoom": ["Scale"]} if ortho else
          {"rotate": ["Yaw", "Pitch"], "pan": ["FX", "FY", "FZ"], "zoom": ["Distance"]})

unified = view_class.startswith("slam_zero_to_hero/")
expect = {
    "right_drag": "pan" if unified else "zoom",
    "wheel_up": "zoom",
    "left_drag": "rotate",
    "middle_drag": "pan",
}

ok = True
errors = open(err_file).read().strip()
if errors:
    print("rviz log errors:\n" + errors)
    ok = False
if steps and steps[0].get("class") != view_class:
    print(f"saved view class is {steps[0].get('class')!r}, expected {view_class!r}")
    ok = False

print(f"{'gesture':12s} {'expected':8s} {'moved':18s} result")
for prev, cur in zip(steps, steps[1:]):
    moved = [g for g, keys in groups.items()
             if any(abs(cur.get(k, 0) - prev.get(k, 0)) > 1e-4 for k in keys)]
    want = expect[cur["step"]]
    good = moved == [want]
    ok &= good
    deltas = {k: round(cur.get(k, 0) - prev.get(k, 0), 4)
              for keys in groups.values() for k in keys}
    print(f"{cur['step']:12s} {want:8s} {','.join(moved) or '-':18s} "
          f"{'PASS' if good else 'FAIL'}  {deltas}")
print("OVERALL", "PASS" if ok else "FAIL", view_class)
sys.exit(0 if ok else 1)
