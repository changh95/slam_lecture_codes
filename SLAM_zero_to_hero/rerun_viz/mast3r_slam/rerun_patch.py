#!/usr/bin/env python3
"""Patch /MASt3R-SLAM/main.py to inject Rerun real-time visualization."""

MAIN_PY = "/MASt3R-SLAM/main.py"

with open(MAIN_PY) as f:
    src = f.read()

# 1) Add imports
import_block = """
import os as os
import rerun as rr
import rerun.blueprint as rrb
import numpy as _rr_np

_RR_TRAJECTORY = []
"""
src = src.replace("import yaml\n", "import yaml\n" + import_block, 1)

# 2) Rerun init — before tracker creation
rr_init = '''
    # --- Rerun init ---
    rr.init("mast3r_slam")
    _rr_ip = os.environ.get("SERVER_IP", "localhost")
    if os.environ.get("RERUN_WEB"):
        _srv = rr.serve_grpc(grpc_port=9876)
        _wc = f"rerun+http://{_rr_ip}:9876/proxy"
        rr.serve_web_viewer(web_port=9090, open_browser=False, connect_to=_wc)
        print(f"\\n>>> Rerun web viewer at http://{_rr_ip}:9090 <<<\\n", flush=True)
    elif os.environ.get("RERUN_RRD"):
        rr.save(os.environ["RERUN_RRD"])
    rr.send_blueprint(rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Vertical(row_shares=[0.7, 0.3], contents=[
            rrb.Spatial3DView(name="Map"),
            rrb.Spatial2DView(name="Camera", origin="camera/image"),
        ]),
    ))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
'''
src = src.replace("    tracker = FrameTracker(model, keyframes, device)\n",
                  rr_init + "    tracker = FrameTracker(model, keyframes, device)\n")

# 3) After image load, log the raw image
img_log = '''
        # --- Rerun: log frame ---
        rr.set_time("frame", sequence=i)
        try:
            _im = _rr_np.asarray(img)
            if _im.dtype != _rr_np.uint8:
                _im = (_rr_np.clip(_im * 255, 0, 255) if _im.max() <= 1.0 else _im).astype(_rr_np.uint8)
            if _im.ndim == 3:
                rr.log("camera/image", rr.Image(_im))
        except Exception as _e:
            pass
'''
src = src.replace("        if save_frames:\n            frames.append(img)\n",
                  img_log + "        if save_frames:\n            frames.append(img)\n")

# 4) After set_frame in TRACKING mode, log pose + trajectory
pose_log = '''
            # --- Rerun: log pose ---
            try:
                _T = frame.T_WC.matrix().detach().cpu().numpy().squeeze()
                if _T.shape == (4, 4):
                    _t = _T[:3, 3].astype(_rr_np.float32)
                    _R = _T[:3, :3]
                    _tr = _R[0,0]+_R[1,1]+_R[2,2]
                    if _tr > 0:
                        _s = 0.5/_rr_np.sqrt(_tr+1.0); _w=0.25/_s
                        _x=(_R[2,1]-_R[1,2])*_s; _y=(_R[0,2]-_R[2,0])*_s; _z=(_R[1,0]-_R[0,1])*_s
                    else:
                        _w,_x,_y,_z = 1.0,0.0,0.0,0.0
                    rr.log("camera", rr.Transform3D(translation=_t, quaternion=[_x,_y,_z,_w]))
                    rr.log("camera", rr.Pinhole(focal_length=[525.0, 525.0], principal_point=[319.5, 239.5], resolution=[640, 480], image_plane_distance=0.15))
                    _RR_TRAJECTORY.append(_t.tolist())
                    if len(_RR_TRAJECTORY) >= 2:
                        rr.log("trajectory", rr.LineStrips3D([_RR_TRAJECTORY], colors=[[0,200,255]]))
            except Exception:
                pass
'''
src = src.replace(
    "            states.set_frame(frame)\n\n        elif mode == Mode.RELOC:",
    "            states.set_frame(frame)\n" + pose_log + "\n        elif mode == Mode.RELOC:")

# 5) After keyframes.append(frame), log point cloud
kf_log = '''
            # --- Rerun: log factor graph ---
            try:
                _kf_positions = []
                for _ki in range(len(keyframes)):
                    _kf = keyframes[_ki]
                    _kfT = _kf.T_WC.matrix().detach().cpu().numpy().squeeze()
                    _kf_positions.append(_kfT[:3, 3].astype(_rr_np.float32))
                if _kf_positions:
                    _kf_pts = _rr_np.array(_kf_positions)
                    rr.log("graph/nodes", rr.Points3D(_kf_pts, radii=0.02, colors=[[255, 100, 0]]))
                    _edges = []
                    for _ei in range(len(_kf_positions) - 1):
                        _edges.append([_kf_positions[_ei].tolist(), _kf_positions[_ei+1].tolist()])
                    if _edges:
                        rr.log("graph/edges", rr.LineStrips3D(_edges, colors=[[255, 200, 0]]))
            except Exception:
                pass
            # --- Rerun: log keyframe point cloud ---
            try:
                _X = frame.X_canon
                _C = frame.C
                _T = frame.T_WC.matrix().detach().cpu().numpy().squeeze()
                if _X is not None:
                    _pts = _X.detach().cpu().numpy().reshape(-1, 3)
                    if _C is not None:
                        _conf = _C.detach().cpu().numpy().reshape(-1)
                        _pts = _pts[_conf > 1.5]
                    if len(_pts) > 0:
                        _pts_w = (_T[:3,:3] @ _pts.T).T + _T[:3,3]
                        if len(_pts_w) > 50000:
                            _idx = _rr_np.random.choice(len(_pts_w), 50000, replace=False)
                            _pts_w = _pts_w[_idx]
                        rr.log(f"map/kf_{len(keyframes)}", rr.Points3D(_pts_w.astype(_rr_np.float32), radii=0.005))
            except Exception:
                pass
'''
src = src.replace(
    "            keyframes.append(frame)\n            states.queue_global_optimization",
    "            keyframes.append(frame)\n" + kf_log + "            states.queue_global_optimization")

# 6) Keep-alive at end (guarded for __main__ only)
keep_alive = '''

# --- Rerun: keep web viewer alive ---
if __name__ == "__main__" and os.environ.get("RERUN_WEB"):
    print("\\nSLAM complete. Rerun viewer still running.", flush=True)
    print("Press Ctrl+C to exit.", flush=True)
    import time as _time
    try:
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        pass
'''
src = src + keep_alive

with open(MAIN_PY, "w") as f:
    f.write(src)

print("Rerun patch applied to main.py")
