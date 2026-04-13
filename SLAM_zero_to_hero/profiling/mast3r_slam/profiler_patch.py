#!/usr/bin/env python3
"""
Patch MASt3R-SLAM source files to inject py_profiler blocks around major components.

Instruments:
  main.py:
    - SLAM/ImageLoad        (dataset[i])
    - SLAM/FrameCreation    (create_frame)
    - SLAM/MonoInference    (mast3r_inference_mono in INIT/RELOC)
    - SLAM/Tracking         (tracker.track)
    - SLAM/KeyframeInsert   (keyframes.append + queue_global_optimization)

  tracker.py:
    - SLAM/FeatureMatching  (mast3r_match_asymmetric)
    - SLAM/PoseEstimation   (opt_pose_*_sim3)
    - SLAM/PointmapUpdate   (update_pointmap + keyframe write-back)
"""

import sys
sys.path.insert(0, "/profiling")

# ---- Patch main.py ----
MAIN_PY = "/MASt3R-SLAM/main.py"
with open(MAIN_PY) as f:
    src = f.read()

# Add profiler import + enable
src = src.replace(
    "import yaml\n",
    "import yaml\nimport os as _os\nimport sys as _sys\n_sys.path.insert(0, '/profiling')\nfrom py_profiler import enable as _prof_enable, dump as _prof_dump, block as _prof_block\n_prof_enable()\n"
)

# ImageLoad: wrap "timestamp, img = dataset[i]"
src = src.replace(
    "        timestamp, img = dataset[i]\n",
    "        with _prof_block('SLAM/ImageLoad'):\n"
    "            timestamp, img = dataset[i]\n"
)

# FrameCreation: wrap "frame = create_frame(...)"
src = src.replace(
    "        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)\n",
    "        with _prof_block('SLAM/FrameCreation'):\n"
    "            frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)\n"
)

# MonoInference (INIT): wrap mast3r_inference_mono in INIT block
src = src.replace(
    "            # Initialize via mono inference, and encoded features neeed for database\n"
    "            X_init, C_init = mast3r_inference_mono(model, frame)\n",
    "            # Initialize via mono inference, and encoded features neeed for database\n"
    "            with _prof_block('SLAM/MonoInference'):\n"
    "                X_init, C_init = mast3r_inference_mono(model, frame)\n"
)

# Tracking: wrap tracker.track
src = src.replace(
    "            add_new_kf, match_info, try_reloc = tracker.track(frame)\n",
    "            with _prof_block('SLAM/Tracking'):\n"
    "                add_new_kf, match_info, try_reloc = tracker.track(frame)\n"
)

# MonoInference (RELOC): wrap mast3r_inference_mono in RELOC block
src = src.replace(
    "            X, C = mast3r_inference_mono(model, frame)\n",
    "            with _prof_block('SLAM/MonoInference'):\n"
    "                X, C = mast3r_inference_mono(model, frame)\n"
)

# KeyframeInsert: wrap keyframes.append + queue
src = src.replace(
    "        if add_new_kf:\n"
    "            keyframes.append(frame)\n"
    "            states.queue_global_optimization(len(keyframes) - 1)\n",
    "        if add_new_kf:\n"
    "            with _prof_block('SLAM/KeyframeInsert'):\n"
    "                keyframes.append(frame)\n"
    "                states.queue_global_optimization(len(keyframes) - 1)\n"
)

# Wrap the entire while loop in SLAM/FullRun block
src = src.replace(
    "    i = 0\n"
    "    fps_timer = time.time()\n",
    "    i = 0\n"
    "    fps_timer = time.time()\n"
    "    _fullrun_ctx = _prof_block('SLAM/FullRun')\n"
    "    _fullrun_ctx.__enter__()\n"
)

# Close FullRun block and dump after the SLAM loop ends (at save_results)
src = src.replace(
    "    if dataset.save_results:\n"
    "        save_dir, seq_name = eval.prepare_savedir(args, dataset)\n"
    "        eval.save_traj(",
    "    _fullrun_ctx.__exit__(None, None, None)\n"
    "    _prof_output = _os.environ.get('PROFILER_OUTPUT', '/output/mast3r_slam.json')\n"
    "    _prof_dump(_prof_output)\n"
    "    print(f'Profiler data saved to {_prof_output}', flush=True)\n"
    "    if dataset.save_results:\n"
    "        save_dir, seq_name = eval.prepare_savedir(args, dataset)\n"
    "        eval.save_traj("
)

# ---- Patch run_backend in main.py ----
# Add profiler enable at start of run_backend
src = src.replace(
    "def run_backend(cfg, model, states, keyframes, K):\n"
    "    set_global_config(cfg)\n",
    "def run_backend(cfg, model, states, keyframes, K):\n"
    "    set_global_config(cfg)\n"
    "    _prof_enable()\n"
)

# Wrap retrieval_database.update
src = src.replace(
    "        retrieval_inds = retrieval_database.update(\n"
    "            frame,\n"
    "            add_after_query=True,\n",
    "        with _prof_block('SLAM/Retrieval'):\n"
    "          retrieval_inds = retrieval_database.update(\n"
    "            frame,\n"
    "            add_after_query=True,\n"
)
# Fix closing of retrieval block - need to handle the indentation
src = src.replace(
    "            min_thresh=config[\"retrieval\"][\"min_thresh\"],\n"
    "        )\n"
    "        kf_idx += retrieval_inds\n",
    "            min_thresh=config[\"retrieval\"][\"min_thresh\"],\n"
    "          )\n"
    "        kf_idx += retrieval_inds\n"
)

# Wrap factor_graph.add_factors
src = src.replace(
    "            factor_graph.add_factors(\n"
    "                kf_idx, frame_idx, config[\"local_opt\"][\"min_match_frac\"]\n"
    "            )\n",
    "            with _prof_block('SLAM/FactorAddition'):\n"
    "                factor_graph.add_factors(\n"
    "                    kf_idx, frame_idx, config[\"local_opt\"][\"min_match_frac\"]\n"
    "                )\n"
)

# Wrap solve_GN (bundle adjustment)
src = src.replace(
    "        if config[\"use_calib\"]:\n"
    "            factor_graph.solve_GN_calib()\n"
    "        else:\n"
    "            factor_graph.solve_GN_rays()\n",
    "        with _prof_block('SLAM/BundleAdjustment'):\n"
    "            if config[\"use_calib\"]:\n"
    "                factor_graph.solve_GN_calib()\n"
    "            else:\n"
    "                factor_graph.solve_GN_rays()\n"
)

# Wrap relocalization
src = src.replace(
    "            success = relocalization(frame, keyframes, factor_graph, retrieval_database)\n",
    "            with _prof_block('SLAM/Relocalization'):\n"
    "                success = relocalization(frame, keyframes, factor_graph, retrieval_database)\n"
)

# Dump backend profiler data before backend exits
# The while loop ends when mode == TERMINATED, then function returns
# Add dump right before the function ends (before the if __name__ block)
src = src.replace(
    "\nif __name__ == \"__main__\":",
    "    _backend_output = _os.environ.get('PROFILER_OUTPUT', '/output/mast3r_slam.json')\n"
    "    _backend_output = _backend_output.replace('.json', '_backend.json')\n"
    "    _prof_dump(_backend_output)\n"
    "    print(f'Backend profiler data saved to {_backend_output}', flush=True)\n"
    "\nif __name__ == \"__main__\":"
)

with open(MAIN_PY, "w") as f:
    f.write(src)
print("Profiler patch applied to main.py")

# ---- Patch tracker.py ----
TRACKER_PY = "/MASt3R-SLAM/mast3r_slam/tracker.py"
with open(TRACKER_PY) as f:
    tsrc = f.read()

# Add profiler import
tsrc = tsrc.replace(
    "from mast3r_slam.mast3r_utils import mast3r_match_asymmetric\n",
    "from mast3r_slam.mast3r_utils import mast3r_match_asymmetric\n"
    "import sys as _sys\n_sys.path.insert(0, '/profiling')\nfrom py_profiler import block as _prof_block\n"
)

# FeatureMatching: wrap mast3r_match_asymmetric call
tsrc = tsrc.replace(
    "        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(\n"
    "            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k\n"
    "        )\n",
    "        with _prof_block('SLAM/FeatureMatching'):\n"
    "            idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(\n"
    "                self.model, frame, keyframe, idx_i2j_init=self.idx_f2k\n"
    "            )\n"
)

# PoseEstimation: wrap opt_pose calls (both variants)
tsrc = tsrc.replace(
    "            if not use_calib:\n"
    "                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(\n"
    "                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt\n"
    "                )\n",
    "            if not use_calib:\n"
    "              with _prof_block('SLAM/PoseEstimation'):\n"
    "                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(\n"
    "                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt\n"
    "                )\n"
)

tsrc = tsrc.replace(
    "                T_WCf, T_CkCf = self.opt_pose_calib_sim3(\n",
    "              with _prof_block('SLAM/PoseEstimation'):\n"
    "                T_WCf, T_CkCf = self.opt_pose_calib_sim3(\n"
)

# PointmapUpdate: wrap the keyframe update_pointmap + write-back section
tsrc = tsrc.replace(
    "        # Use pose to transform points to update keyframe\n"
    "        Xkk = T_CkCf.act(Xkf)\n"
    "        keyframe.update_pointmap(Xkk, Ckf)\n"
    "        # write back the fitered pointmap\n"
    "        self.keyframes[len(self.keyframes) - 1] = keyframe\n",
    "        with _prof_block('SLAM/PointmapUpdate'):\n"
    "            # Use pose to transform points to update keyframe\n"
    "            Xkk = T_CkCf.act(Xkf)\n"
    "            keyframe.update_pointmap(Xkk, Ckf)\n"
    "            # write back the fitered pointmap\n"
    "            self.keyframes[len(self.keyframes) - 1] = keyframe\n"
)

with open(TRACKER_PY, "w") as f:
    f.write(tsrc)
print("Profiler patch applied to tracker.py")
