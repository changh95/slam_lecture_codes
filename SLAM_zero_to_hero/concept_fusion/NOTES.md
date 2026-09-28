# ConceptFusion — notes

## The flat `~/data/icl_nuim/rgb|depth` folders are unusable

The old `download_icl_nuim.py` (fixed 2026-09-28; it now extracts into `<sequence>/`) extracted all 16 ICL-NUIM archives into the same directory. Every
archive contains `rgb/`, `depth/` and `associations.txt` with the same file names
(`0.png`, `1.png`, …), so each one overwrites the previous: the 1509 frames left in
`~/data/icl_nuim/rgb` are a mix of several trajectories (the 1241-line `associations.txt`
belongs to a 1241-frame sequence, yet 1509 images remain). None of 18 sampled frames
(0, 50, …, 850) matches the same frame of living room 2. Only the `*.gt.freiburg` files survive intact, because their names differ.

This demo therefore uses its own clean copy, extracted into a sub-folder:

```
~/data/icl_nuim/living_room_traj2_frei_png/
  rgb/0..880.png  depth/0..880.png  associations.txt  livingRoom2.gt.freiburg  livingRoom2n.gt.sim
```

(881 frames, 463 MB). The downloader also has no `--list` flag. Both are to be fixed in
the downloader itself (extract each archive into `<archive name>/`), not here.

## Poses: `*.gt.sim`, not `*.gt.freiburg`

gradslam's `ICLDataset` (branch `conceptfusion`) reads the `*.gt.sim` file (3×4 [R|t] per
frame, the "Global_RT_Trajectory_GT" link on the ICL page) and pairs it with
`fy = -480` in `dataconfigs/icl.yaml` — POV-Ray's left-handed camera. The frei_png archive
does not contain it; it is at `https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom2n.gt.sim`
(the page only offers the `n` variant; the trajectory is the same, noise only affects the images).

The `.sim` file has **880** poses for **881** frames, and gradslam slices images and poses
independently, so `end=880` is required or the last strided frame gets the wrong pose.

## Why living room 2

Upstream defaults to `living_room_traj1_frei_png`. Trajectory 2 sweeps across the sofa,
the coffee table and the shelves in 880 frames, so a 44-frame stride-20 subset already
covers most of the room.

## Software versions

The original Dockerfile used CUDA 11.8 + cu118 wheels, which cannot run kernels on sm_120.
Now CUDA 12.8.1 + torch 2.7.1+cu128. ConceptFusion itself has no compiled code.
`rerun-sdk 0.33` requires numpy ≥ 2, so numpy 2.1.3; matplotlib is pinned to 3.8.4 because
upstream `demo_text_query.py` / `demo_click_query.py` call `matplotlib.cm.get_cmap`, removed in 3.9.
