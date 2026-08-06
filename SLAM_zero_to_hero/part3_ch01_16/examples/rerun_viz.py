#!/usr/bin/env python3
"""Live rerun streaming for the part3 chapter-1 optimizer-backend exercises.

Python counterpart of the ``rerun_viz.hpp`` used by the C++ chapters. It logs to
the same recording names and the same entity paths, so running the SymForce
chapter against the same viewer as the g2o / GTSAM / Ceres chapters overlays
their solutions for comparison.

Everything degrades to a no-op when no viewer answers, so the examples always
run and print their numbers regardless.
"""
from __future__ import annotations

import math
import os
import socket
from typing import Iterable, Sequence

import numpy as np
import rerun as rr

# Recording names, shared with the C++ chapters so solutions overlay.
CURVE_FITTING_RECORDING = "part3_curve_fitting"
POSE_GRAPH_RECORDING = "part3_pose_graph"
BUNDLE_ADJUSTMENT_RECORDING = "part3_bundle_adjustment"

DEFAULT_URL = "rerun+http://127.0.0.1:9876/proxy"

# Edge kinds, matching rerun_viz.hpp.
ODOMETRY = 0
LOOP = 1
LOOP_REJECTED = 2

GT_COLOR = [60, 190, 110]
INIT_COLOR = [150, 150, 150]
OPT_COLOR = [225, 70, 70]
DATA_COLOR = [120, 120, 120]
LOOP_COLOR = [40, 130, 240]
REJECT_COLOR = [225, 70, 70]
CAM_COLOR = [0, 120, 255]
POINT_COLOR = [80, 200, 120]


def viewer_url() -> str:
    """Viewer address: RERUN_URL env var, or the default local viewer."""
    return os.environ.get("RERUN_URL", DEFAULT_URL)


def viewer_reachable(url: str, timeout: float = 2.0) -> bool:
    """True when a TCP server accepts connections at the URL's host:port.

    connect_grpc() never fails on an absent viewer - it just retries in the
    background - so probe reachability first rather than buffering data nobody
    will read.
    """
    if "://" not in url:
        return True
    hostport = url.split("://", 1)[1].split("/", 1)[0]
    if ":" not in hostport:
        return True
    host, _, port = hostport.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def curve_model(abc: Sequence[float], x: np.ndarray | float):
    """The model every chapter's curve-fitting exercise fits."""
    a, b, c = abc[0], abc[1], abc[2]
    return np.exp(a * np.asarray(x) ** 2 + b * np.asarray(x) + c)


def _curve_strip(abc: Sequence[float], x_min: float, x_max: float, samples: int = 200):
    xs = np.linspace(x_min, x_max, samples)
    return np.column_stack([xs, curve_model(abc, xs)])


class Viz:
    """One streaming connection, scoped to a single exercise.

    ``lib`` is the library name and becomes the entity-path segment that keeps
    two chapters' results apart inside a shared recording.
    """

    def __init__(self, recording: str, lib: str):
        self.lib = lib
        self.connected = False
        self._x_min = 0.0
        self._x_max = 1.0
        url = viewer_url()
        if not viewer_reachable(url):
            print(
                f"Note: no rerun viewer reachable at {url} - running without live "
                f"streaming.\n"
                f"      Start one on the host first (rerun &); with Docker add "
                f"--network=host."
            )
            return
        # A fixed recording id (not a fresh random one) is what lets another
        # chapter's process append to the same recording and overlay.
        self._stream = rr.RecordingStream(recording, recording_id=recording)
        self._stream.connect_grpc(url)
        self.connected = True
        print(f"Streaming to rerun viewer at {url} as '{lib}'")

    # ------------------------------------------------------------- helpers

    def _log(self, path: str, archetype, static: bool = False) -> None:
        self._stream.log(path, archetype, static=static)

    def _set_iter(self, iteration: int) -> None:
        self._stream.set_time("iteration", sequence=iteration)

    # ---------------------------------------------------------- curve fit

    def curve_setup(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        gt: Sequence[float],
        init: Sequence[float],
    ) -> None:
        """Static context: the samples, the true curve, and the initial guess."""
        if not self.connected or len(xs) == 0:
            return
        self._x_min, self._x_max = float(np.min(xs)), float(np.max(xs))
        self._log(
            "curve/observations",
            rr.Points2D(np.column_stack([xs, ys]), colors=DATA_COLOR, radii=rr.Radius.ui_points(2.0)),
            static=True,
        )
        self._log(
            "curve/ground_truth",
            rr.LineStrips2D(
                [_curve_strip(gt, self._x_min, self._x_max)],
                colors=GT_COLOR,
                radii=rr.Radius.ui_points(2.0),
            ),
            static=True,
        )
        self._log(
            f"curve/{self.lib}/initial",
            rr.LineStrips2D(
                [_curve_strip(init, self._x_min, self._x_max)],
                colors=INIT_COLOR,
                radii=rr.Radius.ui_points(1.5),
            ),
            static=True,
        )
        self._log(
            f"cost/{self.lib}",
            rr.SeriesLines(names=[self.lib], colors=[OPT_COLOR], widths=[1.5]),
            static=True,
        )

    def curve_iteration(self, iteration: int, abc: Sequence[float], cost: float) -> None:
        """One solver iteration: the current curve plus cost and parameters."""
        if not self.connected:
            return
        self._set_iter(iteration)
        self._log(
            f"curve/{self.lib}/fitted",
            rr.LineStrips2D(
                [_curve_strip(abc, self._x_min, self._x_max)],
                colors=OPT_COLOR,
                radii=rr.Radius.ui_points(2.5),
            ),
        )
        self._log(f"cost/{self.lib}", rr.Scalars(cost))
        for name, value in zip("abc", abc):
            self._log(f"params/{self.lib}/{name}", rr.Scalars(float(value)))

    # -------------------------------------------------------- pose graph

    def pose_graph_setup(
        self,
        gt: Sequence[Sequence[float]],
        init: Sequence[Sequence[float]],
        edges: Iterable[Sequence[int]],
    ) -> None:
        """Static context: ground truth, the noisy initial estimate, the edges.

        Heading arrows go alongside the positions because the shared square-loop
        problem ends where it started: pose 4 sits exactly on pose 0 and differs
        only in orientation, so the loop-closure constraint is invisible in a
        position-only plot.
        """
        if not self.connected or len(gt) == 0:
            return
        edges = list(edges)
        self._log_poses_2d("graph/ground_truth", gt, GT_COLOR, static=True)
        self._log_poses_2d(f"graph/{self.lib}/initial", init, INIT_COLOR, static=True)
        self._log_loops_2d("graph/ground_truth/loop_closures", gt, edges, static=True)
        self._log(
            f"cost/{self.lib}",
            rr.SeriesLines(names=[self.lib], colors=[OPT_COLOR], widths=[1.5]),
            static=True,
        )

    def pose_graph_iteration(
        self,
        iteration: int,
        poses: Sequence[Sequence[float]],
        cost: float,
        edges: Iterable[Sequence[int]] = (),
    ) -> None:
        """One solver iteration of the 2D pose graph."""
        if not self.connected or len(poses) == 0:
            return
        self._set_iter(iteration)
        self._log_poses_2d(f"graph/{self.lib}/optimized", poses, OPT_COLOR)
        edges = list(edges)
        if edges:
            self._log_loops_2d(
                f"graph/{self.lib}/optimized/loop_closures", poses, edges, static=False
            )
        self._log(f"cost/{self.lib}", rr.Scalars(cost))

    def _log_poses_2d(self, base, poses, color, static: bool = False) -> None:
        pts = np.asarray([[p[0], p[1]] for p in poses], dtype=float)
        self._log(base + "/poses", rr.Points2D(pts, colors=color, radii=rr.Radius.ui_points(4.0)), static)
        self._log(
            base + "/path",
            rr.LineStrips2D([pts], colors=color, radii=rr.Radius.ui_points(2.0)),
            static,
        )
        span = 0.0
        if len(pts) > 1:
            span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        length = 0.18 * span if span > 0 else 0.2
        vectors = np.asarray(
            [[length * math.cos(p[2]), length * math.sin(p[2])] for p in poses], dtype=float
        )
        self._log(
            base + "/heading",
            rr.Arrows2D(vectors=vectors, origins=pts, colors=color),
            static,
        )

    def _log_loops_2d(self, path, poses, edges, static: bool = True) -> None:
        strips, markers = [], []
        for e in edges:
            i, j, kind = int(e[0]), int(e[1]), int(e[2])
            if kind == ODOMETRY or not (0 <= i < len(poses) and 0 <= j < len(poses)):
                continue
            a = [poses[i][0], poses[i][1]]
            b = [poses[j][0], poses[j][1]]
            strips.append(np.asarray([a, b], dtype=float))
            # The square loop closes onto its own start, so the edge can be
            # zero-length; markers keep it visible either way.
            markers.extend([a, b])
        if not markers:
            return
        self._log(
            path,
            rr.LineStrips2D(strips, colors=LOOP_COLOR, radii=rr.Radius.ui_points(2.0)),
            static,
        )
        self._log(
            path + "/endpoints",
            rr.Points2D(
                np.asarray(markers, dtype=float),
                colors=LOOP_COLOR,
                radii=rr.Radius.ui_points(7.0),
            ),
            static,
        )

    # ------------------------------------------------- bundle adjustment

    def ba_setup(self, initial_points) -> None:
        """Static reference cloud: where the landmarks started."""
        if not self.connected:
            return
        self._log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        pts = np.asarray(initial_points, dtype=float)
        if pts.size == 0:
            return
        self._log(
            "world/initial_points",
            rr.Points3D(pts, colors=DATA_COLOR, radii=rr.Radius.ui_points(1.0)),
            static=True,
        )
        self._log(
            f"reprojection_error/{self.lib}",
            rr.SeriesLines(names=[self.lib], colors=[OPT_COLOR], widths=[1.5]),
            static=True,
        )

    def ba_iteration(
        self,
        iteration: int,
        points,
        cameras,
        sq_error: float,
        rmse_px: float,
        robust_cost: float = -1.0,
    ) -> None:
        """One bundle-adjustment iteration.

        ``sq_error`` must be the raw sum of squared reprojection error and
        ``rmse_px`` its per-observation RMS, so the number means the same thing
        in every chapter whether or not that chapter uses a robust kernel.
        """
        if not self.connected:
            return
        self._set_iter(iteration)
        pts = np.asarray(points, dtype=float)
        cams = np.asarray(cameras, dtype=float)
        if pts.size:
            self._log(
                f"world/{self.lib}/landmarks",
                rr.Points3D(pts, colors=POINT_COLOR, radii=rr.Radius.ui_points(1.5)),
            )
        if cams.size:
            self._log(
                f"world/{self.lib}/cameras",
                rr.Points3D(cams, colors=CAM_COLOR, radii=rr.Radius.ui_points(6.0)),
            )
        self._log(f"reprojection_error/{self.lib}", rr.Scalars(sq_error))
        self._log(f"rmse_px/{self.lib}", rr.Scalars(rmse_px))
        if robust_cost >= 0.0:
            self._log(f"robust_cost/{self.lib}", rr.Scalars(robust_cost))
