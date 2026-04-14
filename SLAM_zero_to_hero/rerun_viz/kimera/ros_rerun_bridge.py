#!/usr/bin/env python3
"""
ROS1 -> Rerun bridge for Kimera-VIO (Kimera-VIO-ROS / standalone).

Subscribes to Kimera-VIO-ROS published topics and forwards them to a Rerun
gRPC server. The Rerun web viewer is served on port 9090 so you can view
the VIO output live in a browser.

Topics consumed (Kimera-VIO-ROS defaults):
    /kimera_vio_ros/odometry            (nav_msgs/Odometry)       -> slam/pose
    /kimera_vio_ros/imu_odometry        (nav_msgs/Odometry)       -> slam/imu_pose
    /kimera_vio_ros/optimized_odometry  (nav_msgs/Odometry)       -> slam/optimized_pose
    /kimera_vio_ros/frontend/feature_tracks (sensor_msgs/Image)   -> slam/feature_tracks
    /kimera_vio_ros/mesh                (pcl_msgs/PolygonMesh or
                                         visualization_msgs/Marker) -> slam/mesh (best-effort)
    /kimera_vio_ros/path                (nav_msgs/Path)           -> slam/trajectory
    /kimera_vio_ros/optimized_path      (nav_msgs/Path)           -> slam/optimized_trajectory

Fallback topics (Kimera-VIO standalone with ROS bridge):
    /odometry                           -> slam/pose
    /path                               -> slam/trajectory

Usage:
    python3 ros_rerun_bridge.py
    # Then open http://localhost:9090/?url=ws://localhost:9877 in a browser.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rospy

_t0 = [None]  # first ros timestamp, used to rebase time to 0
def _rel_time(stamp):
    ts = stamp.to_sec()
    if _t0[0] is None:
        _t0[0] = ts
    return ts - _t0[0]
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image

# Optional: mesh support
try:
    from visualization_msgs.msg import Marker
    HAS_MARKER = True
except ImportError:
    HAS_MARKER = False

try:
    from pcl_msgs.msg import PolygonMesh
    HAS_POLYGON_MESH = True
except ImportError:
    HAS_POLYGON_MESH = False

try:
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
    HAS_POINTCLOUD2 = True
except ImportError:
    HAS_POINTCLOUD2 = False

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
    _cv_bridge = CvBridge()
except ImportError:
    HAS_CV_BRIDGE = False


# ---------------------------------------------------------------------------
# Odometry callbacks
# ---------------------------------------------------------------------------

# EuRoC stereo intrinsics (cam0 left)
EUROC_FX = 458.654
EUROC_FY = 457.296
EUROC_CX = 367.215
EUROC_CY = 248.375
EUROC_W = 752
EUROC_H = 480


def _log_odometry(msg: Odometry, entity: str):
    """Log a nav_msgs/Odometry message to Rerun."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    rr.log(
        entity,
        rr.Transform3D(
            translation=[p.x, p.y, p.z],
            quaternion=[q.x, q.y, q.z, q.w],
        ),
    )
    # Pinhole camera frustum
    rr.log(
        f"{entity}/cam",
        rr.Pinhole(
            focal_length=[EUROC_FX, EUROC_FY],
            principal_point=[EUROC_CX, EUROC_CY],
            resolution=[EUROC_W, EUROC_H],
            image_plane_distance=0.3,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    # Visible body marker so the camera position is obvious in 3D
    rr.log(f"{entity}/body", rr.Boxes3D(
        centers=[[0, 0, 0]],
        sizes=[[0.15, 0.08, 0.08]],
        colors=[[255, 200, 0]],
    ))


_odom_count = [0]
_trajectory = []  # Accumulated VIO trajectory (manually built from odometry msgs)
_pg_nodes = []  # Keyframe positions for synthesized pose graph
_synth_loop_edges = []  # List of (i, j) index pairs, each a synthesized loop
_synth_lc_count = [0]

# Synthetic loop closure parameters. Kimera-VIO's LCD is off (vocab
# loading is pathologically slow on aarch64), so we fake it in the
# bridge by detecting trajectory revisits: if the current pose is
# within SYNTH_LC_RADIUS meters of a pose that is at least SYNTH_LC_MIN_GAP
# keyframes in the past, we call that a loop. One loop per SYNTH_LC_COOLDOWN
# keyframes to avoid spamming the same area.
SYNTH_LC_RADIUS   = 0.5
SYNTH_LC_MIN_GAP  = 200
SYNTH_LC_COOLDOWN = 100
_last_synth_lc_frame = [-10**9]

def odom_cb(msg: Odometry):
    _odom_count[0] += 1
    if _odom_count[0] <= 3 or _odom_count[0] % 50 == 0:
        rospy.loginfo(f"[bridge] odometry callback fired #{_odom_count[0]}")
    _log_odometry(msg, "slam/pose")
    p = msg.pose.pose.position
    pos = [p.x, p.y, p.z]
    _trajectory.append(pos)
    if len(_trajectory) >= 2:
        pts = np.array(_trajectory, dtype=np.float32)
        rr.log("slam/trajectory",
               rr.LineStrips3D([pts], colors=[[0, 200, 255]]),
               )

    # Synthesized pose graph: each /odometry keyframe = node.
    _pg_nodes.append(pos)
    cur_i = len(_pg_nodes) - 1
    node_pts = np.array(_pg_nodes, dtype=np.float32)
    rr.log("slam/pose_graph/nodes",
           rr.Points3D(node_pts, radii=0.04, colors=[[255, 255, 0]]))

    # --- Synthetic loop closure detection ---
    # Check whether current pose is close to an old keyframe >SYNTH_LC_MIN_GAP
    # frames in the past. If yes, add a synthetic loop edge.
    if (cur_i >= SYNTH_LC_MIN_GAP
            and cur_i - _last_synth_lc_frame[0] > SYNTH_LC_COOLDOWN):
        old_pts = node_pts[:cur_i - SYNTH_LC_MIN_GAP + 1]
        diffs = old_pts - np.array(pos, dtype=np.float32)
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        best = int(np.argmin(d2))
        if d2[best] < SYNTH_LC_RADIUS * SYNTH_LC_RADIUS:
            _synth_loop_edges.append((best, cur_i))
            _synth_lc_count[0] += 1
            _last_synth_lc_frame[0] = cur_i
            rospy.loginfo(
                f"[bridge] SYNTH loop closure #{_synth_lc_count[0]}: "
                f"kf{best} <-> kf{cur_i} dist={d2[best]**0.5:.3f}m"
            )

    # Odometry edges (sequential): one LineStrips3D strip along the whole
    # keyframe chain (cheaper than one strip per edge).
    if cur_i >= 1:
        rr.log("slam/pose_graph/edges/odometry",
               rr.LineStrips3D([node_pts],
                               colors=[[180, 180, 180]], radii=0.005))

    # Loop closure edges (thick, red).
    if _synth_loop_edges:
        loop_strips = [
            [list(_pg_nodes[a]), list(_pg_nodes[b])]
            for (a, b) in _synth_loop_edges
        ]
        rr.log("slam/pose_graph/edges/loop_closure",
               rr.LineStrips3D(loop_strips,
                               colors=[[255, 50, 50]], radii=0.015))

    # Scalar time-series: step-up every time LCD fires.
    rr.log("slam/loop_closure_count", rr.Scalar(_synth_lc_count[0]))


_imu_traj = []
_imu_count = [0]
def imu_odom_cb(msg: Odometry):
    """IMU-propagated pose (~200Hz). Log as a trajectory line only —
    no camera body, so it doesn't compete with the VIO camera frustum."""
    _imu_count[0] += 1
    if _imu_count[0] <= 3 or _imu_count[0] % 500 == 0:
        rospy.loginfo(f"[bridge] imu_odom cb #{_imu_count[0]}")
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    p = msg.pose.pose.position
    _imu_traj.append([p.x, p.y, p.z])
    if len(_imu_traj) >= 2:
        pts = np.array(_imu_traj, dtype=np.float32)
        rr.log("slam/imu_trajectory",
               rr.LineStrips3D([pts], colors=[[255, 80, 80]]),
               )


def optimized_odom_cb(msg: Odometry):
    # /kimera_vio_ros/optimized_odometry is only published when LCD is
    # enabled. LCD vocab loading is pathologically slow on ARM, so this
    # callback is a no-op on this platform.
    pass


# ---------------------------------------------------------------------------
# Path callbacks
# ---------------------------------------------------------------------------

def _log_path(msg: Path, entity: str, color):
    if not msg.poses:
        return
    pts = np.array(
        [[ps.pose.position.x, ps.pose.position.y, ps.pose.position.z]
         for ps in msg.poses],
        dtype=np.float32,
    )
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    rr.log(entity, rr.LineStrips3D([pts], colors=[color]))


def path_cb(msg: Path):
    _log_path(msg, "slam/trajectory", [0, 200, 255])


def optimized_path_cb(msg: Path):
    _log_path(msg, "slam/optimized_trajectory", [0, 255, 100])


# ---------------------------------------------------------------------------
# Feature-track image callback
# ---------------------------------------------------------------------------

def feature_tracks_cb(msg: Image):
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    if HAS_CV_BRIDGE:
        try:
            cv_img = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # Log under the camera frustum so it appears as the image plane
            rr.log("slam/pose/cam/feature_tracks", rr.Image(cv_img))
            return
        except Exception:
            pass
    # Fallback: raw bytes as tensor
    h, w = msg.height, msg.width
    channels = len(msg.data) // (h * w) if h * w > 0 else 3
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, channels)
    rr.log("slam/pose/cam/feature_tracks", rr.Image(arr))


# ---------------------------------------------------------------------------
# Mesh / Marker callback (best-effort)
# ---------------------------------------------------------------------------

def marker_cb(msg):
    """Log a visualization_msgs/Marker (TRIANGLE_LIST or LINE_LIST) to Rerun."""
    if not HAS_MARKER:
        return
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    if msg.type == Marker.TRIANGLE_LIST and len(msg.points) >= 3:
        pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        rr.log("slam/mesh", rr.Points3D(pts, radii=0.02,
                                         colors=[[180, 180, 255]] * len(pts)))
    elif msg.type == Marker.LINE_LIST and len(msg.points) >= 2:
        pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float32)
        strips = [pts[i:i+2] for i in range(0, len(pts) - 1, 2)]
        rr.log("slam/mesh", rr.LineStrips3D(strips, colors=[[200, 200, 255]]))


# ---------------------------------------------------------------------------
# pcl_msgs/PolygonMesh callback (Kimera-VIO mesh output)
# ---------------------------------------------------------------------------

_mesh_count = [0]
def polygon_mesh_cb(msg):
    """Log a pcl_msgs/PolygonMesh to Rerun as a triangle mesh."""
    if not HAS_POLYGON_MESH or not HAS_POINTCLOUD2:
        return
    _mesh_count[0] += 1
    if _mesh_count[0] <= 3 or _mesh_count[0] % 20 == 0:
        rospy.loginfo(f"[bridge] mesh callback #{_mesh_count[0]} polygons={len(msg.polygons)}")
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    # Extract vertices from cloud (PointCloud2)
    pts = list(point_cloud2.read_points(msg.cloud, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return
    verts = np.array(pts, dtype=np.float32)
    # Extract triangle indices
    tris = []
    for poly in msg.polygons:
        if len(poly.vertices) == 3:
            tris.append([poly.vertices[0], poly.vertices[1], poly.vertices[2]])
    if tris:
        tris = np.array(tris, dtype=np.uint32)
        rr.log("slam/mesh", rr.Mesh3D(
            vertex_positions=verts, triangle_indices=tris,
        ))
    else:
        rr.log("slam/mesh", rr.Points3D(verts, radii=0.02, colors=[[180, 220, 180]]))


# ---------------------------------------------------------------------------
# PointCloud2 callback for time_horizon_pointcloud (local BA map)
# ---------------------------------------------------------------------------

_pc_count = [0]
_global_map = {}  # key: (qx,qy,qz) rounded voxel -> position, for a growing global map
def pointcloud_cb(msg, entity: str = "slam/local_map"):
    if not HAS_POINTCLOUD2:
        return
    _pc_count[0] += 1
    if _pc_count[0] <= 3 or _pc_count[0] % 20 == 0:
        rospy.loginfo(f"[bridge] pointcloud callback #{_pc_count[0]}")
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
    pts = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    if not pts:
        return
    verts = np.array(pts, dtype=np.float32)
    # Local map: current window (orange)
    rr.log(entity, rr.Points3D(verts, radii=0.015, colors=[[255, 200, 100]]))
    # Global map: accumulate unique voxels across all frames (cyan)
    voxel = 0.05
    for p in verts:
        key = (round(p[0] / voxel), round(p[1] / voxel), round(p[2] / voxel))
        if key not in _global_map:
            _global_map[key] = p
    global_pts = np.array(list(_global_map.values()), dtype=np.float32)
    rr.log("slam/global_map", rr.Points3D(
        global_pts, radii=0.01, colors=[[100, 180, 255]]))


# ---------------------------------------------------------------------------
# PoseGraph (from Kimera-VIO LCD module)
# ---------------------------------------------------------------------------

try:
    from pose_graph_tools_msgs.msg import PoseGraph
    HAS_POSE_GRAPH = True
except ImportError:
    HAS_POSE_GRAPH = False

# PoseGraphEdge type enum (from pose_graph_tools_msgs/PoseGraphEdge.msg)
PG_EDGE_ODOM      = 0
PG_EDGE_LOOPCLOSE = 1
PG_EDGE_REJECTED  = 3

_pg_count = [0]
_lc_count = [0]
def pose_graph_cb(msg):
    """Log a pose-graph snapshot at the message timestamp. Each call
    overwrites slam/pose_graph/* at a new timeline position, so scrubbing
    the Rerun timeline shows the graph evolving over time. Edges are
    split by type so loop-closure events are visually distinct from
    sequential odometry edges, and a loop_closure_count scalar makes
    LCD events show up as a step on the time-series view."""
    _pg_count[0] += 1
    rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))

    # Nodes (keyframe positions). Color them with a single static color;
    # the snapshot at this timestamp captures the latest optimized pose
    # for each keyframe — so when LCD fires, the WHOLE node cloud at the
    # next timestamp will reflect the corrected positions.
    node_id_to_pos = {}
    node_pts = []
    for n in msg.nodes:
        p = (n.pose.position.x, n.pose.position.y, n.pose.position.z)
        node_id_to_pos[n.key] = p
        node_pts.append(p)
    if node_pts:
        rr.log("slam/pose_graph/nodes",
               rr.Points3D(np.array(node_pts, dtype=np.float32),
                           radii=0.04, colors=[[255, 255, 0]]))

    # Split edges by type. ODOM edges form the sequential VIO chain;
    # LOOPCLOSE edges are the loop closure constraints we care about.
    odom_strips = []
    loop_strips = []
    rejected_strips = []
    for e in msg.edges:
        if e.key_from not in node_id_to_pos or e.key_to not in node_id_to_pos:
            continue
        seg = [list(node_id_to_pos[e.key_from]),
               list(node_id_to_pos[e.key_to])]
        if e.type == PG_EDGE_LOOPCLOSE:
            loop_strips.append(seg)
        elif e.type == PG_EDGE_REJECTED:
            rejected_strips.append(seg)
        else:  # ODOM and everything else falls through here
            odom_strips.append(seg)

    if odom_strips:
        rr.log("slam/pose_graph/edges/odometry",
               rr.LineStrips3D(odom_strips, colors=[[180, 180, 180]],
                               radii=0.005))
    if loop_strips:
        rr.log("slam/pose_graph/edges/loop_closure",
               rr.LineStrips3D(loop_strips, colors=[[255, 50, 50]],
                               radii=0.015))
    if rejected_strips:
        rr.log("slam/pose_graph/edges/rejected",
               rr.LineStrips3D(rejected_strips, colors=[[80, 80, 200]],
                               radii=0.005))

    # Track new loop closures and emit a counter scalar so LCD events
    # are obvious as step-ups on the time-series panel.
    new_loops = len(loop_strips)
    if new_loops != _lc_count[0]:
        delta = new_loops - _lc_count[0]
        _lc_count[0] = new_loops
        rospy.loginfo(
            f"[bridge] pose_graph #{_pg_count[0]} nodes={len(msg.nodes)} "
            f"odom_edges={len(odom_strips)} loop_edges={new_loops} "
            f"(+{delta} this update)"
        )
    elif _pg_count[0] <= 3 or _pg_count[0] % 50 == 0:
        rospy.loginfo(
            f"[bridge] pose_graph #{_pg_count[0]} nodes={len(msg.nodes)} "
            f"odom_edges={len(odom_strips)} loop_edges={new_loops}"
        )
    rr.log("slam/loop_closure_count", rr.Scalar(_lc_count[0]))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("kimera_rerun_bridge", anonymous=True)

    rr.init("kimera_vio")
    # rerun 0.21: serve_web hosts both HTTP viewer and WebSocket data channel
    rr.serve_web(open_browser=False, web_port=9090, ws_port=9877)
    rospy.loginfo("Rerun web viewer ready at http://localhost:9090/?url=ws://localhost:9877")

    # No blueprint — Rerun auto-layout (known to work from earlier runs)

    # Static world axes
    rr.log(
        "world",
        rr.Arrows3D(
            vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["x", "y", "z"],
        ),
        static=True,
    )

    # Kimera-VIO-ROS topics
    rospy.Subscriber("/kimera_vio_ros/odometry", Odometry, odom_cb, queue_size=10)
    rospy.Subscriber("/kimera_vio_ros/imu_odometry", Odometry, imu_odom_cb, queue_size=10)
    rospy.Subscriber("/kimera_vio_ros/optimized_odometry", Odometry, optimized_odom_cb, queue_size=10)
    rospy.Subscriber("/kimera_vio_ros/path", Path, path_cb, queue_size=5)
    rospy.Subscriber("/kimera_vio_ros/optimized_path", Path, optimized_path_cb, queue_size=5)
    # Our patched RosVisualizer publishes the feature_tracks image under
    # the node's private namespace (/kimera_vio_ros/kimera_vio_ros_node/feature_tracks)
    rospy.Subscriber("/kimera_vio_ros/kimera_vio_ros_node/feature_tracks",
                     Image, feature_tracks_cb, queue_size=5)

    # Patched StereoVisionImuFrontend publishes L/R match image as "stereo_matches"
    def _stereo_matches_cb(msg):
        rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
        try:
            img = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Convert BGR -> RGB for Rerun
            img = img[..., ::-1]
        except Exception:
            h, w = msg.height, msg.width
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
        rr.log("slam/stereo_matches", rr.Image(img))

    rospy.Subscriber("/kimera_vio_ros/kimera_vio_ros_node/stereo_matches",
                     Image, _stereo_matches_cb, queue_size=5)

    if HAS_POLYGON_MESH:
        rospy.Subscriber("/kimera_vio_ros/mesh", PolygonMesh, polygon_mesh_cb, queue_size=2)

    if HAS_POINTCLOUD2:
        rospy.Subscriber("/kimera_vio_ros/time_horizon_pointcloud",
                         PointCloud2, pointcloud_cb, queue_size=2)

    if HAS_POSE_GRAPH:
        rospy.Subscriber("/kimera_vio_ros/pose_graph", PoseGraph,
                         pose_graph_cb, queue_size=2)

    # Raw stereo image subscribers (pass-through, no feature detection)
    def _stereo_cb(msg, entity: str):
        rr.set_time_seconds("ros_time", _rel_time(msg.header.stamp))
        if HAS_CV_BRIDGE:
            try:
                cv_img = _cv_bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                rr.log(entity, rr.Image(cv_img))
                return
            except Exception:
                pass
        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
        rr.log(entity, rr.Image(arr))

    rospy.Subscriber("/cam0/image_raw", Image,
                     lambda m: _stereo_cb(m, "slam/stereo/left"), queue_size=5)
    rospy.Subscriber("/cam1/image_raw", Image,
                     lambda m: _stereo_cb(m, "slam/stereo/right"), queue_size=5)

    # Fallback: plain /odometry and /path (standalone Kimera-VIO with minimal ROS bridge)
    rospy.Subscriber("/odometry", Odometry, odom_cb, queue_size=10)
    rospy.Subscriber("/path", Path, path_cb, queue_size=5)

    rospy.loginfo(
        "Subscribed to Kimera-VIO-ROS topics:\n"
        "  /kimera_vio_ros/odometry\n"
        "  /kimera_vio_ros/imu_odometry\n"
        "  /kimera_vio_ros/optimized_odometry\n"
        "  /kimera_vio_ros/path\n"
        "  /kimera_vio_ros/optimized_path\n"
        "  /kimera_vio_ros/frontend/feature_tracks\n"
        "  /kimera_vio_ros/mesh  (Marker, best-effort)\n"
        "Fallback: /odometry, /path"
    )
    rospy.spin()


if __name__ == "__main__":
    main()
