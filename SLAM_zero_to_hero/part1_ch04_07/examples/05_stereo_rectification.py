#!/usr/bin/env python3
"""
05_stereo_rectification.py - Stereo Image Rectification

This example demonstrates stereo rectification to:
1. Make epipolar lines horizontal (parallel to image rows)
2. Enable efficient stereo matching for depth estimation
3. Align images so corresponding points have same y-coordinate

After rectification, depth can be computed from disparity:
    depth = (baseline * fx) / disparity
"""

import cv2
import numpy as np


def create_stereo_scene(
    image_size=(640, 480),
    baseline=0.1,
    focal_length=500
):
    """
    Create a synthetic stereo scene with 3D points and their projections.
    """

    # Camera parameters
    K = np.array([
        [focal_length, 0, image_size[0]/2],
        [0, focal_length, image_size[1]/2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist = np.array([-0.15, 0.05, 0.0, 0.0, 0], dtype=np.float64)

    # Stereo geometry
    R = np.eye(3, dtype=np.float64)
    T = np.array([[baseline], [0.0], [0.0]], dtype=np.float64)

    # Create 3D points in left camera frame
    np.random.seed(42)
    n_points = 50

    points_3d = np.zeros((n_points, 3))
    points_3d[:, 0] = np.random.uniform(-0.3, 0.3, n_points)  # X
    points_3d[:, 1] = np.random.uniform(-0.2, 0.2, n_points)  # Y
    points_3d[:, 2] = np.random.uniform(0.5, 2.0, n_points)   # Z (depth)

    # Project to left camera
    rvec_left = np.zeros(3)
    tvec_left = np.zeros(3)
    pts_left, _ = cv2.projectPoints(points_3d, rvec_left, tvec_left, K, dist)
    pts_left = pts_left.reshape(-1, 2)

    # Project to right camera (apply stereo transform)
    rvec_right, _ = cv2.Rodrigues(R)
    tvec_right = T.flatten()
    pts_right, _ = cv2.projectPoints(points_3d, rvec_right, tvec_right, K, dist)
    pts_right = pts_right.reshape(-1, 2)

    # Filter points visible in both images
    margin = 20
    valid = (
        (pts_left[:, 0] >= margin) & (pts_left[:, 0] < image_size[0] - margin) &
        (pts_left[:, 1] >= margin) & (pts_left[:, 1] < image_size[1] - margin) &
        (pts_right[:, 0] >= margin) & (pts_right[:, 0] < image_size[0] - margin) &
        (pts_right[:, 1] >= margin) & (pts_right[:, 1] < image_size[1] - margin)
    )

    points_3d = points_3d[valid]
    pts_left = pts_left[valid]
    pts_right = pts_right[valid]

    # Create images with points
    img_left = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200
    img_right = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200

    for pt in pts_left:
        cv2.circle(img_left, (int(pt[0]), int(pt[1])), 5, 0, -1)
    for pt in pts_right:
        cv2.circle(img_right, (int(pt[0]), int(pt[1])), 5, 0, -1)

    return {
        'img_left': img_left,
        'img_right': img_right,
        'pts_left': pts_left,
        'pts_right': pts_right,
        'points_3d': points_3d,
        'K': K,
        'dist': dist,
        'R': R,
        'T': T,
        'image_size': image_size
    }


def compute_rectification(K1, dist1, K2, dist2, R, T, image_size):
    """
    Compute stereo rectification transforms.

    Returns rotation matrices and projection matrices for both cameras.
    """

    # Stereo rectification
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, dist1,
        K2, dist2,
        image_size,
        R, T,
        alpha=0  # 0 = crop to valid region, 1 = keep all pixels
    )

    return {
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2,
        'Q': Q,
        'roi1': validPixROI1,
        'roi2': validPixROI2
    }


def compute_rectification_maps(K, dist, R, P, image_size):
    """Compute undistortion and rectification maps."""

    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist, R, P, image_size, cv2.CV_32FC1
    )

    return map_x, map_y


def draw_epipolar_lines(img_left, img_right, n_lines=10):
    """Draw horizontal lines to verify rectification."""

    img_left_color = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    img_right_color = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

    h = img_left.shape[0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0)]

    for i, y in enumerate(np.linspace(50, h-50, n_lines).astype(int)):
        color = colors[i % len(colors)]
        cv2.line(img_left_color, (0, y), (img_left.shape[1]-1, y), color, 1)
        cv2.line(img_right_color, (0, y), (img_right.shape[1]-1, y), color, 1)

    return img_left_color, img_right_color


def main():
    print("=" * 60)
    print("Stereo Image Rectification")
    print("=" * 60)

    IMAGE_SIZE = (640, 480)
    BASELINE = 0.1
    FOCAL_LENGTH = 500

    # =========================================================================
    # 1. Create stereo scene
    # =========================================================================
    print("\n--- 1. Creating Stereo Scene ---")

    scene = create_stereo_scene(IMAGE_SIZE, BASELINE, FOCAL_LENGTH)

    print(f"Image size: {IMAGE_SIZE}")
    print(f"Baseline: {BASELINE*100:.1f} cm")
    print(f"Number of points: {len(scene['points_3d'])}")

    # =========================================================================
    # 2. Analyze unrectified images
    # =========================================================================
    print("\n--- 2. Unrectified Image Analysis ---")

    # Check y-coordinate differences (before rectification)
    y_diffs = scene['pts_left'][:, 1] - scene['pts_right'][:, 1]

    print("Y-coordinate differences between correspondences:")
    print(f"  Mean: {np.mean(np.abs(y_diffs)):.2f} pixels")
    print(f"  Max:  {np.max(np.abs(y_diffs)):.2f} pixels")
    print("(Should be ~0 after rectification)")

    cv2.imwrite("unrectified_left.png", scene['img_left'])
    cv2.imwrite("unrectified_right.png", scene['img_right'])
    print("\nSaved: unrectified_left.png, unrectified_right.png")

    # =========================================================================
    # 3. Compute rectification
    # =========================================================================
    print("\n--- 3. Computing Rectification ---")

    K1 = scene['K']
    K2 = scene['K'].copy()
    dist1 = scene['dist']
    dist2 = scene['dist'].copy()
    R = scene['R']
    T = scene['T']

    rect = compute_rectification(K1, dist1, K2, dist2, R, T, IMAGE_SIZE)

    print("Rectification rotation R1 (left camera):")
    print(rect['R1'])

    print("\nRectification rotation R2 (right camera):")
    print(rect['R2'])

    print("\nProjection matrix P1:")
    print(rect['P1'])

    print("\nProjection matrix P2:")
    print(rect['P2'])

    # =========================================================================
    # 4. Understanding the Q matrix
    # =========================================================================
    print("\n--- 4. Disparity-to-Depth Matrix (Q) ---")

    print("Q matrix (for reprojectImageTo3D):")
    print(rect['Q'])

    print("""
    Q converts (x, y, disparity) to (X, Y, Z):

    [X]       [x - cx    ]
    [Y] = Q * [y - cy    ]
    [Z]       [disparity ]
    [W]       [1         ]

    Then: X_3d = X/W, Y_3d = Y/W, Z_3d = Z/W

    depth = baseline * fx / disparity
    """)

    # Extract baseline from Q matrix
    baseline_from_Q = 1.0 / rect['Q'][3, 2]
    print(f"\nBaseline from Q matrix: {abs(baseline_from_Q)*100:.2f} cm")

    # =========================================================================
    # 5. Apply rectification
    # =========================================================================
    print("\n--- 5. Applying Rectification ---")

    # Compute maps
    map1_x, map1_y = compute_rectification_maps(K1, dist1, rect['R1'], rect['P1'], IMAGE_SIZE)
    map2_x, map2_y = compute_rectification_maps(K2, dist2, rect['R2'], rect['P2'], IMAGE_SIZE)

    # Apply rectification
    img_rect_left = cv2.remap(scene['img_left'], map1_x, map1_y, cv2.INTER_LINEAR)
    img_rect_right = cv2.remap(scene['img_right'], map2_x, map2_y, cv2.INTER_LINEAR)

    print("Applied rectification transforms")

    cv2.imwrite("rectified_left.png", img_rect_left)
    cv2.imwrite("rectified_right.png", img_rect_right)
    print("Saved: rectified_left.png, rectified_right.png")

    # =========================================================================
    # 6. Rectify points
    # =========================================================================
    print("\n--- 6. Rectifying Point Correspondences ---")

    # Undistort and rectify points
    pts_left_rect = cv2.undistortPoints(
        scene['pts_left'].reshape(-1, 1, 2).astype(np.float32),
        K1, dist1, R=rect['R1'], P=rect['P1']
    ).reshape(-1, 2)

    pts_right_rect = cv2.undistortPoints(
        scene['pts_right'].reshape(-1, 1, 2).astype(np.float32),
        K2, dist2, R=rect['R2'], P=rect['P2']
    ).reshape(-1, 2)

    # Check y-coordinate alignment after rectification
    y_diffs_rect = pts_left_rect[:, 1] - pts_right_rect[:, 1]

    print("Y-coordinate differences after rectification:")
    print(f"  Mean: {np.mean(np.abs(y_diffs_rect)):.4f} pixels")
    print(f"  Max:  {np.max(np.abs(y_diffs_rect)):.4f} pixels")
    print("(Should be very close to 0)")

    # =========================================================================
    # 7. Compute disparity
    # =========================================================================
    print("\n--- 7. Computing Disparity ---")

    # Disparity = x_left - x_right (for rectified images)
    disparity = pts_left_rect[:, 0] - pts_right_rect[:, 0]

    print("Sample disparity values:")
    for i in range(min(5, len(disparity))):
        depth_true = scene['points_3d'][i, 2]
        depth_from_disp = (BASELINE * FOCAL_LENGTH) / disparity[i] if disparity[i] > 0 else float('inf')
        print(f"  Point {i}: disparity={disparity[i]:.2f}, "
              f"depth_est={depth_from_disp:.3f}m, "
              f"depth_true={depth_true:.3f}m")

    # =========================================================================
    # 8. Draw epipolar lines
    # =========================================================================
    print("\n--- 8. Epipolar Line Visualization ---")

    img_left_lines, img_right_lines = draw_epipolar_lines(
        img_rect_left, img_rect_right, n_lines=8
    )

    # Combine side by side
    combined = np.hstack([img_left_lines, img_right_lines])
    cv2.imwrite("rectified_with_epipolar_lines.png", combined)
    print("Saved: rectified_with_epipolar_lines.png")
    print("Epipolar lines should be horizontal and aligned between images")

    # =========================================================================
    # 9. Alpha parameter effect
    # =========================================================================
    print("\n--- 9. Alpha Parameter Effect ---")

    print("""
    stereoRectify alpha parameter:
    - alpha=0: Crop to only valid pixels (no black borders)
    - alpha=1: Keep all original pixels (may have black borders)
    """)

    for alpha in [0.0, 0.5, 1.0]:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, dist1, K2, dist2, IMAGE_SIZE, R, T, alpha=alpha
        )
        print(f"  alpha={alpha}: ROI_left={roi1}, ROI_right={roi2}")

    # =========================================================================
    # 10. Depth from disparity
    # =========================================================================
    print("\n--- 10. Depth from Disparity ---")

    print("""
    After rectification, depth computation is simple:

        Z = baseline * fx / disparity

    Where:
        - baseline: distance between cameras
        - fx: focal length in pixels
        - disparity: x_left - x_right (in pixels)

    Accuracy depends on:
        1. Disparity precision (sub-pixel matching helps)
        2. Baseline (larger = more accurate at distance)
        3. Calibration quality
    """)

    # Demonstrate depth accuracy
    depths_true = scene['points_3d'][:, 2]
    depths_est = (BASELINE * FOCAL_LENGTH) / disparity

    depth_errors = np.abs(depths_est - depths_true)
    relative_errors = depth_errors / depths_true * 100

    print("Depth estimation accuracy:")
    print(f"  Mean absolute error: {np.mean(depth_errors)*100:.2f} cm")
    print(f"  Mean relative error: {np.mean(relative_errors):.2f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - cv2.stereoRectify() computes R1, R2, P1, P2, Q")
    print("  - After rectification, epipolar lines are horizontal")
    print("  - Corresponding points have same y-coordinate")
    print("  - Disparity = x_left - x_right")
    print("  - Depth = baseline * fx / disparity")
    print("=" * 60)


if __name__ == "__main__":
    main()
