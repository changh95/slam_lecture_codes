#!/usr/bin/env python3
"""
04_stereo_calibration.py - Stereo Camera Calibration

This example demonstrates stereo camera calibration to estimate:
1. Individual camera intrinsics (K1, K2, dist1, dist2)
2. Relative pose between cameras (R, T)
3. Essential matrix E and Fundamental matrix F

These parameters enable stereo rectification and depth estimation.
"""

import cv2
import numpy as np
import yaml
import os


def create_synthetic_stereo_images(
    board_size=(9, 6),
    n_images=15,
    image_size=(640, 480),
    baseline=0.1  # 10cm baseline
):
    """
    Create synthetic stereo calibration image pairs.

    Returns synchronized left/right images with known ground truth.
    """

    # Camera intrinsics (same for both cameras)
    fx, fy = 500, 500
    cx, cy = 320, 240
    K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K2 = K1.copy()

    # Slight distortion
    dist1 = np.array([-0.1, 0.05, 0.0, 0.0, 0], dtype=np.float64)
    dist2 = np.array([-0.12, 0.06, 0.0, 0.0, 0], dtype=np.float64)

    # True stereo extrinsics: camera2 is shifted right by baseline
    R_true = np.eye(3, dtype=np.float64)  # Cameras are parallel
    T_true = np.array([[baseline], [0.0], [0.0]], dtype=np.float64)

    # Object points (3D checkerboard corners)
    square_size = 0.025  # 25mm
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images_left = []
    images_right = []
    all_objpoints = []
    all_imgpoints_left = []
    all_imgpoints_right = []

    np.random.seed(42)

    for i in range(n_images * 2):  # Generate extra to ensure enough valid pairs
        if len(images_left) >= n_images:
            break

        # Random board pose (relative to left camera)
        rvec = np.random.uniform(-0.4, 0.4, 3).reshape(3, 1)
        tvec = np.array([
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(0.4, 0.8)
        ]).reshape(3, 1)

        # Project to left camera
        imgpts_left, _ = cv2.projectPoints(objp, rvec, tvec, K1, dist1)
        imgpts_left = imgpts_left.reshape(-1, 2)

        # Transform board pose to right camera frame
        R_board, _ = cv2.Rodrigues(rvec)
        T_board = tvec

        # Board pose in right camera: R_right = R_true @ R_board, T_right = R_true @ T_board + T_true
        R_board_right = R_true @ R_board
        T_board_right = R_true @ T_board + T_true

        rvec_right, _ = cv2.Rodrigues(R_board_right)

        # Project to right camera
        imgpts_right, _ = cv2.projectPoints(objp, rvec_right, T_board_right, K2, dist2)
        imgpts_right = imgpts_right.reshape(-1, 2)

        # Check if all points are visible in both images
        margin = 20
        valid_left = np.all(
            (imgpts_left >= [margin, margin]) &
            (imgpts_left <= [image_size[0]-margin, image_size[1]-margin]),
            axis=1
        )
        valid_right = np.all(
            (imgpts_right >= [margin, margin]) &
            (imgpts_right <= [image_size[0]-margin, image_size[1]-margin]),
            axis=1
        )

        if np.all(valid_left) and np.all(valid_right):
            # Create images
            img_left = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200
            img_right = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200

            for pt in imgpts_left:
                cv2.circle(img_left, (int(pt[0]), int(pt[1])), 4, 0, -1)
            for pt in imgpts_right:
                cv2.circle(img_right, (int(pt[0]), int(pt[1])), 4, 0, -1)

            images_left.append(img_left)
            images_right.append(img_right)
            all_objpoints.append(objp)
            all_imgpoints_left.append(imgpts_left.reshape(-1, 1, 2).astype(np.float32))
            all_imgpoints_right.append(imgpts_right.reshape(-1, 1, 2).astype(np.float32))

    ground_truth = {
        'K1': K1, 'K2': K2,
        'dist1': dist1, 'dist2': dist2,
        'R': R_true, 'T': T_true
    }

    return (images_left, images_right,
            all_objpoints, all_imgpoints_left, all_imgpoints_right,
            ground_truth)


def stereo_calibrate(
    object_points, image_points_left, image_points_right,
    image_size, K1=None, dist1=None, K2=None, dist2=None
):
    """
    Perform stereo camera calibration.

    If K1/K2 are not provided, intrinsics are estimated.
    """

    flags = 0

    if K1 is None or K2 is None:
        # Estimate everything
        flags = cv2.CALIB_FIX_INTRINSIC
        # First calibrate each camera individually
        ret1, K1, dist1, _, _ = cv2.calibrateCamera(
            object_points, image_points_left, image_size, None, None
        )
        ret2, K2, dist2, _, _ = cv2.calibrateCamera(
            object_points, image_points_right, image_size, None, None
        )
        flags = cv2.CALIB_FIX_INTRINSIC

    # Stereo calibration
    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        K1, dist1,
        K2, dist2,
        image_size,
        flags=flags
    )

    return ret, K1, dist1, K2, dist2, R, T, E, F


def save_stereo_calibration(K1, dist1, K2, dist2, R, T, filename):
    """Save stereo calibration to YAML file."""
    data = {
        'camera_matrix_left': K1.tolist(),
        'distortion_left': dist1.tolist(),
        'camera_matrix_right': K2.tolist(),
        'distortion_right': dist2.tolist(),
        'rotation_matrix': R.tolist(),
        'translation_vector': T.tolist(),
    }

    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Saved stereo calibration to: {filename}")


def main():
    print("=" * 60)
    print("Stereo Camera Calibration")
    print("=" * 60)

    # Configuration
    BOARD_SIZE = (9, 6)
    IMAGE_SIZE = (640, 480)
    N_IMAGES = 15
    BASELINE = 0.1  # 10cm

    # =========================================================================
    # 1. Generate synthetic stereo data
    # =========================================================================
    print("\n--- 1. Generating Stereo Calibration Data ---")

    (images_left, images_right,
     objpoints, imgpoints_left, imgpoints_right,
     ground_truth) = create_synthetic_stereo_images(
         board_size=BOARD_SIZE,
         n_images=N_IMAGES,
         image_size=IMAGE_SIZE,
         baseline=BASELINE
     )

    print(f"Generated {len(images_left)} stereo pairs")
    print(f"Baseline: {BASELINE*100:.1f} cm")

    print("\nGround truth stereo extrinsics:")
    print(f"  R (rotation):\n{ground_truth['R']}")
    print(f"  T (translation): {ground_truth['T'].flatten()}")

    # =========================================================================
    # 2. Individual camera calibration
    # =========================================================================
    print("\n--- 2. Individual Camera Calibration ---")

    ret_left, K1_est, dist1_est, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, IMAGE_SIZE, None, None
    )

    ret_right, K2_est, dist2_est, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, IMAGE_SIZE, None, None
    )

    print(f"Left camera RMS error: {ret_left:.4f} pixels")
    print(f"Right camera RMS error: {ret_right:.4f} pixels")

    print("\nEstimated left camera matrix:")
    print(K1_est)

    print("\nEstimated right camera matrix:")
    print(K2_est)

    # =========================================================================
    # 3. Stereo calibration
    # =========================================================================
    print("\n--- 3. Stereo Calibration ---")

    flags = cv2.CALIB_FIX_INTRINSIC  # Use pre-calibrated intrinsics

    ret_stereo, K1_stereo, dist1_stereo, K2_stereo, dist2_stereo, R, T, E, F = \
        cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            K1_est, dist1_est,
            K2_est, dist2_est,
            IMAGE_SIZE,
            flags=flags
        )

    print(f"\nStereo calibration RMS error: {ret_stereo:.4f} pixels")

    # =========================================================================
    # 4. Analyze results
    # =========================================================================
    print("\n--- 4. Results Analysis ---")

    print("\nEstimated rotation matrix R:")
    print(R)

    print("\nEstimated translation vector T:")
    print(T.flatten())

    # Compute baseline
    baseline_est = np.linalg.norm(T)
    print(f"\nEstimated baseline: {baseline_est*100:.2f} cm")
    print(f"True baseline: {BASELINE*100:.2f} cm")
    print(f"Error: {abs(baseline_est - BASELINE)*1000:.2f} mm")

    # Rotation error
    R_error = R @ ground_truth['R'].T
    angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
    print(f"\nRotation error: {np.degrees(angle_error):.4f} degrees")

    # Translation direction error
    T_true_norm = ground_truth['T'].flatten() / np.linalg.norm(ground_truth['T'])
    T_est_norm = T.flatten() / np.linalg.norm(T)
    dir_error = np.arccos(np.clip(np.dot(T_true_norm, T_est_norm), -1, 1))
    print(f"Translation direction error: {np.degrees(dir_error):.4f} degrees")

    # =========================================================================
    # 5. Essential and Fundamental matrices
    # =========================================================================
    print("\n--- 5. Essential and Fundamental Matrices ---")

    print("\nEssential matrix E:")
    print(E)

    print("\nFundamental matrix F:")
    print(F)

    print("""
    Relationship:
    - E encodes relative pose (R, T) between cameras
    - F = K2^-T @ E @ K1^-1
    - For point correspondences: p2^T @ F @ p1 = 0
    """)

    # Verify Essential matrix decomposition
    U, S, Vt = np.linalg.svd(E)
    print(f"\nEssential matrix singular values: {S}")
    print("(Should be [1, 1, 0] up to scale)")

    # =========================================================================
    # 6. Epipolar constraint verification
    # =========================================================================
    print("\n--- 6. Epipolar Constraint Verification ---")

    # Use first image pair
    pts_left = imgpoints_left[0].reshape(-1, 2)
    pts_right = imgpoints_right[0].reshape(-1, 2)

    # Compute epipolar errors
    errors = []
    for i in range(len(pts_left)):
        p1 = np.array([pts_left[i, 0], pts_left[i, 1], 1.0])
        p2 = np.array([pts_right[i, 0], pts_right[i, 1], 1.0])

        # Epipolar constraint: p2^T @ F @ p1 = 0
        error = abs(p2 @ F @ p1)
        errors.append(error)

    print(f"Mean epipolar error: {np.mean(errors):.6f}")
    print(f"Max epipolar error: {np.max(errors):.6f}")
    print("(Should be close to 0 for perfect calibration)")

    # =========================================================================
    # 7. Different calibration flags
    # =========================================================================
    print("\n--- 7. Calibration Flags Comparison ---")

    flags_options = [
        (cv2.CALIB_FIX_INTRINSIC, "CALIB_FIX_INTRINSIC"),
        (0, "Refine all parameters"),
        (cv2.CALIB_SAME_FOCAL_LENGTH, "CALIB_SAME_FOCAL_LENGTH"),
    ]

    for flags, name in flags_options:
        ret, _, _, _, _, R_test, T_test, _, _ = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            K1_est.copy(), dist1_est.copy(),
            K2_est.copy(), dist2_est.copy(),
            IMAGE_SIZE,
            flags=flags
        )
        baseline_test = np.linalg.norm(T_test)
        print(f"  {name:30s}: RMS={ret:.4f}, baseline={baseline_test*100:.2f}cm")

    # =========================================================================
    # 8. Save calibration
    # =========================================================================
    print("\n--- 8. Saving Calibration ---")

    output_dir = "../calibration_results"
    os.makedirs(output_dir, exist_ok=True)

    save_stereo_calibration(
        K1_stereo, dist1_stereo,
        K2_stereo, dist2_stereo,
        R, T,
        os.path.join(output_dir, "stereo_calibration.yaml")
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - cv2.stereoCalibrate() estimates R, T, E, F")
    print("  - Pre-calibrate each camera for better results")
    print("  - CALIB_FIX_INTRINSIC uses known intrinsics")
    print("  - Baseline = ||T|| (distance between cameras)")
    print("  - Verify with epipolar constraint")
    print("=" * 60)


if __name__ == "__main__":
    main()
