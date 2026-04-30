#!/usr/bin/env python3
"""
02_camera_calibration.py - Monocular Camera Calibration

This example demonstrates camera calibration to estimate:
1. Camera intrinsic matrix K (fx, fy, cx, cy)
2. Distortion coefficients (k1, k2, p1, p2, k3)

These parameters are essential for SLAM and any computer vision application.
"""

import cv2
import numpy as np
import yaml
import os


def create_synthetic_calibration_images(
    board_size=(9, 6),
    n_images=10,
    image_size=(640, 480),
    fx=500, fy=500, cx=320, cy=240,
    k1=-0.2, k2=0.1, p1=0.0, p2=0.0
):
    """Create synthetic calibration images with known parameters."""

    # True camera matrix and distortion
    K_true = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_true = np.array([k1, k2, p1, p2, 0], dtype=np.float64)

    # Object points (3D)
    square_size = 0.025  # 25mm
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images = []
    all_objpoints = []
    all_imgpoints = []

    np.random.seed(42)

    for i in range(n_images):
        # Random rotation and translation
        rvec = np.random.uniform(-0.3, 0.3, 3).reshape(3, 1)
        tvec = np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(0.3, 0.6)
        ]).reshape(3, 1)

        # Project points
        imgpoints, _ = cv2.projectPoints(objp, rvec, tvec, K_true, dist_true)
        imgpoints = imgpoints.reshape(-1, 2)

        # Check if all points are in image
        in_bounds = np.all(
            (imgpoints >= [10, 10]) & (imgpoints <= [image_size[0]-10, image_size[1]-10]),
            axis=1
        )

        if np.all(in_bounds):
            # Create image with projected corners
            img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200

            # Draw corners as small circles
            for pt in imgpoints:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 3, 0, -1)

            images.append(img)
            all_objpoints.append(objp)
            all_imgpoints.append(imgpoints.reshape(-1, 1, 2).astype(np.float32))

    return images, all_objpoints, all_imgpoints, K_true, dist_true


def calibrate_camera(object_points, image_points, image_size):
    """
    Calibrate camera from object/image point correspondences.

    Returns:
        ret: RMS reprojection error
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=0
    )

    return ret, K, dist, rvecs, tvecs


def save_calibration(K, dist, filename="calibration.yaml"):
    """Save calibration to YAML file."""
    data = {
        'camera_matrix': K.tolist(),
        'distortion_coefficients': dist.tolist(),
        'image_width': 640,
        'image_height': 480,
    }

    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Saved calibration to: {filename}")


def load_calibration(filename="calibration.yaml"):
    """Load calibration from YAML file."""
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix'])
    dist = np.array(data['distortion_coefficients'])

    return K, dist


def main():
    print("=" * 60)
    print("Camera Calibration")
    print("=" * 60)

    # Configuration
    BOARD_SIZE = (9, 6)
    IMAGE_SIZE = (640, 480)
    N_IMAGES = 15

    # =========================================================================
    # 1. Generate synthetic calibration data
    # =========================================================================
    print("\n--- 1. Generating Calibration Data ---")

    images, objpoints, imgpoints, K_true, dist_true = create_synthetic_calibration_images(
        board_size=BOARD_SIZE,
        n_images=N_IMAGES,
        image_size=IMAGE_SIZE
    )

    print(f"Generated {len(images)} calibration images")
    print(f"\nTrue camera matrix:\n{K_true}")
    print(f"\nTrue distortion: {dist_true.flatten()}")

    # =========================================================================
    # 2. Run calibration
    # =========================================================================
    print("\n--- 2. Running Calibration ---")

    ret, K_est, dist_est, rvecs, tvecs = calibrate_camera(
        objpoints, imgpoints, IMAGE_SIZE
    )

    print(f"\nCalibration complete!")
    print(f"RMS reprojection error: {ret:.4f} pixels")

    # =========================================================================
    # 3. Compare results
    # =========================================================================
    print("\n--- 3. Calibration Results ---")

    print(f"\nEstimated camera matrix:")
    print(K_est)

    print(f"\nEstimated distortion coefficients:")
    print(f"  k1={dist_est[0][0]:.6f}, k2={dist_est[0][1]:.6f}")
    print(f"  p1={dist_est[0][2]:.6f}, p2={dist_est[0][3]:.6f}")
    print(f"  k3={dist_est[0][4]:.6f}")

    # Compare with ground truth
    print("\n--- Comparison with Ground Truth ---")

    print(f"\nFocal length:")
    print(f"  True:      fx={K_true[0,0]:.2f}, fy={K_true[1,1]:.2f}")
    print(f"  Estimated: fx={K_est[0,0]:.2f}, fy={K_est[1,1]:.2f}")
    print(f"  Error:     Δfx={abs(K_true[0,0]-K_est[0,0]):.2f}, "
          f"Δfy={abs(K_true[1,1]-K_est[1,1]):.2f}")

    print(f"\nPrincipal point:")
    print(f"  True:      cx={K_true[0,2]:.2f}, cy={K_true[1,2]:.2f}")
    print(f"  Estimated: cx={K_est[0,2]:.2f}, cy={K_est[1,2]:.2f}")

    print(f"\nDistortion (k1, k2):")
    print(f"  True:      k1={dist_true[0]:.4f}, k2={dist_true[1]:.4f}")
    print(f"  Estimated: k1={dist_est[0][0]:.4f}, k2={dist_est[0][1]:.4f}")

    # =========================================================================
    # 4. Per-image reprojection error
    # =========================================================================
    print("\n--- 4. Per-Image Reprojection Error ---")

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_reproj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K_est, dist_est
        )
        error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
        total_error += error
        print(f"  Image {i+1}: {error:.4f} pixels")

    print(f"\nMean error: {total_error/len(objpoints):.4f} pixels")

    # =========================================================================
    # 5. Save calibration
    # =========================================================================
    print("\n--- 5. Saving Calibration ---")

    output_dir = "../calibration_results"
    os.makedirs(output_dir, exist_ok=True)

    save_calibration(K_est, dist_est, os.path.join(output_dir, "calibration.yaml"))

    # =========================================================================
    # 6. Guidelines
    # =========================================================================
    print("\n--- 6. Calibration Best Practices ---")

    print("""
    For good calibration:

    1. NUMBER OF IMAGES: Use 10-20 images minimum

    2. COVERAGE: Cover the entire image area
       - Place board in corners and center
       - Vary distance (close and far)

    3. ANGLES: Vary the board orientation
       - Tilted in X, Y, and Z
       - Different rotations

    4. QUALITY: Ensure sharp, well-lit images
       - No motion blur
       - Good lighting, no shadows

    5. BOARD QUALITY: Use a flat, rigid board
       - Printed on matte paper/board
       - Known, accurate square size

    6. CHECK ERROR: Reprojection error should be < 0.5 pixels
       - > 1.0 pixel indicates problems
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - cv2.calibrateCamera() estimates K and distortion")
    print("  - Provide object points (3D) and image points (2D)")
    print("  - RMS error indicates calibration quality")
    print("  - Save calibration for later use")
    print("=" * 60)


if __name__ == "__main__":
    main()
