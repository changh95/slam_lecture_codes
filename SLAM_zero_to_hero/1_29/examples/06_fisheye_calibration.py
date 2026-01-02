#!/usr/bin/env python3
"""
06_fisheye_calibration.py - Fisheye Camera Calibration

This example demonstrates fisheye camera calibration using the
Kannala-Brandt model (cv2.fisheye module).

Fisheye lenses have very wide FOV (>180°) and require a different
distortion model than the standard pinhole + radial/tangential model.

Kannala-Brandt model:
    θ_d = θ(1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸)

Where:
    θ = atan(r) is the angle from optical axis
    θ_d is the distorted angle
    r = sqrt(x² + y²) is the radius in normalized image plane
"""

import cv2
import numpy as np
import yaml
import os


def create_synthetic_fisheye_images(
    board_size=(9, 6),
    n_images=15,
    image_size=(640, 480),
    fx=250, fy=250, cx=320, cy=240,
    k1=-0.02, k2=0.01, k3=-0.005, k4=0.002
):
    """
    Create synthetic fisheye calibration images.

    Uses Kannala-Brandt model: θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
    """

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)

    # Object points
    square_size = 0.03  # 30mm
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    images = []
    all_objpoints = []
    all_imgpoints = []

    np.random.seed(42)

    for i in range(n_images * 3):
        if len(images) >= n_images:
            break

        # Random board pose
        rvec = np.random.uniform(-0.5, 0.5, 3).reshape(3, 1).astype(np.float64)
        tvec = np.array([
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(0.25, 0.5)
        ], dtype=np.float64).reshape(3, 1)

        # Project using fisheye model
        imgpoints, _ = cv2.fisheye.projectPoints(
            objp.reshape(-1, 1, 3).astype(np.float64),
            rvec, tvec, K, D
        )
        imgpoints = imgpoints.reshape(-1, 2)

        # Check bounds
        margin = 15
        in_bounds = np.all(
            (imgpoints >= [margin, margin]) &
            (imgpoints <= [image_size[0]-margin, image_size[1]-margin]),
            axis=1
        )

        if np.all(in_bounds):
            img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 200

            for pt in imgpoints:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, 0, -1)

            images.append(img)
            all_objpoints.append(objp.reshape(-1, 1, 3).astype(np.float64))
            all_imgpoints.append(imgpoints.reshape(-1, 1, 2).astype(np.float64))

    return images, all_objpoints, all_imgpoints, K, D


def fisheye_calibrate(object_points, image_points, image_size):
    """
    Calibrate fisheye camera using Kannala-Brandt model.
    """

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    # Calibration flags
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_FIX_SKEW
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        K, D,
        flags=flags,
        criteria=criteria
    )

    return ret, K, D, rvecs, tvecs


def undistort_fisheye(image, K, D, balance=0.0):
    """
    Undistort fisheye image.

    balance: 0.0 = crop all invalid pixels, 1.0 = keep all pixels
    """

    h, w = image.shape[:2]

    # Get new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )

    # Compute maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
    )

    # Apply
    undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    return undistorted, new_K


def save_fisheye_calibration(K, D, filename):
    """Save fisheye calibration to YAML."""
    data = {
        'camera_matrix': K.tolist(),
        'distortion_coefficients': D.flatten().tolist(),
        'model': 'kannala_brandt'
    }

    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Saved fisheye calibration to: {filename}")


def main():
    print("=" * 60)
    print("Fisheye Camera Calibration")
    print("=" * 60)

    # Configuration
    BOARD_SIZE = (9, 6)
    IMAGE_SIZE = (640, 480)
    N_IMAGES = 15

    # =========================================================================
    # 1. Fisheye distortion model
    # =========================================================================
    print("\n--- 1. Fisheye Distortion Model ---")

    print("""
    Kannala-Brandt model (used by cv2.fisheye):

    For a 3D point P = (X, Y, Z):
        1. Compute angle: θ = atan2(r, Z), where r = sqrt(X² + Y²)
        2. Apply distortion: θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)
        3. Project: x' = θ_d * X/r, y' = θ_d * Y/r
        4. Pixel coords: u = fx*x' + cx, v = fy*y' + cy

    Parameters:
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  (3x3 matrix)
        D = [k1, k2, k3, k4]  (4 distortion coefficients)

    Key differences from pinhole model:
        - Uses θ (angle) instead of r (radius)
        - No tangential distortion terms
        - 4 radial coefficients (vs 3 for pinhole)
    """)

    # =========================================================================
    # 2. Generate synthetic data
    # =========================================================================
    print("\n--- 2. Generating Calibration Data ---")

    images, objpoints, imgpoints, K_true, D_true = create_synthetic_fisheye_images(
        board_size=BOARD_SIZE,
        n_images=N_IMAGES,
        image_size=IMAGE_SIZE
    )

    print(f"Generated {len(images)} calibration images")
    print(f"\nTrue camera matrix K:")
    print(K_true)
    print(f"\nTrue distortion D: {D_true.flatten()}")

    # =========================================================================
    # 3. Run fisheye calibration
    # =========================================================================
    print("\n--- 3. Running Fisheye Calibration ---")

    ret, K_est, D_est, rvecs, tvecs = fisheye_calibrate(
        objpoints, imgpoints, IMAGE_SIZE
    )

    print(f"\nCalibration complete!")
    print(f"RMS reprojection error: {ret:.4f} pixels")

    print(f"\nEstimated camera matrix K:")
    print(K_est)

    print(f"\nEstimated distortion D: {D_est.flatten()}")

    # =========================================================================
    # 4. Compare with ground truth
    # =========================================================================
    print("\n--- 4. Comparison with Ground Truth ---")

    print(f"\nFocal length:")
    print(f"  True:      fx={K_true[0,0]:.2f}, fy={K_true[1,1]:.2f}")
    print(f"  Estimated: fx={K_est[0,0]:.2f}, fy={K_est[1,1]:.2f}")

    print(f"\nPrincipal point:")
    print(f"  True:      cx={K_true[0,2]:.2f}, cy={K_true[1,2]:.2f}")
    print(f"  Estimated: cx={K_est[0,2]:.2f}, cy={K_est[1,2]:.2f}")

    print(f"\nDistortion coefficients:")
    print(f"  True:      k1={D_true[0]:.6f}, k2={D_true[1]:.6f}, k3={D_true[2]:.6f}, k4={D_true[3]:.6f}")
    D_flat = D_est.flatten()
    print(f"  Estimated: k1={D_flat[0]:.6f}, k2={D_flat[1]:.6f}, k3={D_flat[2]:.6f}, k4={D_flat[3]:.6f}")

    # =========================================================================
    # 5. Undistort fisheye image
    # =========================================================================
    print("\n--- 5. Fisheye Undistortion ---")

    # Create a distorted test image with grid
    test_img = np.ones((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.uint8) * 255

    # Draw radial lines
    center = (IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2)
    for angle in range(0, 360, 15):
        rad = np.radians(angle)
        x = int(center[0] + 300 * np.cos(rad))
        y = int(center[1] + 300 * np.sin(rad))
        cv2.line(test_img, center, (x, y), 0, 1)

    # Draw concentric circles
    for r in range(50, 300, 50):
        cv2.circle(test_img, center, r, 0, 1)

    # Apply fisheye distortion (simulate)
    # Create meshgrid
    h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    X, Y = np.meshgrid(x, y)

    # Normalize
    x_norm = (X - K_true[0, 2]) / K_true[0, 0]
    y_norm = (Y - K_true[1, 2]) / K_true[1, 1]

    # Compute r and theta
    r = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan(r)
    theta = np.where(r > 0, theta, 0)

    # Apply distortion
    k1, k2, k3, k4 = D_true.flatten()
    theta2 = theta**2
    theta_d = theta * (1 + k1*theta2 + k2*theta2**2 + k3*theta2**3 + k4*theta2**4)

    # Scale factor
    scale = np.where(r > 0, theta_d / r, 1)
    x_dist = x_norm * scale
    y_dist = y_norm * scale

    # Convert to pixel
    map_x = (x_dist * K_true[0, 0] + K_true[0, 2]).astype(np.float32)
    map_y = (y_dist * K_true[1, 1] + K_true[1, 2]).astype(np.float32)

    distorted_img = cv2.remap(test_img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    cv2.imwrite("fisheye_distorted.png", distorted_img)
    print("Saved: fisheye_distorted.png")

    # Undistort with different balance values
    for balance in [0.0, 0.5, 1.0]:
        undist, new_K = undistort_fisheye(distorted_img, K_true, D_true, balance)
        cv2.imwrite(f"fisheye_undistorted_balance_{balance:.1f}.png", undist)
        print(f"Saved: fisheye_undistorted_balance_{balance:.1f}.png")

    # =========================================================================
    # 6. Calibration flags
    # =========================================================================
    print("\n--- 6. Fisheye Calibration Flags ---")

    print("""
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        Recompute extrinsics in each iteration

    cv2.fisheye.CALIB_CHECK_COND
        Check condition number (stability)

    cv2.fisheye.CALIB_FIX_SKEW
        Fix skew to 0 (orthogonal axes)

    cv2.fisheye.CALIB_FIX_K1, CALIB_FIX_K2, CALIB_FIX_K3, CALIB_FIX_K4
        Fix specific distortion coefficients

    cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT
        Fix principal point to center

    cv2.fisheye.CALIB_FIX_FOCAL_LENGTH
        Fix focal length (requires initial K)
    """)

    # =========================================================================
    # 7. Comparison with standard model
    # =========================================================================
    print("\n--- 7. Fisheye vs Standard Pinhole Model ---")

    print("""
    When to use fisheye model (cv2.fisheye):
    - FOV > 100° (typically fisheye lenses)
    - Severe barrel distortion
    - Action cameras (GoPro), 360° cameras

    When to use standard model (cv2.calibrateCamera):
    - FOV < 100° (standard lenses)
    - Moderate distortion
    - Most industrial and consumer cameras

    Key API differences:
    - cv2.fisheye.calibrate() vs cv2.calibrateCamera()
    - cv2.fisheye.undistortImage() vs cv2.undistort()
    - D has 4 params (k1-k4) vs 5+ params (k1,k2,p1,p2,k3,...)
    """)

    # =========================================================================
    # 8. Undistort points
    # =========================================================================
    print("\n--- 8. Undistorting Points ---")

    # Sample distorted points
    distorted_pts = np.array([
        [100, 100],
        [320, 240],
        [540, 380],
        [50, 50],
        [590, 430]
    ], dtype=np.float64).reshape(-1, 1, 2)

    # Undistort points
    undistorted_pts = cv2.fisheye.undistortPoints(distorted_pts, K_true, D_true)

    print("Distorted -> Undistorted (normalized) points:")
    for i in range(len(distorted_pts)):
        d = distorted_pts[i, 0]
        u = undistorted_pts[i, 0]
        print(f"  ({d[0]:.1f}, {d[1]:.1f}) -> ({u[0]:.4f}, {u[1]:.4f})")

    # With output camera matrix
    undistorted_pts_px = cv2.fisheye.undistortPoints(distorted_pts, K_true, D_true, P=K_true)
    print("\nWith P=K (pixel coordinates):")
    for i in range(len(distorted_pts)):
        d = distorted_pts[i, 0]
        u = undistorted_pts_px[i, 0]
        print(f"  ({d[0]:.1f}, {d[1]:.1f}) -> ({u[0]:.1f}, {u[1]:.1f})")

    # =========================================================================
    # 9. Save calibration
    # =========================================================================
    print("\n--- 9. Saving Calibration ---")

    output_dir = "../calibration_results"
    os.makedirs(output_dir, exist_ok=True)

    save_fisheye_calibration(K_est, D_est, os.path.join(output_dir, "fisheye_calibration.yaml"))

    # =========================================================================
    # 10. Per-image error
    # =========================================================================
    print("\n--- 10. Per-Image Reprojection Error ---")

    total_error = 0
    for i in range(len(objpoints)):
        imgpts_reproj, _ = cv2.fisheye.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K_est, D_est
        )
        error = cv2.norm(imgpoints[i], imgpts_reproj, cv2.NORM_L2) / len(imgpts_reproj)
        total_error += error
        print(f"  Image {i+1}: {error:.4f} pixels")

    print(f"\nMean error: {total_error/len(objpoints):.4f} pixels")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - cv2.fisheye uses Kannala-Brandt model")
    print("  - 4 distortion coefficients: k1, k2, k3, k4")
    print("  - Use for FOV > 100° or severe barrel distortion")
    print("  - cv2.fisheye.calibrate(), undistortImage(), undistortPoints()")
    print("  - balance parameter controls FOV vs black borders")
    print("=" * 60)


if __name__ == "__main__":
    main()
