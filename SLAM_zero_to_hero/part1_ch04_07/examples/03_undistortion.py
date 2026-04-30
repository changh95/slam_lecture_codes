#!/usr/bin/env python3
"""
03_undistortion.py - Image Undistortion

This example demonstrates how to remove lens distortion from images
using calibrated camera parameters.

Two methods are shown:
1. cv2.undistort() - Simple but slower for multiple images
2. cv2.initUndistortRectifyMap() + cv2.remap() - Faster for video/multiple images
"""

import cv2
import numpy as np


def create_distorted_image(
    image_size=(640, 480),
    K=None,
    dist_coeffs=None
):
    """Create a synthetic distorted image with a grid pattern."""

    if K is None:
        K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)

    if dist_coeffs is None:
        # Strong barrel distortion for visualization
        dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0], dtype=np.float64)

    # Create undistorted grid image first
    img_undist = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255

    # Draw grid lines
    for x in range(0, image_size[0], 40):
        cv2.line(img_undist, (x, 0), (x, image_size[1]-1), 0, 1)
    for y in range(0, image_size[1], 40):
        cv2.line(img_undist, (0, y), (image_size[0]-1, y), 0, 1)

    # Draw circles at intersections
    for x in range(0, image_size[0], 80):
        for y in range(0, image_size[1], 80):
            cv2.circle(img_undist, (x, y), 5, 0, -1)

    # Apply distortion by inverse mapping
    # Create meshgrid of pixel coordinates
    h, w = image_size[1], image_size[0]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize coordinates
    x_norm = (x_coords - K[0, 2]) / K[0, 0]
    y_norm = (y_coords - K[1, 2]) / K[1, 1]

    # Apply distortion model
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2

    k1, k2, p1, p2, k3 = dist_coeffs.flatten()[:5] if len(dist_coeffs.flatten()) >= 5 else (*dist_coeffs.flatten(), 0)

    # Radial distortion
    radial = 1 + k1*r2 + k2*r4

    x_dist = x_norm * radial
    y_dist = y_norm * radial

    # Tangential distortion
    x_dist += 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
    y_dist += p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm

    # Convert back to pixel coordinates
    x_dist_px = x_dist * K[0, 0] + K[0, 2]
    y_dist_px = y_dist * K[1, 1] + K[1, 2]

    # Create distorted image using remap
    map_x = x_dist_px.astype(np.float32)
    map_y = y_dist_px.astype(np.float32)

    img_dist = cv2.remap(img_undist, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return img_dist, img_undist, K, dist_coeffs


def undistort_simple(image, K, dist_coeffs):
    """
    Simple undistortion using cv2.undistort().

    This recomputes the mapping for each call - good for single images.
    """
    return cv2.undistort(image, K, dist_coeffs)


def undistort_with_maps(image, K, dist_coeffs, new_K=None):
    """
    Undistortion using precomputed maps.

    This is more efficient for video or multiple images.
    """
    h, w = image.shape[:2]

    if new_K is None:
        # Use optimal new camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (w, h), alpha=1.0
        )

    # Compute undistortion maps
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1
    )

    # Apply remapping
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return undistorted, new_K, (map_x, map_y)


def main():
    print("=" * 60)
    print("Image Undistortion")
    print("=" * 60)

    IMAGE_SIZE = (640, 480)

    # =========================================================================
    # 1. Create distorted test image
    # =========================================================================
    print("\n--- 1. Creating Distorted Test Image ---")

    img_dist, img_undist_gt, K, dist_coeffs = create_distorted_image(IMAGE_SIZE)

    print(f"Image size: {IMAGE_SIZE}")
    print(f"\nCamera matrix K:")
    print(K)
    print(f"\nDistortion coefficients:")
    print(f"  k1={dist_coeffs[0]:.4f}, k2={dist_coeffs[1]:.4f}")
    print(f"  p1={dist_coeffs[2]:.4f}, p2={dist_coeffs[3]:.4f}")

    # =========================================================================
    # 2. Simple undistortion
    # =========================================================================
    print("\n--- 2. Simple Undistortion (cv2.undistort) ---")

    img_undist1 = undistort_simple(img_dist, K, dist_coeffs)

    print("Applied cv2.undistort()")
    print("This method recomputes the mapping each call")

    cv2.imwrite("distorted.png", img_dist)
    cv2.imwrite("undistorted_simple.png", img_undist1)
    print("Saved: distorted.png, undistorted_simple.png")

    # =========================================================================
    # 3. Undistortion with precomputed maps
    # =========================================================================
    print("\n--- 3. Undistortion with Maps (cv2.remap) ---")

    img_undist2, new_K, maps = undistort_with_maps(img_dist, K, dist_coeffs)

    print("Using cv2.initUndistortRectifyMap() + cv2.remap()")
    print("\nNew camera matrix (after undistortion):")
    print(new_K)

    cv2.imwrite("undistorted_remap.png", img_undist2)
    print("Saved: undistorted_remap.png")

    # =========================================================================
    # 4. Alpha parameter effect
    # =========================================================================
    print("\n--- 4. Alpha Parameter Effect ---")

    print("""
    The alpha parameter in getOptimalNewCameraMatrix() controls
    the trade-off between field of view and black borders:

    alpha=0: All pixels in undistorted image are valid (smaller FOV)
    alpha=1: All source pixels are retained (may have black borders)
    """)

    h, w = img_dist.shape[:2]

    for alpha in [0.0, 0.5, 1.0]:
        new_K_alpha, roi = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (w, h), alpha=alpha
        )

        map_x, map_y = cv2.initUndistortRectifyMap(
            K, dist_coeffs, None, new_K_alpha, (w, h), cv2.CV_32FC1
        )

        undist_alpha = cv2.remap(img_dist, map_x, map_y, cv2.INTER_LINEAR)

        print(f"  alpha={alpha}: ROI = {roi}")
        cv2.imwrite(f"undistorted_alpha_{alpha:.1f}.png", undist_alpha)

    print("Saved: undistorted_alpha_0.0.png, undistorted_alpha_0.5.png, undistorted_alpha_1.0.png")

    # =========================================================================
    # 5. Cropping to valid region
    # =========================================================================
    print("\n--- 5. Cropping to Valid Region ---")

    new_K_crop, roi = cv2.getOptimalNewCameraMatrix(
        K, dist_coeffs, (w, h), alpha=0
    )

    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K_crop, (w, h), cv2.CV_32FC1
    )

    undist_full = cv2.remap(img_dist, map_x, map_y, cv2.INTER_LINEAR)

    # Crop to ROI
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undist_cropped = undist_full[y:y+rh, x:x+rw]
        print(f"Original size: {w}x{h}")
        print(f"Cropped size: {rw}x{rh}")
        print(f"ROI: x={x}, y={y}, w={rw}, h={rh}")

        cv2.imwrite("undistorted_cropped.png", undist_cropped)
        print("Saved: undistorted_cropped.png")
    else:
        print("ROI is empty (alpha=0 may result in no valid region)")

    # =========================================================================
    # 6. Performance comparison
    # =========================================================================
    print("\n--- 6. Performance Comparison ---")

    import time

    n_iterations = 100

    # Method 1: cv2.undistort()
    start = time.time()
    for _ in range(n_iterations):
        _ = cv2.undistort(img_dist, K, dist_coeffs)
    time1 = time.time() - start

    # Method 2: precomputed maps
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, K, (w, h), cv2.CV_32FC1
    )

    start = time.time()
    for _ in range(n_iterations):
        _ = cv2.remap(img_dist, map_x, map_y, cv2.INTER_LINEAR)
    time2 = time.time() - start

    print(f"Processing {n_iterations} images:")
    print(f"  cv2.undistort():    {time1*1000:.2f} ms total, {time1/n_iterations*1000:.3f} ms/image")
    print(f"  cv2.remap() (maps): {time2*1000:.2f} ms total, {time2/n_iterations*1000:.3f} ms/image")
    print(f"  Speedup: {time1/time2:.1f}x")

    # =========================================================================
    # 7. Undistorting points
    # =========================================================================
    print("\n--- 7. Undistorting Points ---")

    print("""
    For SLAM, we often need to undistort individual points
    rather than entire images (more efficient).
    """)

    # Create some distorted points
    distorted_pts = np.array([
        [100, 100],
        [320, 240],
        [540, 380],
        [50, 430],
        [590, 50]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Undistort points
    undistorted_pts = cv2.undistortPoints(distorted_pts, K, dist_coeffs, P=K)

    print("Distorted -> Undistorted point mapping:")
    for i in range(len(distorted_pts)):
        d = distorted_pts[i][0]
        u = undistorted_pts[i][0]
        print(f"  ({d[0]:.1f}, {d[1]:.1f}) -> ({u[0]:.1f}, {u[1]:.1f})")

    # =========================================================================
    # 8. Normalized coordinates
    # =========================================================================
    print("\n--- 8. Normalized Coordinates ---")

    print("""
    cv2.undistortPoints() without P returns normalized coordinates
    (on the ideal image plane at z=1).
    """)

    # Undistort to normalized coordinates (no P matrix)
    normalized_pts = cv2.undistortPoints(distorted_pts, K, dist_coeffs)

    print("Distorted pixels -> Normalized coordinates:")
    for i in range(len(distorted_pts)):
        d = distorted_pts[i][0]
        n = normalized_pts[i][0]
        print(f"  ({d[0]:.1f}, {d[1]:.1f}) -> ({n[0]:.4f}, {n[1]:.4f})")

    print("\nTo convert back to pixels: x_px = fx * x_norm + cx")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - cv2.undistort() for single images (simple)")
    print("  - cv2.initUndistortRectifyMap() + remap() for video (fast)")
    print("  - getOptimalNewCameraMatrix() controls FOV vs borders")
    print("  - cv2.undistortPoints() for individual points")
    print("  - Use normalized coordinates for SLAM pipelines")
    print("=" * 60)


if __name__ == "__main__":
    main()
