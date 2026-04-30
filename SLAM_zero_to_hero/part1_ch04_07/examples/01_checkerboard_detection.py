#!/usr/bin/env python3
"""
01_checkerboard_detection.py - Checkerboard Corner Detection

This example demonstrates how to detect checkerboard corners
for camera calibration using OpenCV.

The checkerboard pattern is commonly used because:
1. Corners are easy to detect precisely
2. Known geometry (regular grid)
3. Easy to manufacture accurately
"""

import cv2
import numpy as np
import os


def create_synthetic_checkerboard_image(
    board_size=(9, 6),
    square_size=50,
    border=100,
    noise_level=5
):
    """Create a synthetic checkerboard image for testing."""

    rows, cols = board_size
    img_height = (rows + 1) * square_size + 2 * border
    img_width = (cols + 1) * square_size + 2 * border

    # Create white image
    img = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Draw checkerboard squares
    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                x1 = border + j * square_size
                y1 = border + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), 0, -1)

    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def detect_checkerboard_corners(image, board_size=(9, 6)):
    """
    Detect checkerboard corners in an image.

    Args:
        image: Grayscale or color image
        board_size: (columns, rows) of inner corners

    Returns:
        success: True if corners were found
        corners: Detected corner coordinates (Nx1x2)
    """

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find checkerboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    success, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if success:
        # Refine corner positions to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return success, corners


def main():
    print("=" * 60)
    print("Checkerboard Corner Detection")
    print("=" * 60)

    # Board configuration
    BOARD_SIZE = (9, 6)  # Number of inner corners (columns, rows)

    print(f"\nBoard size: {BOARD_SIZE[0]} x {BOARD_SIZE[1]} inner corners")

    # =========================================================================
    # 1. Create synthetic checkerboard image
    # =========================================================================
    print("\n--- 1. Creating Synthetic Checkerboard ---")

    img = create_synthetic_checkerboard_image(board_size=BOARD_SIZE)
    print(f"Created image: {img.shape}")

    # =========================================================================
    # 2. Detect corners
    # =========================================================================
    print("\n--- 2. Detecting Corners ---")

    success, corners = detect_checkerboard_corners(img, BOARD_SIZE)

    if success:
        print(f"SUCCESS: Found {len(corners)} corners")
        print(f"Corner array shape: {corners.shape}")

        # Print first few corners
        print("\nFirst 5 corner positions:")
        for i in range(min(5, len(corners))):
            print(f"  Corner {i}: ({corners[i][0][0]:.2f}, {corners[i][0][1]:.2f})")
    else:
        print("FAILED: Could not find corners")

    # =========================================================================
    # 3. Draw corners
    # =========================================================================
    print("\n--- 3. Drawing Corners ---")

    # Convert to color for visualization
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if success:
        # Draw detected corners
        cv2.drawChessboardCorners(img_color, BOARD_SIZE, corners, success)

        # Save result
        cv2.imwrite("checkerboard_detected.png", img_color)
        print("Saved: checkerboard_detected.png")
    else:
        print("Cannot draw corners - detection failed")

    # =========================================================================
    # 4. Different detection scenarios
    # =========================================================================
    print("\n--- 4. Testing Different Scenarios ---")

    scenarios = [
        ("Clean image", 0),
        ("Low noise", 10),
        ("Medium noise", 30),
        ("High noise", 50),
    ]

    for name, noise in scenarios:
        test_img = create_synthetic_checkerboard_image(
            board_size=BOARD_SIZE,
            noise_level=noise
        )
        success, _ = detect_checkerboard_corners(test_img, BOARD_SIZE)
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name:20s} (noise={noise:2d}): {status}")

    # =========================================================================
    # 5. Sub-pixel accuracy
    # =========================================================================
    print("\n--- 5. Sub-pixel Refinement ---")

    print("""
    cornerSubPix() refines corner positions to sub-pixel accuracy.

    Parameters:
    - winSize: Half of the search window size (e.g., (11,11))
    - zeroZone: Half of the dead region (-1,-1 to disable)
    - criteria: Termination criteria (iterations, epsilon)

    This is crucial for accurate calibration!
    """)

    # Compare with and without refinement
    success, corners_raw = cv2.findChessboardCorners(img, BOARD_SIZE, None)

    if success:
        # Without refinement
        print("Sample corner without refinement:")
        print(f"  Position: ({corners_raw[0][0][0]:.4f}, {corners_raw[0][0][1]:.4f})")

        # With refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            img, corners_raw.copy(), (11, 11), (-1, -1), criteria
        )
        print("Same corner with sub-pixel refinement:")
        print(f"  Position: ({corners_refined[0][0][0]:.4f}, {corners_refined[0][0][1]:.4f})")

    # =========================================================================
    # 6. Object points
    # =========================================================================
    print("\n--- 6. Object Points ---")

    print("""
    For calibration, we also need 3D object points.
    These are the real-world coordinates of the corners.

    Assuming the board is at Z=0 and square size is known:
    """)

    square_size = 0.025  # 25mm squares in meters

    # Generate object points
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= square_size

    print(f"Object points shape: {objp.shape}")
    print(f"Square size: {square_size * 1000:.1f} mm")
    print("\nFirst 5 object points (in meters):")
    for i in range(5):
        print(f"  Point {i}: ({objp[i][0]:.4f}, {objp[i][1]:.4f}, {objp[i][2]:.4f})")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - Use cv2.findChessboardCorners() for detection")
    print("  - Use cv2.cornerSubPix() for sub-pixel accuracy")
    print("  - Use cv2.drawChessboardCorners() for visualization")
    print("  - Board size = number of INNER corners")
    print("  - Object points define the 3D coordinates")
    print("=" * 60)


if __name__ == "__main__":
    main()
