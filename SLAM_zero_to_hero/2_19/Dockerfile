FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_19

# Install OpenCV with aruco module (from contrib)
# The base image may have OpenCV 4.x, but we need contrib modules for aruco
RUN apt-get update && apt-get install -y \
    libopencv-contrib-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build the project
RUN cd /workspace/2_19 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Generate sample markers for testing
RUN cd /workspace/2_19/build && \
    ./marker_detection --generate --id 0 --size 200 --output /workspace/2_19/data/marker_0.png && \
    ./marker_detection --generate --id 1 --size 200 --output /workspace/2_19/data/marker_1.png && \
    ./marker_detection --generate --id 42 --size 200 --output /workspace/2_19/data/marker_42.png && \
    ./charuco_calibration --generate --cols 5 --rows 7 --board-output /workspace/2_19/data/charuco_board.png

# Create a sample calibration file for testing (default values)
RUN echo "%YAML:1.0" > /workspace/2_19/data/sample_calib.yaml && \
    echo "image_width: 640" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "image_height: 480" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "camera_matrix: !!opencv-matrix" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   rows: 3" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   cols: 3" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   dt: d" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   data: [ 600., 0., 320., 0., 600., 240., 0., 0., 1. ]" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "distortion_coefficients: !!opencv-matrix" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   rows: 5" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   cols: 1" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   dt: d" >> /workspace/2_19/data/sample_calib.yaml && \
    echo "   data: [ 0., 0., 0., 0., 0. ]" >> /workspace/2_19/data/sample_calib.yaml

# Create sample marker map file (known marker world positions)
RUN echo "# Marker world positions: id tx ty tz qx qy qz qw" > /workspace/2_19/data/marker_map.txt && \
    echo "# Marker 0 at origin" >> /workspace/2_19/data/marker_map.txt && \
    echo "0 0.0 0.0 0.0 0.0 0.0 0.0 1.0" >> /workspace/2_19/data/marker_map.txt && \
    echo "# Marker 1 at 1m along X axis" >> /workspace/2_19/data/marker_map.txt && \
    echo "1 1.0 0.0 0.0 0.0 0.0 0.0 1.0" >> /workspace/2_19/data/marker_map.txt && \
    echo "# Marker 42 at 2m along X, 1m along Y" >> /workspace/2_19/data/marker_map.txt && \
    echo "42 2.0 1.0 0.0 0.0 0.0 0.0 1.0" >> /workspace/2_19/data/marker_map.txt

WORKDIR /workspace/2_19/build
