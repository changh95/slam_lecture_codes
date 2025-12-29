FROM slam:base

WORKDIR /workspace
COPY . /workspace/2_15

# Build the project
RUN cd /workspace/2_15 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/2_15/build

# Default command: show usage info
CMD ["sh", "-c", "echo 'Monocular Visual Odometry Tutorial' && \
    echo '' && \
    echo 'Available executables:' && \
    echo '  ./feature_tracking_demo <image_dir> [num_frames]' && \
    echo '  ./monocular_vo_demo <image_dir> [focal] [cx] [cy]' && \
    echo '  ./run_vo_kitti <image_dir> <poses_file> [max_frames]' && \
    echo '' && \
    echo 'Example with KITTI:' && \
    echo '  docker run -v /path/to/kitti:/data -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix slam_zero_to_hero:2_15 ./run_vo_kitti /data/sequences/00/image_0 /data/poses/00.txt'"]
