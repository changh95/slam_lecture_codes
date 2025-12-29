FROM slam:base

WORKDIR /workspace

# Copy project files
COPY . /workspace/2_29

# Build the project
RUN cd /workspace/2_29 && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for results
RUN mkdir -p /output

WORKDIR /workspace/2_29/build

# Default command: show usage
CMD ["sh", "-c", "echo '=== ICP using PCL Tutorial ===' && \
    echo '' && \
    echo 'Available executables:' && \
    echo '  ./icp_basic source.pcd target.pcd' && \
    echo '  ./icp_basic --generate' && \
    echo '  ./icp_point_to_plane source.pcd target.pcd' && \
    echo '  ./icp_point_to_plane --generate' && \
    echo '  ./icp_visualization source.pcd target.pcd' && \
    echo '  ./icp_visualization --generate [--step]' && \
    echo '  ./lidar_odometry /path/to/velodyne/' && \
    echo '  ./lidar_odometry --generate' && \
    echo '' && \
    echo 'Example with KITTI dataset:' && \
    echo '  docker run -it --rm \\' && \
    echo '    -v /path/to/kitti:/data \\' && \
    echo '    slam_zero_to_hero:2_29 \\' && \
    echo '    ./lidar_odometry /data/sequences/00/velodyne/' && \
    echo '' && \
    echo 'Example with visualization (requires X11):' && \
    echo '  docker run -it --rm \\' && \
    echo '    -e DISPLAY=\$DISPLAY \\' && \
    echo '    -v /tmp/.X11-unix:/tmp/.X11-unix \\' && \
    echo '    slam_zero_to_hero:2_29 \\' && \
    echo '    ./icp_visualization --generate' && \
    echo '' && \
    /bin/bash"]
