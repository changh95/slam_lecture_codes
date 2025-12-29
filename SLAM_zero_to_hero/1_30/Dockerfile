FROM osrf/ros:noetic-desktop-full

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    autoconf \
    automake \
    python3-pip \
    python3-scipy \
    python3-matplotlib \
    python3-wxgtk4.0 \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    doxygen \
    libopencv-dev \
    libpoco-dev \
    libtbb-dev \
    libblas-dev \
    liblapack-dev \
    libv4l-dev \
    python3-catkin-tools \
    && apt-get clean

# Create catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Initialize catkin workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Clone Kalibr
RUN cd /catkin_ws/src && \
    git clone https://github.com/ethz-asl/kalibr.git

# Clone Allan Variance ROS (for IMU calibration)
RUN cd /catkin_ws/src && \
    git clone https://github.com/ori-drs/allan_variance_ros.git

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin build -j4"

# Create data directory
RUN mkdir -p /data

# Setup environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

# Set environment variable for focal length initialization
ENV KALIBR_MANUAL_FOCAL_LENGTH_INIT=1

WORKDIR /catkin_ws

CMD ["/bin/bash"]
