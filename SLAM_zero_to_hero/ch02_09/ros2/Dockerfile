FROM osrf/ros:humble-desktop-full

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    && apt-get clean

# Create ROS2 workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Copy source
COPY src /ros2_ws/src

# Build
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Setup entrypoint
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
