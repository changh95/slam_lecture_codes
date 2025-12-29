FROM osrf/ros:noetic-desktop-full

MAINTAINER changh95
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    && apt-get clean

# Create catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Copy source
COPY src /catkin_ws/src

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Setup entrypoint
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
