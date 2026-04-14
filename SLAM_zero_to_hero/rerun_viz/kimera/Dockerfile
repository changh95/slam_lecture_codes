# Kimera-VIO-ROS + Rerun visualization
# Base: ROS Noetic on Ubuntu 20.04 (Python 3.8)
FROM docker.io/ros:noetic
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
    build-essential cmake git wget python3 python3-pip \
    libboost-all-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev libtbb-dev \
    libopencv-dev ros-noetic-cv-bridge \
    ros-noetic-tf2 ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs ros-noetic-tf2-eigen \
    ros-noetic-pcl-ros ros-noetic-eigen-conversions \
    ros-noetic-image-transport \
    ros-noetic-visualization-msgs ros-noetic-interactive-markers && \
    apt-get clean

# Rerun Python SDK + numpy
# Ubuntu 20.04 (noetic) ships Python 3.8; pin rerun-sdk to 0.21.0
# (last version compatible with Python 3.8).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir numpy "rerun-sdk==0.21.0"

# Build GTSAM 4.2
RUN git clone https://github.com/borglab/gtsam.git -b 4.2 && \
    cd gtsam && mkdir build && cd build && \
    cmake .. -DGTSAM_BUILD_TESTS=OFF \
             -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
             -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf gtsam

# Install Eigen 3.4.1 from source
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.zip && \
    unzip -q eigen-3.4.1.zip && cd eigen-3.4.1 && \
    mkdir build && cd build && cmake .. && make install && \
    cd / && rm -rf eigen-3.4.1 eigen-3.4.1.zip

# Build OpenGV
RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && \
    rm -f modules/FindEigen.cmake && \
    sed -i 's/find_package(Eigen REQUIRED)/find_package(Eigen3 REQUIRED)/' CMakeLists.txt && \
    sed -i 's/\${EIGEN_INCLUDE_DIRS}/\${EIGEN3_INCLUDE_DIR}/g' CMakeLists.txt && \
    sed -i 's/\${EIGEN_INCLUDE_DIR}/\${EIGEN3_INCLUDE_DIR}/g' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf opengv

# Build DBoW2
RUN git clone https://github.com/dorian3d/DBoW2.git && \
    cd DBoW2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf DBoW2

# Build Kimera-RPGO
RUN git clone https://github.com/MIT-SPARK/Kimera-RPGO.git && \
    cd Kimera-RPGO && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig && \
    cd / && rm -rf Kimera-RPGO

# Build Kimera-VIO + Kimera-VIO-ROS inside a catkin workspace.
# Kimera-VIO must live INSIDE the workspace so that rospack/find can
# resolve `$(find kimera_vio)/params/...` in the ROS launch files at
# runtime (params/ is only shipped with the source tree).
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/MIT-SPARK/Kimera-VIO.git --recursive && \
    git clone https://github.com/catkin/catkin_simple.git && \
    git clone https://github.com/MIT-SPARK/pose_graph_tools.git -b archive/ros_noetic && \
    git clone https://github.com/MIT-SPARK/Kimera-VIO-ROS.git

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-catkin-tools && rm -rf /var/lib/apt/lists/*

# EuRoC params default to ground-truth-based initialisation, which
# requires a gt_odom_topic that doesn't exist in the MH_*_easy bags.
# Switch to IMU-based auto-init (1) so we can run on any bag.
RUN sed -i 's/^autoInitialize: 0/autoInitialize: 1/' \
    /catkin_ws/src/Kimera-VIO/params/Euroc/BackendParams.yaml

# Gzip ORBvoc.yml so OpenCV FileStorage reads less from disk.
# The uncompressed YAML is 270 MB and takes many minutes to parse on ARM;
# the gzipped version (~60 MB) is decoded in-memory and cuts the load
# time significantly. FileStorage auto-detects the .yml.gz extension.
RUN gzip -k -9 /catkin_ws/src/Kimera-VIO/vocabulary/ORBvoc.yml && \
    sed -i 's|vocabulary/ORBvoc.yml|vocabulary/ORBvoc.yml.gz|g' \
        /catkin_ws/src/Kimera-VIO-ROS/launch/kimera_vio_ros.launch \
        /catkin_ws/src/Kimera-VIO-ROS/launch/*.launch || true

# Patch RosVisualizer.cpp to publish the frontend feature_tracks image
# to the ROS topic. Upstream only pushes mesh images into
# images_to_display_ (via the OpenCV visualizer), so the
# /kimera_vio_ros/frontend/feature_tracks topic stays silent when
# using the RosVisualizer pipeline. This patch casts the frontend
# output to StereoFrontendOutput and publishes the feature_tracks cv::Mat.
# Patch StereoVisionImuFrontend to always push the stereo-matches image
# to the display queue (as "stereo_matches") so RosDisplay publishes it.
# Upstream only generates this image when logger_ + FLAGS_save_frontend_images
# is set, and it only writes to disk.
RUN python3 <<'PYEOF'
path = "/catkin_ws/src/Kimera-VIO/src/frontend/StereoVisionImuFrontend.cpp"
with open(path) as f:
    src = f.read()

# Add a helper call right after the feature_tracks displayImage block.
old = """    if (display_queue_ && FLAGS_visualize_feature_tracks) {
      displayImage(stereoFrame_k_->timestamp_,
                   \"feature_tracks\",
                   tracker_->getTrackerImage(stereoFrame_lkf_->left_frame_,
                                             stereoFrame_k_->left_frame_),
                   display_queue_);
    }"""

new = old + """

    // --- PATCH: build + publish stereo L/R match image ---
    if (display_queue_) {
      const Frame& lfk = stereoFrame_k_->left_frame_;
      const Frame& rfk = stereoFrame_k_->right_frame_;
      if (lfk.img_.cols == rfk.img_.cols &&
          lfk.img_.rows == rfk.img_.rows) {
        DMatchVec match_vec;
        const StatusKeypointsCV& rkps =
            stereoFrame_k_->right_keypoints_rectified_;
        if (lfk.keypoints_.size() == rfk.keypoints_.size()) {
          for (size_t i = 0; i < lfk.keypoints_.size(); ++i) {
            if (lfk.landmarks_[i] != -1 &&
                rkps[i].first == KeypointStatus::VALID) {
              match_vec.push_back(cv::DMatch(i, i, 0));
            }
          }
        }
        cv::Mat img_lr = UtilsOpenCV::DrawCornersMatches(
            tracker_->getTrackerImage(stereoFrame_lkf_->left_frame_, lfk),
            lfk.keypoints_, rfk.img_, rfk.keypoints_, match_vec, false);
        displayImage(stereoFrame_k_->timestamp_,
                     \"stereo_matches\",
                     img_lr,
                     display_queue_);
      }
    }"""

if old not in src:
    raise SystemExit("StereoVisionImuFrontend displayImage block not found")
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("StereoVisionImuFrontend.cpp patched to publish stereo_matches image")
PYEOF

RUN python3 <<'PYEOF'
path = "/catkin_ws/src/Kimera-VIO-ROS/src/RosVisualizer.cpp"
with open(path) as f:
    src = f.read()

old = """void RosVisualizer::publishFrontendOutput(
    const FrontendOutputPacketBase::ConstPtr& output) const {
  CHECK(output);
  if (frontend_stats_pub_.getNumSubscribers() > 0) {
    publishFrontendStats(output);
  }
}"""

new = """void RosVisualizer::publishFrontendOutput(
    const FrontendOutputPacketBase::ConstPtr& output) const {
  CHECK(output);
  if (frontend_stats_pub_.getNumSubscribers() > 0) {
    publishFrontendStats(output);
  }
  // --- PATCH: publish feature_tracks image from StereoFrontendOutput ---
  auto stereo_output =
      std::dynamic_pointer_cast<const StereoFrontendOutput>(output);
  if (stereo_output && !stereo_output->feature_tracks_.empty() &&
      image_publishers_) {
    std_msgs::Header header;
    header.stamp.fromNSec(stereo_output->timestamp_);
    header.frame_id = base_link_frame_id_;
    image_publishers_->publish(
        "feature_tracks",
        cv_bridge::CvImage(header, "bgr8", stereo_output->feature_tracks_)
            .toImageMsg());
  }
}"""

if old not in src:
    raise SystemExit("RosVisualizer publishFrontendOutput block not found")
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("RosVisualizer.cpp patched to publish feature_tracks image")
PYEOF

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && \
    catkin init && \
    catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    catkin build

RUN sed -i "6i source \"/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh

# Rerun bridge + entrypoint
COPY ros_rerun_bridge.py /app/ros_rerun_bridge.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
VOLUME /data
WORKDIR /data
EXPOSE 9090 9877
ENTRYPOINT ["/app/entrypoint.sh"]
