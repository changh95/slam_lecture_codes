# Source-only image that demo Dockerfiles copy the plugin from, so each demo
# keeps building with its own folder as the build context:
#   podman build -t slam_zero_to_hero:rviz_unified_controls rviz_unified_controls
# then, in a demo Dockerfile:
#   COPY --from=localhost/slam_zero_to_hero:rviz_unified_controls /rviz_unified_controls /opt/rviz_unified_controls
#   RUN /opt/rviz_unified_controls/install_ros1.sh /opt/rviz_unified_controls   # or install_ros2.sh
FROM scratch
COPY . /rviz_unified_controls
