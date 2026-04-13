#!/bin/bash
# Patch glim_rosbag.cpp to add easy_profiler init/dump and signal handler
set -e
FILE="$1"

# Add profiler includes and signal handler at the top
cat > /tmp/profiler_header.cpp << 'HEADER'
#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#include <csignal>
#include <cstdlib>
static void profiler_signal_handler(int sig) {
  profiler::dumpBlocksToFile("/output/glim_profiler.prof");
  fprintf(stderr, "Profiler data saved to /output/glim_profiler.prof\n");
  exit(0);
}
#endif
HEADER

# Prepend header
cat /tmp/profiler_header.cpp "$FILE" > /tmp/patched.cpp

# Add EASY_PROFILER_ENABLE and signal handlers after ros::init
sed -i '/ros::init/a\
#ifdef BUILD_WITH_EASY_PROFILER\
  EASY_PROFILER_ENABLE;\
  signal(SIGTERM, profiler_signal_handler);\
  signal(SIGINT, profiler_signal_handler);\
#endif' /tmp/patched.cpp

# Add profiler dump before return 0
sed -i '/return 0/i\
#ifdef BUILD_WITH_EASY_PROFILER\
  profiler::dumpBlocksToFile("/output/glim_profiler.prof");\
#endif' /tmp/patched.cpp

cp /tmp/patched.cpp "$FILE"
echo "Patched $FILE with easy_profiler support"
