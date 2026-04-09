// Offline GLIM runner with easy_profiler instrumentation
// This wraps GLIM's offline_viewer or creates a minimal pipeline runner
#include <iostream>
#include <csignal>

#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#endif

static volatile bool running = true;

void signal_handler(int sig) {
  running = false;
#ifdef BUILD_WITH_EASY_PROFILER
  profiler::dumpBlocksToFile("/output/glim_profiler.prof");
  std::cout << "Profiler data saved to /output/glim_profiler.prof" << std::endl;
#endif
}

// If GLIM has an offline_viewer or glim_rosbag, wrap it.
// Otherwise this serves as the profiler init/dump entry point.
// The actual GLIM pipeline is invoked as a library.
//
// Usage (when BUILD_WITH_EASY_PROFILER=ON):
//   The profiler is enabled at startup and data is flushed to
//   /output/glim_profiler.prof on SIGINT/SIGTERM.
//   Load the resulting .prof file with easy_profiler GUI or
//   convert it with the convert_prof.sh script in this repo.
//
// Integration note:
//   GLIM is a library with no standalone main(). This file is
//   intended to be linked by ROS nodes (glim_ros) or the
//   offline rosbag runner. The signal handler and profiler
//   init/dump logic here should be called from the actual
//   entry point before and after the GLIM pipeline runs.

void glim_profiler_init() {
#ifdef BUILD_WITH_EASY_PROFILER
  EASY_PROFILER_ENABLE;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  std::cout << "easy_profiler enabled. Send SIGINT to dump profile data." << std::endl;
#endif
}

void glim_profiler_dump(const char* output_path) {
#ifdef BUILD_WITH_EASY_PROFILER
  profiler::dumpBlocksToFile(output_path);
  std::cout << "Profiler data saved to " << output_path << std::endl;
#else
  (void)output_path;
#endif
}
