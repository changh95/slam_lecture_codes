cmake_minimum_required(VERSION 3.18)
project(CUDA_ACCELERATION LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV (KLT, FAST, resize on the CPU)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d video)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Find the CUDA runtime (the examples call cudaDeviceSynchronize, cudaMallocHost, ...)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# Find vilib (installed by the Dockerfile, built for this GPU) and easy_profiler (timing)
find_package(vilib REQUIRED)
find_package(easy_profiler REQUIRED)

# The examples are plain C++: every CUDA kernel lives in libvilib.so
set(EXAMPLES pyramid_benchmark fast_benchmark tracker_benchmark)
foreach(example ${EXAMPLES})
    add_executable(${example} examples/${example}.cpp)
    target_include_directories(${example} PRIVATE include)
    target_link_libraries(${example} vilib::vilib CUDA::cudart ${OpenCV_LIBS} easy_profiler)
endforeach()

# Re-print the comparison tables of any .prof file (e.g. one recorded on a Jetson)
add_executable(prof_summary examples/prof_summary.cpp)
target_include_directories(prof_summary PRIVATE include)
target_link_libraries(prof_summary easy_profiler)

# Optional: rerun C++ SDK, for frontend_rerun (the CPU and GPU front ends on one Rerun timeline)
find_package(rerun_sdk QUIET)
if(rerun_sdk_FOUND)
    add_executable(frontend_rerun examples/frontend_rerun.cpp)
    target_include_directories(frontend_rerun PRIVATE include)
    target_compile_definitions(frontend_rerun PRIVATE FRONTEND_BLUEPRINT="${CMAKE_CURRENT_SOURCE_DIR}/blueprint.py")
    target_link_libraries(frontend_rerun vilib::vilib CUDA::cudart ${OpenCV_LIBS} easy_profiler rerun_sdk)
else()
    message(STATUS "rerun C++ SDK not found - frontend_rerun is not built")
endif()

# Print summary
message(STATUS "")
message(STATUS "=== CUDA Acceleration Hands-on (vilib vs CPU) ===")
message(STATUS "Executables:")
message(STATUS "  - pyramid_benchmark: 5-level image pyramid, vilib CPU / OpenCV vs vilib CUDA")
message(STATUS "  - fast_benchmark: FAST-9 + 32x32 grid NMS, Rosten / cv::FAST vs vilib CUDA")
message(STATUS "  - tracker_benchmark: KLT front end on EuRoC MH_01, OpenCV vs vilib CUDA")
message(STATUS "  - prof_summary: comparison tables from any .prof file")
if(rerun_sdk_FOUND)
    message(STATUS "  - frontend_rerun: the KLT front end on a whole EuRoC sequence, CPU and GPU on one Rerun timeline")
endif()
message(STATUS "")
message(STATUS "Build with: mkdir build && cd build && cmake .. && make -j")
message(STATUS "")
