cmake_minimum_required(VERSION 3.10)
project(dbow2_global_features LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Find DBoW2
# DBoW2 does not provide a CMake config, so we specify paths manually
# When built from source and installed to /usr/local:
set(DBoW2_INCLUDE_DIRS "/usr/local/include")
set(DBoW2_LIBS "/usr/local/lib/libDBoW2.so")

# Check if DBoW2 library exists
if(NOT EXISTS ${DBoW2_LIBS})
    # Try alternative locations
    if(EXISTS "/usr/lib/libDBoW2.so")
        set(DBoW2_LIBS "/usr/lib/libDBoW2.so")
    elseif(EXISTS "/usr/local/lib/libDBoW2.a")
        set(DBoW2_LIBS "/usr/local/lib/libDBoW2.a")
    else()
        message(FATAL_ERROR "DBoW2 library not found! Please build and install DBoW2 first.")
    endif()
endif()

message(STATUS "DBoW2 include dirs: ${DBoW2_INCLUDE_DIRS}")
message(STATUS "DBoW2 library: ${DBoW2_LIBS}")

include_directories(${DBoW2_INCLUDE_DIRS})

# Vocabulary Training Example
add_executable(vocabulary_training examples/vocabulary_training.cpp)
target_link_libraries(vocabulary_training
    ${OpenCV_LIBS}
    ${DBoW2_LIBS}
)

# Loop Closure Detection Example
add_executable(loop_closure_detection examples/loop_closure_detection.cpp)
target_link_libraries(loop_closure_detection
    ${OpenCV_LIBS}
    ${DBoW2_LIBS}
)

# Print build summary
message(STATUS "")
message(STATUS "=== DBoW2 Global Features Tutorial ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  vocabulary_training     - Create ORB vocabulary from images")
message(STATUS "  loop_closure_detection  - Loop closure detection demo")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j4")
message(STATUS "")
