cmake_minimum_required(VERSION 3.10)
project(rpgo_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(GTSAM REQUIRED)
find_package(KimeraRPGO REQUIRED)

include_directories(
    ${GTSAM_INCLUDE_DIR}
)

# Example 1: RPGO Basics
add_executable(rpgo_basics
    examples/rpgo_basics.cpp)
target_link_libraries(rpgo_basics
    gtsam
    KimeraRPGO)

# Example 2: Outlier Rejection
add_executable(rpgo_outlier_rejection
    examples/rpgo_outlier_rejection.cpp)
target_link_libraries(rpgo_outlier_rejection
    gtsam
    KimeraRPGO)
