cmake_minimum_required(VERSION 3.10)
project(eigen_sophus_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Sophus found: ${Sophus_DIR}")

# Eigen basics
add_executable(eigen_basics examples/eigen_basics.cpp)
target_link_libraries(eigen_basics Eigen3::Eigen)

# Sophus SO(3)
add_executable(sophus_so3 examples/sophus_so3.cpp)
target_link_libraries(sophus_so3 Eigen3::Eigen Sophus::Sophus)

# Sophus SE(3)
add_executable(sophus_se3 examples/sophus_se3.cpp)
target_link_libraries(sophus_se3 Eigen3::Eigen Sophus::Sophus)

# Perturbation model
add_executable(perturbation examples/perturbation.cpp)
target_link_libraries(perturbation Eigen3::Eigen Sophus::Sophus)

message(STATUS "")
message(STATUS "=== Eigen + Sophus Tutorial ===")
message(STATUS "Executables:")
message(STATUS "  eigen_basics  - Basic Eigen operations")
message(STATUS "  sophus_so3    - SO(3) rotation operations")
message(STATUS "  sophus_se3    - SE(3) transformation operations")
message(STATUS "  perturbation  - Perturbation model for optimization")
message(STATUS "")
