cmake_minimum_required(VERSION 3.10...3.18)

if(NOT TARGET gtest_main)
  add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif(NOT TARGET gtest_main)

add_library(basalt-headers-test-utils INTERFACE)
add_library(basalt::basalt-headers-test-utils ALIAS basalt-headers-test-utils)
target_link_libraries(basalt-headers-test-utils INTERFACE gtest)

target_include_directories(basalt-headers-test-utils INTERFACE include)

target_compile_definitions(basalt-headers-test-utils INTERFACE BASALT_ENABLE_BOUNDS_CHECKS)

# List all header files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
target_sources(basalt-headers-test-utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/test_utils.h)

add_executable(test_image src/test_image.cpp)
target_link_libraries(test_image gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_spline src/test_spline.cpp)
target_link_libraries(test_spline gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_spline_se3 src/test_spline_se3.cpp)
target_link_libraries(test_spline_se3 gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_camera src/test_camera.cpp)
target_link_libraries(test_camera gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_sophus src/test_sophus.cpp)
target_link_libraries(test_sophus gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_preintegration src/test_preintegration.cpp)
target_link_libraries(test_preintegration gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

add_executable(test_ceres_spline_helper src/test_ceres_spline_helper.cpp)
target_link_libraries(test_ceres_spline_helper gtest_main basalt::basalt-headers-test-utils basalt::basalt-headers)

# benchmarks (currently doesnt work on macOS and with clang)
if(NOT APPLE AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "ENABLE tests")
  set(BENCHMARK_ENABLE_EXCEPTIONS OFF CACHE BOOL "ENABLE exceptions")
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "ENABLE gtests")
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "ENABLE install")
  add_subdirectory(benchmark EXCLUDE_FROM_ALL)

  add_executable(benchmark_camera src/benchmark_camera.cpp)
  target_link_libraries(benchmark_camera benchmark::benchmark basalt::basalt-headers)
endif()

include(GoogleTest) # for gtest_discover_test

enable_testing()

gtest_discover_tests(test_image)
gtest_discover_tests(test_spline)
gtest_discover_tests(test_spline_se3)
gtest_discover_tests(test_camera)
gtest_discover_tests(test_sophus)
gtest_discover_tests(test_preintegration)
gtest_discover_tests(test_ceres_spline_helper)
