cmake_minimum_required(VERSION 3.10...3.18)

include(CMakePackageConfigHelpers)

project(basalt-headers VERSION 0.1.0 LANGUAGES CXX)

function(ignore_external_warnings target)

  # Silence compile warnings
  get_target_property(_target_type ${target} TYPE)
  if(NOT _target_type STREQUAL "INTERFACE_LIBRARY")
    target_compile_options(${target} PRIVATE "-w")
  endif()

  # Silence include warnings
  get_target_property(_includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
  if(_includes)
    target_include_directories(${target} SYSTEM INTERFACE ${_includes})
  endif()

endfunction(ignore_external_warnings)

include(GNUInstallDirs)
include(CTest) # note: this adds a BUILD_TESTING which defaults to ON

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  if(NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
      message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
      set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
      set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    else()
      message(STATUS "Dind't find ccache")
    endif()
  else()
    message(STATUS "Compiler launcher already set. Not configuring ccache.")
    message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
    message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
  endif()

  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  if(NOT CXX_MARCH)
    set(CXX_MARCH native)
  endif()

  if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(BASALT_HEADERS_MARCH_FLAGS "-march=${CXX_MARCH}")
  else()
    message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # cmake default: "-g"
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN") # cmake default: "-O2 -g -DNDEBUG"
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG") # cmake default: "-O3 -DNDEBUG"
  set(CMAKE_CXX_FLAGS "-Wall -Wextra ${BASALT_HEADERS_MARCH_FLAGS} ${CMAKE_CXX_FLAGS} ${BASALT_HEADERS_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_COVERAGE
      "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline"
  )
  set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
  set(CMAKE_CXX_FLAGS_SANITIZERDEBUG
      "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined"
  )
  set(CMAKE_CXX_FLAGS_SANITIZERRELWITHDEBINFO
      "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined"
  )

endif()

function(report_dependency_location DEP_NAME DEP_TARGET)
  if(${DEP_NAME}_DIR)
    # this is set if dependency is found via find_package
    set(ACTUAL_DEP_CMAKE_DIR ${${DEP_NAME}_DIR})
  else()
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0")
      get_target_property(ACTUAL_DEP_CMAKE_DIR ${DEP_TARGET} SOURCE_DIR)
    else()
      # SOURCE_DIR works for interface targets only since 3.19
      set(ACTUAL_DEP_CMAKE_DIR "")
    endif()
  endif()
  get_target_property(ACTUAL_DEP_INCLUDE_DIR ${DEP_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Using ${DEP_NAME} from: ${ACTUAL_DEP_CMAKE_DIR} (include: ${ACTUAL_DEP_INCLUDE_DIR})")
endfunction()

option(BASALT_BUILTIN_EIGEN "Use builtin Eigen from submodule" ON)
if(NOT TARGET Eigen3::Eigen)
  if(BASALT_BUILTIN_EIGEN)
    message(STATUS "Including internal Eigen from submodule")
    # Disable Eigen tests
    set(BUILD_TESTING_PREV_VALUE ${BUILD_TESTING}
    )# Note: BUILD_TESTING is always defined because we include CTest above
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/eigen EXCLUDE_FROM_ALL)
    # Restore previous value of BUILD_TESTING
    set(BUILD_TESTING ${BUILD_TESTING_PREV_VALUE} CACHE BOOL "Build the testing tree." FORCE)
  else()
    message(STATUS "Finding external Eigen")
    find_package(Eigen3 3.4 REQUIRED)
  endif()
else()
  message(STATUS "Eigen already available")
endif()
report_dependency_location(Eigen3 Eigen3::Eigen)
ignore_external_warnings(eigen)

option(BASALT_BUILTIN_SOPHUS "Use builtin Sophus from submodule" ON)
if(NOT TARGET Sophus::Sophus)
  if(BASALT_BUILTIN_SOPHUS)
    message(STATUS "Including internal Sophus from submodule")
    set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "disable BUILD_SOPHUS_TESTS")
    set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "disable BUILD_SOPHUS_EXAMPLES")
    add_subdirectory(thirdparty/Sophus EXCLUDE_FROM_ALL)
  else()
    message(STATUS "Finding external Sophus")
    find_package(Sophus REQUIRED)
  endif()
else()
  message(STATUS "Sophus already available")
endif()
report_dependency_location(Sophus Sophus::Sophus)
ignore_external_warnings(sophus)

option(BASALT_BUILTIN_CEREAL "Use builtin Cereal from submodule" ON)
if(NOT TARGET cereal::cereal)
  if(BASALT_BUILTIN_CEREAL)
    message(STATUS "Including internal Cereal from submodule")
    add_subdirectory(thirdparty/cereal EXCLUDE_FROM_ALL)
  else()
    message(STATUS "Finding external Cereal")
    find_package(cereal REQUIRED)
  endif()
else()
  message(STATUS "Cereal already available")
endif()
report_dependency_location(cereal cereal::cereal)
ignore_external_warnings(cereal)

add_library(basalt-headers INTERFACE)
add_library(basalt::basalt-headers ALIAS basalt-headers)
target_link_libraries(basalt-headers INTERFACE Eigen3::Eigen Sophus::Sophus cereal::cereal)

# Associate target with include directory
target_include_directories(
  basalt-headers INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# List all header files with target_sources.
# This ensures, e.g., that QtCreator >= 6.0 properly recognize these headers to belong to the project.
# To support cmake < 3.13, use absolute paths (see: https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)
target_sources(
  basalt-headers
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/calib_bias.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/calibration/calibration.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/bal_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/camera_static_assert.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/double_sphere_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/extended_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/fov_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/generic_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/kannala_brandt_camera4.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/pinhole_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/stereographic_param.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/camera/unified_camera.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/image/image.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/image/image_pyr.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/imu/imu_types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/imu/preintegration.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/serialization/eigen_io.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/serialization/headers_serialization.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/spline/ceres_local_param.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/spline/ceres_spline_helper.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/spline/rd_spline.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/spline/se3_spline.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/spline/so3_spline.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/spline/spline_common.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/assert.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/eigen_utils.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/hash.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/basalt/utils/sophus_utils.hpp
)

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR BASALT_HEADERS_BUILD_TESTING) AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
