cmake_minimum_required(VERSION 3.10...3.18)

add_library(nlohmann::json INTERFACE IMPORTED GLOBAL)
set_property(TARGET nlohmann::json PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/json/)

add_subdirectory(basalt-headers EXCLUDE_FROM_ALL)
add_subdirectory(ros EXCLUDE_FROM_ALL)

add_subdirectory(apriltag EXCLUDE_FROM_ALL)
ignore_external_warnings(apriltag)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Enable BUILD_SHARED_LIBS")
set(BUILD_TESTS OFF CACHE BOOL "Enable BUILD_TESTS")
set(BUILD_TOOLS OFF CACHE BOOL "Enable BUILD_TOOLS")
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)

set(BUILD_PANGOLIN_LIBOPENEXR OFF CACHE BOOL "Enable BUILD_PANGOLIN_LIBOPENEXR")
set(BUILD_PANGOLIN_PYTHON OFF CACHE BOOL "Enable BUILD_PANGOLIN_PYTHON")
set(BUILD_EXAMPLES OFF CACHE BOOL "Enable BUILD_EXAMPLES")
set(BUILD_PANGOLIN_REALSENSE OFF CACHE BOOL "Enable librealsense")
set(BUILD_PANGOLIN_REALSENSE2 OFF CACHE BOOL "Enable librealsense2")
set(BUILD_PANGOLIN_LIBTIFF OFF CACHE BOOL "Enable libtiff")

# disable ffmpeg b/c build is broken on macOS since 2022-02
# see: https://github.com/stevenlovegrove/Pangolin/issues/737
set(BUILD_PANGOLIN_FFMPEG OFF CACHE BOOL "Build support for ffmpeg video input")

set(EIGEN_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}")
set(EIGEN_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")

# Hack to disable march=native in opengv
macro(add_definitions)
  if(NOT ${ARGV0} STREQUAL "-march=native")
    _add_definitions(${ARGN})
  endif()
endmacro()
add_subdirectory(opengv EXCLUDE_FROM_ALL)

# fix opengv: c++17 and debug postfix
set_target_properties(opengv PROPERTIES CXX_STANDARD 17 DEBUG_POSTFIX "")

# fix opengv: compile options (CMAKE_CXX_FLAGS is overwritten by Opengv)
target_compile_options(opengv PRIVATE ${BASALT_MARCH_FLAGS} "SHELL:${BASALT_PASSED_CXX_FLAGS}")
ignore_external_warnings(opengv)

# Hack to disable CPack in Pangolin.
macro(include)
  if(NOT ${ARGV0} STREQUAL "CPack")
    _include(${ARGN})
  endif()
endmacro()

# Add pangolin and create an INTERFACE SYSTEM library out of it
# TODO: For cmake 3.25, use the SYSTEM parameter of add_subidrectory to avoid adding SYSTEM and silencing warnings explicitly
add_subdirectory(Pangolin EXCLUDE_FROM_ALL)

get_property(pangolin_targets DIRECTORY Pangolin PROPERTY BUILDSYSTEM_TARGETS)
foreach(_target ${pangolin_targets})
  ignore_external_warnings(${_target})
endforeach()

add_library(pangolin INTERFACE)
target_link_libraries(pangolin INTERFACE ${pangolin_targets})
target_include_directories(pangolin SYSTEM INTERFACE ${pangolin_targets})

# TODO: enable once mpark issue is fixed upstream pangolin
#set_target_properties(pangolin PROPERTIES
#  CXX_STANDARD 17)

# check here the directories for the pangolin and opengv targets, and
# confirm that the eigen-related include dirs match.
foreach(_target opengv pangolin)
  get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_dir IN LISTS _include_dirs)
    if("${_dir}" MATCHES ".*/eigen3?(/unsupported)?$")
      if(NOT _dir MATCHES "^${EIGEN3_INCLUDE_DIR}.*")

      endif()
    endif()
  endforeach()
endforeach()

# opengv and pangolin assume that eigen is found outside the source
# directory and thus include it in INTERFACE_INCLUDE_DIRECTORIES,
# which makes cmake complain that that property contains paths in the
# source folder. Thus, we update the property to only include these
# eigen paths in the "BUILD_INTERFACE" (not "INSTALL").
if(EIGEN3_INCLUDE_DIR MATCHES "^${CMAKE_SOURCE_DIR}.*")
  foreach(_target opengv pangolin)
    get_target_property(_include_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    set(_include_dirs_new "")
    foreach(_dir IN LISTS _include_dirs)
      if(_dir MATCHES ".*/eigen(/unsupported)?$")
        string(REGEX REPLACE "(^${CMAKE_SOURCE_DIR}.*$)" "$<BUILD_INTERFACE:\\1>" _dir "${_dir}")
      endif()
      list(APPEND _include_dirs_new "${_dir}")
    endforeach()
    set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_include_dirs_new}")
  endforeach()
endif()
