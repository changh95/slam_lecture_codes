cmake_minimum_required(VERSION 3.10)
project(BUILDING_CPP_LIBRARIES LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build type: Release (default)")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# ============================================================
# 1. STATIC LIBRARY
# ============================================================
# Create a static library from source files
# Static libraries are linked at compile time
# Extension: .a (Linux/Mac), .lib (Windows)
add_library(mathlib_static STATIC
    src/mathlib.cpp
)

# Specify include directories for this library
# PUBLIC: Both the library and users of the library need these headers
target_include_directories(mathlib_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# 2. SHARED LIBRARY
# ============================================================
# Create a shared library from the same source
# Shared libraries are loaded at runtime
# Extension: .so (Linux), .dylib (Mac), .dll (Windows)
add_library(mathlib_shared SHARED
    src/mathlib.cpp
)

target_include_directories(mathlib_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set library version (optional but recommended)
set_target_properties(mathlib_shared PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)

# ============================================================
# 3. HEADER-ONLY LIBRARY (INTERFACE)
# ============================================================
# Header-only libraries have no source files
# Use INTERFACE keyword - no compilation, just headers
add_library(geometry INTERFACE)

target_include_directories(geometry INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# EXECUTABLES
# ============================================================
# Example using static library
add_executable(use_static_lib examples/use_static_lib.cpp)
target_link_libraries(use_static_lib PRIVATE mathlib_static)

# Example using shared library
add_executable(use_shared_lib examples/use_shared_lib.cpp)
target_link_libraries(use_shared_lib PRIVATE mathlib_shared)

# Example using header-only library
add_executable(use_header_only examples/use_header_only.cpp)
target_link_libraries(use_header_only PRIVATE geometry)

# ============================================================
# INSTALLATION (optional)
# ============================================================
# This is how you would install your library for system-wide use
# install(TARGETS mathlib_static mathlib_shared
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
# )
# install(DIRECTORY include/ DESTINATION include)

# ============================================================
# SUMMARY
# ============================================================
message(STATUS "")
message(STATUS "=== Library Summary ===")
message(STATUS "Static library:      mathlib_static")
message(STATUS "Shared library:      mathlib_shared")
message(STATUS "Header-only library: geometry")
message(STATUS "")
message(STATUS "=== Executables ===")
message(STATUS "use_static_lib  - demonstrates static linking")
message(STATUS "use_shared_lib  - demonstrates shared linking")
message(STATUS "use_header_only - demonstrates header-only library")
message(STATUS "")
