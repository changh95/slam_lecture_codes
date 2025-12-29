cmake_minimum_required(VERSION 3.10)
project(symforce_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find SymForce
find_package(symforce QUIET)

if(symforce_FOUND)
    message(STATUS "SymForce found")

    # Example using generated code
    add_executable(symforce_example examples/symforce_example.cpp)
    target_link_libraries(symforce_example symforce::symforce_opt)
else()
    message(STATUS "SymForce not found - Python examples only")
    message(STATUS "Install SymForce: pip install symforce")
endif()

message(STATUS "")
message(STATUS "=== SymForce Tutorial ===")
message(STATUS "")
message(STATUS "Python examples:")
message(STATUS "  python3 examples/symforce_basics.py")
message(STATUS "")
message(STATUS "Install SymForce:")
message(STATUS "  pip install symforce")
message(STATUS "")
