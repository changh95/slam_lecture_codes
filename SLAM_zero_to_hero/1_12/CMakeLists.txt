cmake_minimum_required(VERSION 3.10)
project(MEMORY_PROFILER LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type - Debug for better Valgrind output
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags for better debugging with Valgrind
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Main example executable
add_executable(memory_examples examples/memory_examples.cpp)

# Enable all warnings
target_compile_options(memory_examples PRIVATE -Wall -Wextra)

message(STATUS "")
message(STATUS "=== Memory Profiler Tutorial ===")
message(STATUS "Executable: memory_examples")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./memory_examples              # Run without leak demo")
message(STATUS "  ./memory_examples --with-leak  # Run with intentional leak")
message(STATUS "")
message(STATUS "Valgrind commands:")
message(STATUS "  valgrind --leak-check=full ./memory_examples --with-leak")
message(STATUS "  valgrind --tool=massif ./memory_examples")
message(STATUS "  ms_print massif.out.<pid>")
message(STATUS "")
