# SymForce examples in this chapter are pure Python — there is nothing to
# compile. SymForce *can* generate C++ from the symbolic residuals
# (Codegen.function(...).generate_function()); this CMakeLists is kept only so
# the chapter layout matches the other part3 tutorials.
cmake_minimum_required(VERSION 3.10)
project(symforce_tutorial NONE)

message(STATUS "")
message(STATUS "=== SymForce Tutorial (Python only) ===")
message(STATUS "Run from the examples/ directory:")
message(STATUS "  python3 examples/symforce_curve_fitting.py")
message(STATUS "  python3 examples/symforce_pose_graph.py")
message(STATUS "  python3 examples/symforce_bundle_adjustment.py problem-21-11315-pre.txt")
message(STATUS "")
