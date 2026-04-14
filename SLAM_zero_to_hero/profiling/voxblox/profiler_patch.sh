#!/bin/bash
# Inject easy_profiler EASY_BLOCK instrumentation into voxblox source files.
# Uses a Python helper because voxblox function signatures span multiple lines,
# which breaks naive line-based sed/awk patching.
set -e

VOXBLOX_SRC="/catkin_ws/src/voxblox"
echo "[profiler_patch] Patching voxblox for easy_profiler..."

inject_block() {
  local file="$1"
  local funcname="$2"
  local block_name="$3"
  python3 - "$file" "$funcname" "$block_name" << 'PY'
import re, sys
path, funcname, block_name = sys.argv[1:4]
with open(path) as f:
    src = f.read()
pat = re.compile(
    re.escape(funcname) + r"\s*\([^{]*?\)\s*\{",
    re.DOTALL,
)
injection = f'\n#ifdef BUILD_WITH_EASY_PROFILER\n  EASY_BLOCK("{block_name}");\n#endif'
def repl(m):
    return m.group(0) + injection
new_src, n = pat.subn(repl, src, count=1)
if n == 0:
    print(f"[warn] no match for {funcname} in {path}", file=sys.stderr)
else:
    with open(path, "w") as f:
        f.write(new_src)
    print(f"[ok] injected {block_name} into {funcname}")
PY
}

add_profiler_header() {
  local file="$1"
  if ! grep -q "easy/profiler.h" "$file"; then
    python3 - "$file" << 'PY'
import sys
path = sys.argv[1]
header = '''#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#else
#define EASY_BLOCK(...)
#define EASY_END_BLOCK
#define EASY_FUNCTION(...)
#define EASY_THREAD(...)
#define EASY_PROFILER_ENABLE
#endif

'''
with open(path) as f:
    src = f.read()
with open(path, "w") as f:
    f.write(header + src)
PY
  fi
}

# Inject EASY_THREAD("name") at the top of a worker function so that
# blocks logged from inside it are actually captured (easy_profiler ignores
# blocks from threads it has never seen).
inject_thread_register() {
  local file="$1"
  local funcname="$2"
  local thread_name="$3"
  python3 - "$file" "$funcname" "$thread_name" << 'PY'
import re, sys
path, funcname, thread_name = sys.argv[1:4]
with open(path) as f:
    src = f.read()
pat = re.compile(re.escape(funcname) + r"\s*\([^{]*?\)\s*\{", re.DOTALL)
injection = (
    f'\n#ifdef BUILD_WITH_EASY_PROFILER\n'
    f'  EASY_THREAD("{thread_name}");\n'
    f'#endif'
)
new_src, n = pat.subn(lambda m: m.group(0) + injection, src, count=1)
if n == 0:
    print(f"[warn] no match for {funcname} in {path}", file=sys.stderr)
else:
    with open(path, "w") as f:
        f.write(new_src)
    print(f"[ok] EASY_THREAD({thread_name}) -> {funcname}")
PY
}

# ---- TSDF integrator (inner voxblox library) ----
# Top-level integratePointCloud captures the full integration window
# (thread spawn + worker join + post-process). The worker thread bodies
# (integrateFunction) need EASY_THREAD so their blocks aren't dropped.
TSDF_SRC="${VOXBLOX_SRC}/voxblox/src/integrator/tsdf_integrator.cc"
add_profiler_header "$TSDF_SRC"
inject_block          "$TSDF_SRC" "SimpleTsdfIntegrator::integratePointCloud" "SLAM/TsdfIntegration"
inject_block          "$TSDF_SRC" "MergedTsdfIntegrator::integratePointCloud" "SLAM/TsdfIntegration"
inject_block          "$TSDF_SRC" "FastTsdfIntegrator::integratePointCloud"   "SLAM/TsdfIntegration"
inject_thread_register "$TSDF_SRC" "SimpleTsdfIntegrator::integrateFunction"  "tsdf_worker"
inject_thread_register "$TSDF_SRC" "FastTsdfIntegrator::integrateFunction"    "tsdf_worker"
inject_block          "$TSDF_SRC" "SimpleTsdfIntegrator::integrateFunction"   "SLAM/TsdfIntegration/Worker"
inject_block          "$TSDF_SRC" "FastTsdfIntegrator::integrateFunction"     "SLAM/TsdfIntegration/Worker"

# ---- ESDF integrator ----
ESDF_SRC="${VOXBLOX_SRC}/voxblox/src/integrator/esdf_integrator.cc"
if [ -f "$ESDF_SRC" ]; then
  add_profiler_header "$ESDF_SRC"
  inject_block "$ESDF_SRC" "EsdfIntegrator::updateFromTsdfLayer"      "SLAM/EsdfIntegration"
  inject_block "$ESDF_SRC" "EsdfIntegrator::updateFromTsdfLayerBatch" "SLAM/EsdfIntegration/Batch"
fi

# ---- Mesh integrator (header-only template) ----
# The mesh code lives in mesh_integrator.h because MeshIntegrator is a
# class template. Injection into the header instruments every TU that
# instantiates it, but the inline-define guard keeps the build clean.
MESH_HDR="${VOXBLOX_SRC}/voxblox/include/voxblox/mesh/mesh_integrator.h"
if [ -f "$MESH_HDR" ]; then
  add_profiler_header "$MESH_HDR"
  inject_block          "$MESH_HDR" "generateMesh"               "SLAM/MeshGeneration"
  inject_thread_register "$MESH_HDR" "generateMeshBlocksFunction" "mesh_worker"
  inject_block          "$MESH_HDR" "generateMeshBlocksFunction" "SLAM/MeshGeneration/Worker"
fi

# ---- voxblox_ros main server ----
SERVER_SRC="${VOXBLOX_SRC}/voxblox_ros/src/tsdf_server.cc"
if [ -f "$SERVER_SRC" ]; then
  add_profiler_header "$SERVER_SRC"
  inject_block "$SERVER_SRC" "TsdfServer::insertPointcloud" "SLAM/FrameProcess"
  inject_block "$SERVER_SRC" "TsdfServer::generateMesh"     "SLAM/MeshPublish"
fi

# ---- profiler enable + signal handler in main ----
for NODE in \
  "${VOXBLOX_SRC}/voxblox_ros/src/tsdf_server_node.cc" \
  "${VOXBLOX_SRC}/voxblox_ros/src/esdf_server_node.cc"; do
  if [ -f "$NODE" ]; then
    python3 - "$NODE" << 'PY'
import sys, re
path = sys.argv[1]
with open(path) as f:
    src = f.read()
header = '''#ifdef BUILD_WITH_EASY_PROFILER
#include <easy/profiler.h>
#include <cstdlib>
#include <cstdio>
static void voxblox_profiler_dump() {
    profiler::dumpBlocksToFile("/output/voxblox_profiler.prof");
    fprintf(stderr, "Profiler data saved to /output/voxblox_profiler.prof\\n");
    fflush(stderr);
}
#endif

'''
src = header + src
# Use atexit() instead of signal() because ros::init installs its own SIGINT
# handler that may override ours, and we want the dump to happen on any exit
# path (signal, normal return, ros::shutdown).
src = re.sub(
    r"(ros::init\([^;]*\);)",
    r"""\1
#ifdef BUILD_WITH_EASY_PROFILER
    EASY_PROFILER_ENABLE;
    atexit(voxblox_profiler_dump);
#endif""",
    src,
    count=1,
)
with open(path, "w") as f:
    f.write(src)
print(f"[ok] profiler init injected into {path}")
PY
  fi
done

echo "[profiler_patch] Done."
