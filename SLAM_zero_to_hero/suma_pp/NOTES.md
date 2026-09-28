# SuMa++ notes

How this image differs from upstream, and why.

## Why not the upstream Dockerfile

Upstream `semantic_suma` (531954d) targets Ubuntu 18.04, CUDA 10, TensorRT 5 and catkin.
None of that runs on an RTX 50xx (Blackwell, sm_120):

- sm_120 needs CUDA >= 12.8 and **TensorRT >= 10.8**. Older TensorRT has no kernels for it.
- `rangenet_lib` (3fc223e) uses the TensorRT 5 API. All of it was removed by TensorRT 10:
  `createNetwork`, `setMaxBatchSize`, `setFp16Mode`, `buildCudaEngine`, `IPluginFactory`,
  `getBindingDimensions`, `enqueue`, `destroy()`.

So the image starts from `nvcr.io/nvidia/tensorrt:25.04-py3`, which ships Ubuntu 24.04,
CUDA 12.9 and TensorRT 10.9.0. On top of that it adds:

| Piece | Version | Note |
|---|---|---|
| semantic_suma | 531954d | patched by `patches/apply_patches.py` |
| rangenet_lib | 3fc223e | `netTensorRT.{hpp,cpp}` replaced by a TensorRT 10 port |
| glow | e66d7f8 | the commit upstream's own Dockerfile pins |
| GTSAM | 4.2.0 | built from source: the 4.0 PPA has no 24.04 package |
| RangeNet++ | DarkNet53, SemanticKITTI | `darknet53.tar.gz` from ipb.uni-bonn.de, in `/opt/darknet53` |

## TensorRT 10 port of rangenet_lib

`patches/rangenet_lib/netTensorRT.cpp` makes these replacements:

- `createNetworkV2(0)`, since explicit batch is the only mode in TRT 10.
- `IBuilderConfig`, `setMemoryPoolLimit(kWORKSPACE, 4 GiB)` and `buildSerializedNetwork`
  replace `buildCudaEngine`.
- I/O uses named tensors: `getIOTensorName`, `getTensorShape`, `getTensorIOMode`.
  Buffers are bound once with `setTensorAddress`, and inference calls `enqueueV3`.
- `delete` replaces `destroy()`.
- `prepareBuffer` sizes its vectors with `assign()`. Upstream called `reserve()` and then
  indexed the vectors, which is undefined behaviour.
- The build is **FP32** by default, as in the upstream Dockerfile. Set `RANGENET_FP16=1`
  to try FP16.
- The engine cache (`model.trt`) goes to `$RANGENET_ENGINE_DIR`, which is `/engine_cache`
  in the image, instead of the read-only model directory. An engine only works on the GPU
  and TensorRT version that built it, so mount a host directory there to keep it between runs.

## Build without catkin

`patches/CMakeLists.ws.txt` becomes `/ws/CMakeLists.txt`. It builds glow, rangenet_lib
and semantic_suma with plain CMake, and fills in the `catkin_*` variables that
semantic_suma's CMakeLists expects.

**`BUILD_SHARED_LIBS ON` is required.** catkin builds shared libraries, and SuMa++ quietly
relies on that. glow registers shaders in its cache from static initializers in the
generated `computation_shaders.cpp`. Nothing references that object by symbol, so in a
static `libsuma.a` the linker drops it. The visualizer then aborts at startup:

```
terminate called after throwing an instance of 'glow::GlShaderError'
  what():  shader/laserscan.vert: Cache does not contain entry with name 'shader/color.glsl'
```

The patch script also does two more things:
- It drops `find_package(catkin)` and `catkin_package`, and switches to C++17 because the
  TensorRT 10 headers need it.
- It changes `CMAKE_RUNTIME_OUTPUT_DIRECTORY` from `${CMAKE_SOURCE_DIR}/bin` to
  `${PROJECT_SOURCE_DIR}/bin`. Without that, `visualizer` ends up in `/ws/bin` instead of
  `/ws/semantic_suma/bin`.

On Ubuntu 24.04 only `libopencv-dev` ships `OpenCVConfig.cmake`; `libopencv-core-dev` does not.

## Autorun mode

SuMa++ has only one front end, the Qt `visualizer`: open a `.bin`, then press play. So that it
can run headless, `apply_patches.py` adds an autorun mode to `VisualizerWindow`, driven by
environment variables:

| Variable | Effect |
|---|---|
| `SUMA_AUTOPLAY=1` | start playing right after the scan given on the command line is opened (turns on "fast mode") |
| `SUMA_OUTPUT_DIR` | where `poses.txt`, `runtime.txt` and the screenshots go |
| `SUMA_MAX_SCANS=N` | stop after N scans (default: whole sequence) |
| `SUMA_SNAPSHOT_EVERY=K` | also save the 3D view to `frames/<scan>.png` every K scans |
| `SUMA_EXIT=1` | quit when done (`SUMA_EXIT_DELAY_MS` waits first) |

At the end it writes:
- `poses.txt`: the optimized poses in the KITTI left-camera frame, converted with `Tr`
  from `calib.txt`. This is the same conversion the GUI's "save poses" uses.
- `runtime.txt`: per-scan timings, in seconds (SuMa's `Stopwatch` measures seconds).
- `suma_follow.png`: the 3D view as it was while running.
- `suma_birdseye.png`: the 3D view with "Birds Eye View" ticked.
- `suma_window.png`: the whole window, 3D view plus Qt panels.

The 3D views come from `QGLWidget::grabFrameBuffer()`. The window image comes from
`QScreen::grabWindow(winId())`, which reads back only the visualizer's own window, even on
a real desktop.

Without the environment variables the visualizer behaves exactly as upstream.

## OpenGL

SuMa++ does its ICP, map rendering and surfel updates in OpenGL 4.x shaders, so it needs a
working OpenGL 4 driver. Verified: the NVIDIA driver (RTX 5090, OpenGL 4.6.0, 580.126.18) on the host
X server, reached with `-e DISPLAY=:1 -v /tmp/.X11-unix:/tmp/.X11-unix` plus the `graphics` driver
capability.

Mesa inside Xvfb does not work (tested 2026-09-28, Mesa 25.2.8 / LLVM 20.1.2):
- llvmpipe aborts with `LLVM ERROR: Cannot emit physreg copy instruction`, with or without
  `LP_NATIVE_VECTOR_WIDTH=128`.
- With `GALLIUM_OVERRIDE_CPU_CAPS=avx` it runs, but the results are wrong: the poses stay at
  identity, the semantic image is black, and the map is empty.
- softpipe supports only GLSL 3.30, and SuMa needs 4.00.

`scripts/run_autorun.sh` still falls back to Xvfb when `$DISPLAY` is unset, in case a
later Mesa fixes this, but do not trust its output without looking at the screenshots.
