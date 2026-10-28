# komrad36 SIMD kernels (vendored)

Header-only CPU code by Kareem Omar ([komrad36](https://github.com/komrad36)), copied unmodified from upstream with
the licenses.

| Folder | Upstream commit | License | Used by |
|---|---|---|---|
| `KFAST/KFAST.h` | [komrad36/KFAST@8f90e44](https://github.com/komrad36/KFAST/tree/8f90e441833510a90c6e4486131d670d79fd06d5) | MIT | `kfast_benchmark` |
| `KFAST/Rosten/` | shipped in the same KFAST commit (Edward Rosten's FAST C code) | BSD-3-Clause | plain C baseline: `fast_9.c` and `nonmax.c` are built into `plain_o3` / `plain_native` |
| `RGB2Y/RGB2Y.h` | [komrad36/RGB2Y@953aa22](https://github.com/komrad36/RGB2Y/tree/953aa22547dd363b6674f9637455866eeaa67c5d) | MIT | `rgb2y_benchmark` |
| `FeatureAngle/FeatureAngle.h` | [komrad36/FeatureAngle@282653c](https://github.com/komrad36/FeatureAngle/tree/282653cedf045abba737bdc2bf3ba989225ed613) | MIT | `feature_angle_benchmark` |
| `FastThreadPool/fast_thread_pool.h` | [komrad36/FastThreadPool@55670cf](https://github.com/komrad36/FastThreadPool/tree/55670cfcc6f11b8994eb5f058b03ab04084d67b1) | MIT | `include/fast_parallel_for.hpp` (every `_MT` variant) |

Upstream demo drivers (`main.cpp`), Makefiles, and test images are not included.

GCC needs a few definitions and includes that the kernel headers take from MSVC (`__forceinline`, `<cstring>`,
`<thread>`, `<vector>`). These come from `include/komrad36_compat.hpp`, so the headers need no edits.

## Multi-threading

RGB2Y and KFAST have a built-in multi-threaded mode (`RGB2Y<true, ...>`, `KFAST<true, ...>`). This exercise does not
use it. It starts new threads with `std::async` on every call, one per hardware thread. On a 32-thread Ryzen 9 7950X,
starting and joining 23–32 empty `std::async` tasks took 0.45–0.65 ms. That is more than RGB2Y's whole
single-threaded run time on a 1080p frame (about 0.2 ms). The `_MT` variants call the headers' row-range workers
(`_RGB2Y`, `_KFAST`) instead, from persistent FastThreadPool workers.

The built-in split in `RGB2Y<true, ...>` also has a bug. It always launches `hw_concur - 1` full chunks, so for some
heights the chunks run past the image. A 370-row image on 32 threads gets 22 chunks of 17 rows (374 rows), the last
thread starts at row 374, and the call segfaults.
