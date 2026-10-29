# komrad36 SIMD kernels (vendored)

CPU code by Kareem Omar ([komrad36](https://github.com/komrad36)): full copies of four upstream repositories, with
their licenses. The Dockerfile copies this folder into the image with the rest of the exercise, so nothing is
downloaded at build time.

| Folder | Upstream commit | License | Used by |
|---|---|---|---|
| `KFAST/` | [komrad36/KFAST@8f90e44](https://github.com/komrad36/KFAST/tree/8f90e441833510a90c6e4486131d670d79fd06d5) | MIT | `KFAST.h`: `kfast_benchmark` |
| `KFAST/Rosten/` | shipped in the same KFAST commit (Edward Rosten's FAST C code) | BSD-3-Clause | plain C baseline: `fast_9.c` and `nonmax.c` are built into `plain_o3` / `plain_native` |
| `RGB2Y/` | [komrad36/RGB2Y@953aa22](https://github.com/komrad36/RGB2Y/tree/953aa22547dd363b6674f9637455866eeaa67c5d) | MIT | `RGB2Y.h`: `rgb2y_benchmark` |
| `FeatureAngle/` | [komrad36/FeatureAngle@282653c](https://github.com/komrad36/FeatureAngle/tree/282653cedf045abba737bdc2bf3ba989225ed613) | MIT | `FeatureAngle.h`: `feature_angle_benchmark` |
| `FastThreadPool/` | [komrad36/FastThreadPool@55670cf](https://github.com/komrad36/FastThreadPool/tree/55670cfcc6f11b8994eb5f058b03ab04084d67b1) | MIT | `fast_thread_pool.h`: `include/fast_parallel_for.hpp` (every komrad36 `_MT` variant) |

Every file the upstream commits track is here except the `.gitignore` files. Those ignore everything (`*`) and
whitelist upstream's own files, so inside this repository they would silently hide any file added next to them.

`CMakeLists.txt` only uses the headers (and Rosten's two C files). Upstream's demo drivers (`RGB2Y/main.cpp`,
`KFAST/main.cpp`), Makefiles, and `test.jpg` / `results.jpg` are kept for reference only. They do not build in
`slam:base` as published: `RGB2Y/main.cpp` hits MSVC's `__forceinline` in `RGB2Y.h`, and `KFAST/main.cpp` uses
OpenCV 2/3 constants (`CV_LOAD_IMAGE_GRAYSCALE`, `CV_WINDOW_NORMAL`) that OpenCV 4 removed.

GCC needs a few definitions and includes that the kernel headers take from MSVC (`__forceinline`, `<cstring>`,
`<thread>`, `<vector>`). These come from `include/komrad36_compat.hpp`, so the headers' code needs no edits.

## Local changes

- **Design comments in `RGB2Y/RGB2Y.h`, `KFAST/KFAST.h`, and `FeatureAngle/FeatureAngle.h`**: added for the course.
  They explain why each kernel is built the way it is (lane widths, fixed-point math, shuffle layout, branch-free
  tests, memory over-reads and over-writes). Every added comment starts with `[SLAM zero-to-hero]`, and only comment
  lines were added: no line of upstream code was changed or removed, so `diff` against the upstream commit shows
  additions only. Each header opens with a "Design notes" block, and notes inside the code mark the decisions as
  they occur. `RGB2Y.h` and `KFAST.h` keep upstream's CRLF line endings.
- **`FastThreadPool/fast_thread_pool.h`, `RemoveWork()`**: the store that hands the queue head to the next consumer
  (`S9`) is `memory_order_release` instead of `memory_order_relaxed`. Without it, under the C++ memory model, a worker
  that removes the next item has no happens-before edge to the `AddWork()` that wrote it, and ThreadSanitizer reports
  data races on the item (read of `m_f`, `delete`). With the change, `feature_angle_benchmark` runs clean under
  ThreadSanitizer. On x86 both orderings compile to the same plain store, so the timings are unaffected. On ARM the
  release store is needed. The line is marked `[SLAM zero-to-hero patch]`.

## Multi-threading

RGB2Y and KFAST have a built-in multi-threaded mode (`RGB2Y<true, ...>`, `KFAST<true, ...>`). This exercise does not
use it. It starts new threads with `std::async` on every call, one per hardware thread. On a 32-thread Ryzen 9 7950X,
starting and joining 32 empty `std::async` tasks took 0.65 ms, more than RGB2Y's whole single-threaded run time on
a 1080p frame (about 0.2 ms). The `_MT` variants call the headers' row-range workers
(`_RGB2Y`, `_KFAST`) instead, from persistent FastThreadPool workers.

RGB2Y's row splitting has two more problems:

- **Rows overwrite each other's first pixels.** RGB2Y stores 16 bytes for every 10 output pixels, so when
  `cols % 10` is 1 to 5, each row's last full store writes up to 5 zero bytes into the start of the next row. The
  thread that converts the next row overwrites them, which is fine within one thread. At a boundary between two
  threads, the result depends on which thread writes last. With upstream's `std::async` split, a 1241-pixel-wide
  image came out right in our runs, because each thread starts well after the previous one. With FastThreadPool
  stripes, it came out wrong on every run until `rgb2y_benchmark` started converting each stripe's last row into
  a scratch row and copying it back.
- **Upstream's chunks can run past the image.** `RGB2Y<true, ...>` always launches `hw_concur - 1` full chunks,
  so for some heights the chunks overrun. A 370-row image on 32 threads gets 22 chunks of 17 rows (374 rows), the
  last thread starts at row 374, and the call segfaults.
