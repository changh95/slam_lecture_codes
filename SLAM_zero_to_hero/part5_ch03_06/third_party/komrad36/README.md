# komrad36 SIMD kernels (vendored)

Header-only CPU kernels by Kareem Omar ([komrad36](https://github.com/komrad36)), copied from upstream with their licenses.

| Folder | Upstream commit | License | Used by |
|---|---|---|---|
| `KFAST/KFAST.h` | [komrad36/KFAST@8f90e44](https://github.com/komrad36/KFAST/tree/8f90e441833510a90c6e4486131d670d79fd06d5) | MIT | `kfast_benchmark` |
| `KFAST/Rosten/` | shipped in the same KFAST commit (Edward Rosten's FAST C code) | BSD-3-Clause | plain C baseline: `fast_9.c` and `nonmax.c` are built into `plain_o3` / `plain_native` |
| `RGB2Y/RGB2Y.h` | [komrad36/RGB2Y@953aa22](https://github.com/komrad36/RGB2Y/tree/953aa22547dd363b6674f9637455866eeaa67c5d) | MIT | `rgb2y_benchmark` |
| `FeatureAngle/FeatureAngle.h` | [komrad36/FeatureAngle@282653c](https://github.com/komrad36/FeatureAngle/tree/282653cedf045abba737bdc2bf3ba989225ed613) | MIT | `feature_angle_benchmark` |

Upstream demo drivers (`main.cpp`), Makefiles, and test images are not included.

## Local changes

Only one change from upstream. The other headers are byte-identical.

- **`RGB2Y/RGB2Y.h`, multi-threaded split**: upstream always launched `hw_concur - 1` chunks of
  `thread_stride` rows. For some image heights the chunks overrun the image. A KITTI frame
  (370 rows) on a 32-thread CPU gets 22 chunks of 17 rows (374 rows), so the last thread starts at
  row 374 and `RGB2Y<true, ...>` segfaults. The patched loop stops once the next chunk would reach
  the last row and gives the remaining rows to the final thread. It is marked
  `[SLAM zero-to-hero patch]` in the source.

GCC needs a few definitions and includes that the headers take from MSVC (`__forceinline`,
`<cstring>`, `<thread>`, `<vector>`). These come from `include/komrad36_compat.hpp`, so the headers
themselves need no edits for GCC.
