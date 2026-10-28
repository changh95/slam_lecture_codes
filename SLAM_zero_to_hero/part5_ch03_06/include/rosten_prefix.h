/**
 * @file rosten_prefix.h
 * @brief Give each compiled copy of Rosten's FAST-9 C code its own symbol names
 *
 * CMake builds fast_9.c and nonmax.c twice, with -O3 and with -O3 -march=native.
 * Both copies end up in one executable, so every global function is renamed to
 * ROSTEN_PREFIX##name (e.g. plain_o3_fast9_detect). CMake force-includes this
 * header into those C files and defines ROSTEN_PREFIX per copy.
 */

#ifndef ROSTEN_PREFIX_H
#define ROSTEN_PREFIX_H

#define ROSTEN_CAT2(a, b) a##b
#define ROSTEN_CAT(a, b) ROSTEN_CAT2(a, b)

#define fast9_corner_score ROSTEN_CAT(ROSTEN_PREFIX, fast9_corner_score)
#define fast9_detect ROSTEN_CAT(ROSTEN_PREFIX, fast9_detect)
#define fast9_score ROSTEN_CAT(ROSTEN_PREFIX, fast9_score)
#define nonmax_suppression ROSTEN_CAT(ROSTEN_PREFIX, nonmax_suppression)

#endif
