/**
 * @file komrad36_compat.hpp
 * @brief Include before any header in third_party/komrad36 when building with GCC
 *
 * komrad36's headers are kept exactly as published. They were written against
 * MSVC, which defines __forceinline and pulls <cstring>, <thread> and <vector>
 * in through other headers; GCC 13 needs all four spelled out.
 */

#pragma once

#include <cstring>
#include <thread>
#include <vector>

#if !defined(_MSC_VER) && !defined(__forceinline)
#define __forceinline inline __attribute__((always_inline))
#endif
