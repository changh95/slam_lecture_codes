#pragma once

#include <cstddef>

namespace basalt {

// to work around static_assert(false, ...)
template <class T>
struct dependent_false : std::false_type {};

template <class T>
inline void hash_combine(std::size_t& seed, const T& value) {
  // Simple hash_combine, see e.g. here:
  // https://github.com/HowardHinnant/hash_append/issues/7
  // Not sure we ever need 32bit, but here it is...
  if constexpr (sizeof(std::size_t) == 4) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
  } else if constexpr (sizeof(std::size_t) == 8) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15LLU + (seed << 12) + (seed >> 4);
  } else {
    static_assert(dependent_false<T>::value, "hash_combine not implemented");
  }
}

}  // namespace basalt
