/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Image pyramid implementation stored as mipmap
*/

#pragma once

#include <basalt/image/image.h>

namespace basalt {

/// @brief Image pyramid that stores levels as mipmap
///
/// \image html mipmap.jpeg
/// Computes image pyramid (see \ref subsample) and stores it as a mipmap
/// (https://en.wikipedia.org/wiki/Mipmap).
template <typename T, class Allocator = DefaultImageAllocator<T>>
class ManagedImagePyr {
 public:
  using PixelType = T;
  using Ptr = std::shared_ptr<ManagedImagePyr<T, Allocator>>;

  /// @brief Default constructor.
  inline ManagedImagePyr() {}

  /// @brief Construct image pyramid from other image.
  ///
  /// @param other image to use for the pyramid level 0
  /// @param num_level number of levels for the pyramid
  inline ManagedImagePyr(const ManagedImage<T>& other, size_t num_levels) { setFromImage(other, num_levels); }

  /// @brief Set image pyramid from other image.
  ///
  /// @param other image to use for the pyramid level 0
  /// @param num_level number of levels for the pyramid
  inline void setFromImage(const ManagedImage<T>& other, size_t num_levels) {
    orig_w = other.w;
    image.Reinitialise(other.w + other.w / 2, other.h);
    image.Fill(0);
    lvl_internal(0).CopyFrom(other);

    for (size_t i = 0; i < num_levels; i++) {
      const Image<const T> l = lvl(i);
      Image<T> lp1 = lvl_internal(i + 1);
      subsample(l, lp1);
    }
  }

  /// @brief Extrapolate image after border with reflection.
  static inline int border101(int x, int h) { return h - 1 - std::abs(h - 1 - x); }

  /// @brief Subsample the image twice in each direction.
  ///
  /// Subsampling is done by convolution with Gaussian kernel
  /// \f[
  /// \frac{1}{256}
  /// \begin{bmatrix}
  ///   1 & 4 & 6 & 4 & 1
  /// \\4 & 16 & 24 & 16 & 4
  /// \\6 & 24 & 36 & 24 & 6
  /// \\4 & 16 & 24 & 16 & 4
  /// \\1 & 4 & 6 & 4 & 1
  /// \\ \end{bmatrix}
  /// \f]
  /// and removing every even-numbered row and column.
  static void subsample(const Image<const T>& img, Image<T>& img_sub) {
    static_assert(std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value);

    constexpr int kernel[5] = {1, 4, 6, 4, 1};

    // accumulator
    ManagedImage<int> tmp(img_sub.h, img.w);

    // Vertical convolution
    {
      for (int r = 0; r < int(img_sub.h); r++) {
        const T* row_m2 = img.RowPtr(std::abs(2 * r - 2));
        const T* row_m1 = img.RowPtr(std::abs(2 * r - 1));
        const T* row = img.RowPtr(2 * r);
        const T* row_p1 = img.RowPtr(border101(2 * r + 1, img.h));
        const T* row_p2 = img.RowPtr(border101(2 * r + 2, img.h));

        for (int c = 0; c < int(img.w); c++) {
          tmp(r, c) = kernel[0] * int(row_m2[c]) + kernel[1] * int(row_m1[c]) + kernel[2] * int(row[c]) +
                      kernel[3] * int(row_p1[c]) + kernel[4] * int(row_p2[c]);
        }
      }
    }

    // Horizontal convolution
    {
      for (int c = 0; c < int(img_sub.w); c++) {
        const int* row_m2 = tmp.RowPtr(std::abs(2 * c - 2));
        const int* row_m1 = tmp.RowPtr(std::abs(2 * c - 1));
        const int* row = tmp.RowPtr(2 * c);
        const int* row_p1 = tmp.RowPtr(border101(2 * c + 1, tmp.h));
        const int* row_p2 = tmp.RowPtr(border101(2 * c + 2, tmp.h));

        for (int r = 0; r < int(tmp.w); r++) {
          int val_int = kernel[0] * row_m2[r] + kernel[1] * row_m1[r] + kernel[2] * row[r] + kernel[3] * row_p1[r] +
                        kernel[4] * row_p2[r];
          T val = ((val_int + (1 << 7)) >> 8);
          img_sub(c, r) = val;
        }
      }
    }
  }

  /// @brief Return const image of the certain level
  ///
  /// @param lvl level to return
  /// @return const image of with the pyramid level
  inline const Image<const T> lvl(size_t lvl) const {
    size_t x = (lvl == 0) ? 0 : orig_w;
    size_t y = (lvl <= 1) ? 0 : (image.h - (image.h >> (lvl - 1)));
    size_t width = (orig_w >> lvl);
    size_t height = (image.h >> lvl);

    return image.SubImage(x, y, width, height);
  }

  /// @brief Return const image of underlying mipmap
  ///
  /// @return const image of of the underlying mipmap representation which can
  /// be for example used for visualization
  inline const Image<const T> mipmap() const { return image.SubImage(0, 0, image.w, image.h); }

  /// @brief Return coordinate offset of the image in the mipmap image.
  ///
  /// @param lvl level to return
  /// @return offset coordinates (2x1 vector)
  template <typename S>
  inline Eigen::Matrix<S, 2, 1> lvl_offset(size_t lvl) {
    size_t x = (lvl == 0) ? 0 : orig_w;
    size_t y = (lvl <= 1) ? 0 : (image.h - (image.h >> (lvl - 1)));

    return Eigen::Matrix<S, 2, 1>(x, y);
  }

 protected:
  /// @brief Return image of the certain level
  ///
  /// @param lvl level to return
  /// @return image of with the pyramid level
  inline Image<T> lvl_internal(size_t lvl) {
    size_t x = (lvl == 0) ? 0 : orig_w;
    size_t y = (lvl <= 1) ? 0 : (image.h - (image.h >> (lvl - 1)));
    size_t width = (orig_w >> lvl);
    size_t height = (image.h >> lvl);

    return image.SubImage(x, y, width, height);
  }

  size_t orig_w;          ///< Width of the original image (level 0)
  ManagedImage<T> image;  ///< Pyramid image stored as a mipmap
};

}  // namespace basalt
