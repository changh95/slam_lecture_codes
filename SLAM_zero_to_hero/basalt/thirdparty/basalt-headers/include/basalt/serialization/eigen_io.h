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
@brief Serialization for Eigen and Sophus types
*/

#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {

// NOTE: Serialization functions for non-basalt types (for now Eigen and Sophus)
// are provided in a separate header to make them available to other libraries
// depending on basalt-headers. Beware that it is not possible to have different
// and incompatible definitions for these same types in other libraries or
// executables that also include basalt-headers, as this would lead to undefined
// behaviour. See
// https://groups.google.com/d/topic/cerealcpp/WswQi_Sh-bw/discussion for a more
// detailed discussion and possible workarounds.

// For binary-archives, don't save a size tag for compact representation.
template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
std::enable_if_t<(_Rows > 0) && (_Cols > 0) && !traits::is_text_archive<Archive>::value> serialize(
    Archive& archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  for (int i = 0; i < _Rows; i++) {
    for (int j = 0; j < _Cols; j++) {
      archive(m(i, j));
    }
  }
  // Note: if we can break binary compatibility, we might want to consider the
  // following. However, this would only work for compact Scalar types that can
  // be serialized / deserialized by memcopy, such as double.
  // archive(binary_data(m.data(), _Rows * _Cols * sizeof(_Scalar)));
}

// For text-archives, save size-tag even for constant size matrices, to ensure
// that the serialization is more compact (e.g. for JSON with size tag is uses a
// simple array, whereas without, it stores a list of pairs like ("value0", v0).
template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
std::enable_if_t<(_Rows > 0) && (_Cols > 0) && traits::is_text_archive<Archive>::value> serialize(
    Archive& archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  size_type s = static_cast<size_type>(_Rows * _Cols);
  archive(make_size_tag(s));
  if (s != _Rows * _Cols) {
    throw std::runtime_error("matrix has incorrect length");
  }
  for (size_t i = 0; i < _Rows; i++) {
    for (size_t j = 0; j < _Cols; j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive, class _Scalar, int _Cols, int _Options, int _MaxRows, int _MaxCols>
std::enable_if_t<(_Cols > 0), void> save(
    Archive& archive, const Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  archive(make_size_tag(static_cast<size_type>(m.size())));
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < _Cols; j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive, class _Scalar, int _Cols, int _Options, int _MaxRows, int _MaxCols>
std::enable_if_t<(_Cols > 0), void> load(
    Archive& archive, Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  size_type size;
  archive(make_size_tag(size));
  m.resize(Eigen::Index(size) / _Cols, _Cols);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < _Cols; j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive, class _Scalar, int _Rows, int _Options, int _MaxRows, int _MaxCols>
std::enable_if_t<(_Rows > 0), void> save(
    Archive& archive, const Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& m) {
  archive(make_size_tag(static_cast<size_type>(m.size())));
  for (int i = 0; i < _Rows; i++) {
    for (int j = 0; j < m.cols(); j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive, class _Scalar, int _Rows, int _Options, int _MaxRows, int _MaxCols>
std::enable_if_t<(_Rows > 0), void> load(
    Archive& archive, Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& m) {
  size_type size;
  archive(make_size_tag(size));
  m.resize(_Rows, Eigen::Index(size) / _Rows);
  for (int i = 0; i < _Rows; i++) {
    for (int j = 0; j < m.cols(); j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive, class _Scalar, int _Options, int _MaxRows, int _MaxCols>
void save(Archive& archive,
          const Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& m) {
  archive(make_size_tag(static_cast<size_type>(m.rows())));
  archive(make_size_tag(static_cast<size_type>(m.cols())));
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive, class _Scalar, int _Options, int _MaxRows, int _MaxCols>
void load(Archive& archive, Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& m) {
  size_type rows;
  size_type cols;
  archive(make_size_tag(rows));
  archive(make_size_tag(cols));
  m.resize(rows, cols);
  for (int i = 0; i < m.rows(); i++) {
    for (int j = 0; j < m.cols(); j++) {
      archive(m(i, j));
    }
  }
}

template <class Archive>
void serialize(Archive& ar, Sophus::SE3d& p) {
  ar(cereal::make_nvp("px", p.translation()[0]), cereal::make_nvp("py", p.translation()[1]),
     cereal::make_nvp("pz", p.translation()[2]), cereal::make_nvp("qx", p.so3().data()[0]),
     cereal::make_nvp("qy", p.so3().data()[1]), cereal::make_nvp("qz", p.so3().data()[2]),
     cereal::make_nvp("qw", p.so3().data()[3]));
}

template <class Archive>
void serialize(Archive& ar, Sophus::Sim3d& p) {
  ar(cereal::make_nvp("px", p.translation()[0]), cereal::make_nvp("py", p.translation()[1]),
     cereal::make_nvp("pz", p.translation()[2]), cereal::make_nvp("qx", p.rxso3().data()[0]),
     cereal::make_nvp("qy", p.rxso3().data()[1]), cereal::make_nvp("qz", p.rxso3().data()[2]),
     cereal::make_nvp("qw", p.rxso3().data()[3]));
}

}  // namespace cereal
