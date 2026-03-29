/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2024, Collabora Ltd.
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
@brief Types and functionality related to debugging different matrices of the system in the UI
@author Mateo de Mayo <mateo.demayo@collabora.com>
*/
#pragma once

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <basalt/image/image.h>
#include <basalt/utils/imu_types.h>
#include <memory>
#include <vector>

namespace basalt::vis {

struct UILandmarkBlock {
  using MatrixXfr = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using LandmarkId = size_t;
  std::shared_ptr<MatrixXfr> storage;
  LandmarkId lmid = -1;
};

struct UILandmarkBlocks {
  using Ptr = std::shared_ptr<UILandmarkBlocks>;
  std::vector<UILandmarkBlock> blocks;
  AbsOrderMap aom;

  size_t getW() const { return blocks.empty() ? 0 : blocks[0].storage->cols(); }
  size_t getH() const {
    size_t h = 0;
    for (const UILandmarkBlock& lmb : blocks) h += lmb.storage->rows();
    return h;
  }
};

enum class UIMAT {
  JR,       //!< Jacobian J = [Jp Jl] and residual r (landmark blocks)
  JR_QR,    //!< Landmark blocks after QR factorization
  JR_M,     //!< Marginalized Jr
  JR_M_QR,  //!< Marginalized Jr_QR
  HB,       //!< Hessian H = J^T J and b = J^T r
  HB_M,     //!< Marginalized Hb
  COUNT,
  NONE,  //!< Special value to symbolize no UIMATs
  ALL,   //!< Special value to symbolize all UIMATs
};
constexpr int UIMAT_COUNT_J = (int)UIMAT::HB;
constexpr int UIMAT_COUNT_H = (int)UIMAT::COUNT - (int)UIMAT::HB;

struct UIJacobians {
  UILandmarkBlocks::Ptr Jr;                    //!< Landmark blocks
  UILandmarkBlocks::Ptr Jr_h;                  //!< Highlighted
  std::shared_ptr<ManagedImage<uint8_t>> img;  //!< Current rendered image
};

struct UIHessians {
  std::shared_ptr<Eigen::MatrixXf> H;
  std::shared_ptr<Eigen::VectorXf> b;
  std::shared_ptr<AbsOrderMap> aom;
  std::shared_ptr<ManagedImage<uint8_t>> img;
};

inline bool is_jacobian(UIMAT u) { return UIMAT::JR <= u && u < UIMAT::HB; }
inline bool is_hessian(UIMAT u) { return UIMAT::HB <= u && u < UIMAT::COUNT; }

}  // namespace basalt::vis
