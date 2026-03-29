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
@brief Serialization for basalt types
*/

#pragma once

#include <basalt/serialization/eigen_io.h>
#include <basalt/calibration/calibration.hpp>
#include <basalt/camera/bal_camera.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::GenericCamera<Scalar>& cam) {
  std::visit(
      [&](const auto& cam) {
        ar(cereal::make_nvp("camera_type", cam.getName()));
        ar(cereal::make_nvp("intrinsics", cam));
      },
      cam.variant);
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::GenericCamera<Scalar>& cam) {
  std::string cam_type;
  ar(cereal::make_nvp("camera_type", cam_type));

  cam = basalt::GenericCamera<Scalar>::fromString(cam_type);

  std::visit([&](auto& cam) { ar(cereal::make_nvp("intrinsics", cam)); }, cam.variant);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::KannalaBrandtCamera4<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("k1", cam.getParam()[4]), cereal::make_nvp("k2", cam.getParam()[5]),
     cereal::make_nvp("k3", cam.getParam()[6]), cereal::make_nvp("k4", cam.getParam()[7]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::KannalaBrandtCamera4<Scalar>& cam) {
  Eigen::Matrix<Scalar, 8, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("k1", intr[4]), cereal::make_nvp("k2", intr[5]),
     cereal::make_nvp("k3", intr[6]), cereal::make_nvp("k4", intr[7]));

  cam = basalt::KannalaBrandtCamera4<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::ExtendedUnifiedCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("alpha", cam.getParam()[4]), cereal::make_nvp("beta", cam.getParam()[5]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::ExtendedUnifiedCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 6, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("alpha", intr[4]), cereal::make_nvp("beta", intr[5]));

  cam = basalt::ExtendedUnifiedCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::UnifiedCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("alpha", cam.getParam()[4]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::UnifiedCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 5, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("alpha", intr[4]));

  cam = basalt::UnifiedCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::PinholeCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::PinholeCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 4, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]));

  cam = basalt::PinholeCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::PinholeRadtan8Camera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("k1", cam.getParam()[4]), cereal::make_nvp("k2", cam.getParam()[5]),
     cereal::make_nvp("p1", cam.getParam()[6]), cereal::make_nvp("p2", cam.getParam()[7]),
     cereal::make_nvp("k3", cam.getParam()[8]), cereal::make_nvp("k4", cam.getParam()[9]),
     cereal::make_nvp("k5", cam.getParam()[10]), cereal::make_nvp("k6", cam.getParam()[11]),
     cereal::make_nvp("rpmax", cam.getRpmax()));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::PinholeRadtan8Camera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 12, 1> intr;
  Scalar rpmax;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("k1", intr[4]), cereal::make_nvp("k2", intr[5]),
     cereal::make_nvp("p1", intr[6]), cereal::make_nvp("p2", intr[7]), cereal::make_nvp("k3", intr[8]),
     cereal::make_nvp("k4", intr[9]), cereal::make_nvp("k5", intr[10]), cereal::make_nvp("k6", intr[11]),
     cereal::make_nvp("rpmax", rpmax));

  cam = basalt::PinholeRadtan8Camera<Scalar>(intr, rpmax);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::DoubleSphereCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("xi", cam.getParam()[4]), cereal::make_nvp("alpha", cam.getParam()[5]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::DoubleSphereCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 6, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("xi", intr[4]), cereal::make_nvp("alpha", intr[5]));

  cam = basalt::DoubleSphereCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::FovCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]), cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]), cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("w", cam.getParam()[4]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::FovCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 5, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("w", intr[4]));

  cam = basalt::FovCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const basalt::BalCamera<Scalar>& cam) {
  ar(cereal::make_nvp("f", cam.getParam()[0]), cereal::make_nvp("k1", cam.getParam()[1]),
     cereal::make_nvp("k2", cam.getParam()[2]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, basalt::BalCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 3, 1> intr;

  ar(cereal::make_nvp("f", intr[0]), cereal::make_nvp("k1", intr[1]), cereal::make_nvp("k2", intr[2]));

  cam = basalt::BalCamera<Scalar>(intr);
}

template <class Archive, class Scalar, int DIM, int ORDER>
inline void save(Archive& ar, const basalt::RdSpline<DIM, ORDER, Scalar>& spline) {
  ar(spline.minTimeNs());
  ar(spline.getTimeIntervalNs());
  ar(spline.getKnots());
}

template <class Archive, class Scalar, int DIM, int ORDER>
inline void load(Archive& ar, basalt::RdSpline<DIM, ORDER, Scalar>& spline) {
  int64_t start_t_ns;
  int64_t dt_ns;
  Eigen::aligned_deque<Eigen::Matrix<Scalar, DIM, 1>> knots;

  ar(start_t_ns);
  ar(dt_ns);
  ar(knots);

  basalt::RdSpline<DIM, ORDER, Scalar> new_spline(dt_ns, start_t_ns);
  for (const auto& k : knots) {
    new_spline.knotsPushBack(k);
  }
  spline = new_spline;
}

template <class Archive, class Scalar>
inline void serialize(Archive& ar, basalt::Calibration<Scalar>& cam) {
  ar(cereal::make_nvp("T_imu_cam", cam.T_i_c), cereal::make_nvp("intrinsics", cam.intrinsics),
     cereal::make_nvp("resolution", cam.resolution),
     cereal::make_nvp("calib_accel_bias", cam.calib_accel_bias.getParam()),
     cereal::make_nvp("calib_gyro_bias", cam.calib_gyro_bias.getParam()),
     cereal::make_nvp("imu_update_rate", cam.imu_update_rate), cereal::make_nvp("accel_noise_std", cam.accel_noise_std),
     cereal::make_nvp("gyro_noise_std", cam.gyro_noise_std), cereal::make_nvp("accel_bias_std", cam.accel_bias_std),
     cereal::make_nvp("gyro_bias_std", cam.gyro_bias_std),
     cereal::make_nvp("cam_time_offset_ns", cam.cam_time_offset_ns), cereal::make_nvp("vignette", cam.vignette));
}

template <class Archive, class Scalar>
inline void serialize(Archive& ar, basalt::MocapCalibration<Scalar>& cam) {
  ar(cereal::make_nvp("T_mocap_world", cam.T_moc_w), cereal::make_nvp("T_imu_marker", cam.T_i_mark),
     cereal::make_nvp("mocap_time_offset_ns", cam.mocap_time_offset_ns),
     cereal::make_nvp("mocap_to_imu_offset_ns", cam.mocap_to_imu_offset_ns));
}

}  // namespace cereal
