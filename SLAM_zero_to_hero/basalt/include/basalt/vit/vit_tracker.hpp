/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2023-2024, Collabora Ltd.
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
@brief Implementation details for the opaque vit::Tracker type.
@author Simon Zeni <simon.zeni@collabora.com>
@author Mateo de Mayo <mateo.demayo@collabora.com>
*/

#pragma once

#include <vit_interface.h>
#include <iostream>
#include <vit_implementation_helper.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace basalt::vit_implementation {

struct Tracker final : vit::Tracker {
  Tracker(const vit::Config *config);
  ~Tracker() override = default;

  vit::Result has_image_format(vit::ImageFormat fmt, bool *out_supported) const override;
  vit::Result get_supported_extensions(vit::TrackerExtensionSet *out_exts) const override;
  vit::Result get_enabled_extensions(vit::TrackerExtensionSet *out_exts) const override;
  vit::Result enable_extension(vit::TrackerExtension extension, bool enabled) override;
  vit::Result start() override;
  vit::Result stop() override;
  vit::Result reset() override;
  vit::Result is_running(bool *running) const override;
  vit::Result add_imu_calibration(const vit::ImuCalibration *calibration) override;
  vit::Result add_camera_calibration(const vit::CameraCalibration *calibration) override;
  vit::Result push_imu_sample(const vit::ImuSample *sample) override;
  vit::Result push_img_sample(const vit::ImgSample *sample) override;
  vit::Result pop_pose(vit::Pose **pose) override;
  vit::Result get_timing_titles(vit::TrackerTimingTitles *out_titles) const override;

 private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

struct Pose final : vit::Pose {
  ~Pose() override = default;

  vit::Result get_data(vit::PoseData *out_data) const override;
  vit::Result get_timing(vit::PoseTiming *out_timing) const override;
  vit::Result get_features(uint32_t camera_index, vit::PoseFeatures *out_features) const override;

 private:
  friend Tracker;
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

}  // namespace basalt::vit_implementation
