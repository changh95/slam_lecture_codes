// Copyright 2023-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Visual-Intertial Tracking tracker C++ helper.
 * @author Jakob Bornecrantz <jakob@collabora.com>
 * @author Simon Zeni <simon.zeni@collabora.com>
 * @author Mateo de Mayo <mateo.demayo@collabora.com>
 */

#pragma once

#include "vit_interface.h"

#include <chrono>
#include <iostream>
#include <vector>

namespace vit {

typedef vit_result_t Result;
typedef vit_image_format_t ImageFormat;
typedef vit_camera_distortion_t CameraDistortion;
typedef vit_tracker_extension_t TrackerExtension;
typedef vit_tracker_extension_set_t TrackerExtensionSet;
typedef vit_tracker_timing_titles TrackerTimingTitles;
typedef vit_tracker_t Tracker;
typedef vit_config_t Config;
typedef vit_pose_t Pose;
typedef vit_config_t Config;
typedef vit_imu_sample_t ImuSample;
typedef vit_mask_t Mask;
typedef vit_img_sample_t ImgSample;
typedef vit_pose_data_t PoseData;
typedef vit_pose_timing_t PoseTiming;
typedef vit_pose_feature_t PoseFeature;
typedef vit_pose_features_t PoseFeatures;
typedef vit_camera_calibration_t CameraCalibration;
typedef vit_inertial_calibration_t InertialCalibration;
typedef vit_imu_calibration_t ImuCalibration;

struct TimeStats {
	TrackerExtensionSet enabled_exts{};
	int64_t ts = INT64_MIN;
	std::vector<int64_t> timings{};
	std::vector<std::vector<vit::PoseFeature>> features_per_cam{};
	const char **timing_titles = nullptr;

  public:
	void addTime(const std::string_view name, int64_t ts = INT64_MIN) {
		if (!enabled_exts.has_pose_timing) {
			return;
		}

		if (timing_titles != nullptr) {
			const std::string expected(timing_titles[timings.size()]);
			if (expected != name) {
				std::cout << "Invalid timing stage\n";
				std::cout << "expected: " << expected;
				std::cout << ", got: " << name << std::endl;
				std::abort();
			}
		}
		if (ts == INT64_MIN) {
			ts = std::chrono::steady_clock::now().time_since_epoch().count();
		}

		timings.push_back(ts);
	}

	void addFeature(size_t cam, const vit::PoseFeature &feature) {
		if (!enabled_exts.has_pose_features) {
			return;
		}

		if (cam >= features_per_cam.size()) {
			std::cout << "Invalid camera index\n";
			std::cout << "has: " << features_per_cam.size();
			std::cout << ", got: " << cam << std::endl;
			std::abort();
		}

		features_per_cam.at(cam).push_back(feature);
	}
};

} // namespace vit

struct vit_tracker {
	virtual ~vit_tracker() = default;

	virtual vit_result_t has_image_format(vit_image_format_t fmt, bool *out_supported) const = 0;
	virtual vit_result_t get_supported_extensions(vit_tracker_extension_set_t *out_extensions) const = 0;
	virtual vit_result_t get_enabled_extensions(vit_tracker_extension_set_t *out_extensions) const = 0;
	virtual vit_result_t enable_extension(vit_tracker_extension_t extension, bool enabled) = 0;
	virtual vit_result_t start() = 0;
	virtual vit_result_t stop() = 0;
	virtual vit_result_t reset() = 0;
	virtual vit_result_t is_running(bool *out_running) const = 0;
	virtual vit_result_t add_imu_calibration(const vit_imu_calibration_t *calibration) = 0;
	virtual vit_result_t add_camera_calibration(const vit_camera_calibration_t *calibration) = 0;
	virtual vit_result_t push_imu_sample(const vit_imu_sample_t *sample) = 0;
	virtual vit_result_t push_img_sample(const vit_img_sample_t *sample) = 0;
	virtual vit_result_t pop_pose(vit_pose_t **pose) = 0;
	virtual vit_result_t get_timing_titles(vit_tracker_timing_titles *out_titles) const = 0;
};

struct vit_pose {
	virtual ~vit_pose() = default;

	virtual vit_result_t get_data(vit_pose_data *out_data) const = 0;
	virtual vit_result_t get_timing(vit_pose_timing *out_timing) const = 0;
	virtual vit_result_t get_features(uint32_t camera_index, vit_pose_features_t *out_features) const = 0;
};
