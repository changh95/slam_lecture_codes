// Copyright 2023-2024, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Visual-Intertial Tracking interface header.
 * @author Mateo de Mayo <mateo.demayo@collabora.com>
 * @author Simon Zeni <simon.zeni@collabora.com>
 * @author Jakob Bornecrantz <jakob@collabora.com>
 *
 * This file contains the declaration of the @ref vit_tracker struct. This
 * header is intended to appear in both consumers (e.g., Monado) and external
 * visual-inertial tracking (VIT) systems (e.g., Basalt). The implementation of
 * `vit_interface` is provided by the external system. Additional data types are
 * declared for the communication between consumers and the system.
 */

#pragma once

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

//! Compatibility with these values should be checked against @ref vit_api_get_version.
#define VIT_HEADER_VERSION_MAJOR 2 //!< API Breakages
#define VIT_HEADER_VERSION_MINOR 0 //!< Backwards compatible API changes
#define VIT_HEADER_VERSION_PATCH 1 //!< Backw. comp. .h-implemented changes

#define VIT_CAMERA_CALIBRATION_DISTORTION_MAX_COUNT 32

/*!
 * Result type used by the VIT system
 *
 * 0 is @ref VIT_SUCCESS, positive values are non fatal results and negative values are errors.
 */
typedef enum vit_result {
	/*!
	 * Operation suceeded
	 */
	VIT_SUCCESS = 0,

	/*!
	 * The operation is not available on the current version
	 */
	VIT_ERROR_INVALID_VERSION = -1,

	/*!
	 * The operation received an invalid value.
	 */
	VIT_ERROR_INVALID_VALUE = -2,

	/*!
	 * The operation was not able to allocate memory to pursue the operation.
	 */
	VIT_ERROR_ALLOCATION_FAILURE = -3,

	/*!
	 * The operation requires an extension that is not supported.
	 */
	VIT_ERROR_NOT_SUPPORTED = -4,

	/*!
	 * The operation requires an extension that is not enabled.
	 */
	VIT_ERROR_NOT_ENABLED = -5,
} vit_result_t;

/*!
 * Image formats.
 */
typedef enum vit_image_format {
	//! 8-bit luminance
	VIT_IMAGE_FORMAT_L8 = 1,
	//! 16-bit luminance
	VIT_IMAGE_FORMAT_L16 = 2,
	//! 24-bit rgb, tightly packed.
	VIT_IMAGE_FORMAT_R8G8B8 = 3,
} vit_image_format_t;

/*!
 * Camera calibration types.
 */
typedef enum vit_camera_distortion {
	//! No distortion (pre-disotorted).
	VIT_CAMERA_DISTORTION_NONE,
	//! Distortion radial-tangential (OpenCV), 4 parameters.
	VIT_CAMERA_DISTORTION_RT4,
	//! Distortion radial-tangential (OpenCV), 5 parameters.
	VIT_CAMERA_DISTORTION_RT5,
	//! Distortion radial-tangential (OpenCV), 8 parameters.
	VIT_CAMERA_DISTORTION_RT8,
	//! Distortion Kannala-Brandt (OpenCV fisheye), 4 parameters.
	VIT_CAMERA_DISTORTION_KB4,
} vit_camera_distortion_t;

/*!
 * Optional functionality a tracker can provide.
 */
typedef enum vit_tracker_extension {
	//! Allows to specify camera calibration data programatically
	VIT_TRACKER_EXTENSION_ADD_CAMERA_CALIBRATION,
	//! Allows to specify IMU calibration data programatically
	VIT_TRACKER_EXTENSION_ADD_IMU_CALIBRATION,
	//! Optionally provide timing information with the returned poses
	VIT_TRACKER_EXTENSION_POSE_TIMING,
	//! Optionally provide feature count information with the returned poses
	VIT_TRACKER_EXTENSION_POSE_FEATURES,
	//! Number of extensions
	VIT_TRACKER_EXTENSION_COUNT,
} vit_tracker_extension_t;

/*!
 * Set of extensions. Field order matches enumerators in @ref vit_tracker_extension.
 */
typedef struct vit_tracker_extension_set {
	bool has_add_camera_calibration;
	bool has_add_imu_calibration;
	bool has_pose_timing;
	bool has_pose_features;
} vit_tracker_extension_set_t;

/*!
 * Visual-Inertial Tracking interface, opaque type.
 */
struct vit_tracker;
typedef struct vit_tracker vit_tracker_t;

/*!
 * Pose interface, opaque type.
 */
struct vit_pose;
typedef struct vit_pose vit_pose_t;

/*!
 * Names of the timestamps returned by `vit_pose_get_timings`
 */
typedef struct vit_tracker_timing_titles {
	uint32_t count;		 //! Number of titles
	const char **titles; //! Names of the measures timestamps
} vit_tracker_timing_titles;

/*!
 * Parameters for creating the system pipeline.
 */
typedef struct vit_config {
	//! Path to a implementation-specific config file. If null, use defaults.
	const char *file;

	//! Number of cameras to use. Required.
	uint32_t cam_count;

	//! Number of IMU to use. Required.
	uint32_t imu_count;

	//! If supported, whether to open the system's UI.
	bool show_ui;
} vit_config_t;

/*!
 * IMU sample type feed into VIT tracker
 */
typedef struct vit_imu_sample {
	//! In nanoseconds
	int64_t timestamp;

	//! Acceleration in meters per second squared (m/s²)
	float ax, ay, az;

	//! Gyro in radians per second (rad/s)
	float wx, wy, wz;
} vit_imu_sample_t;

/*!
 * Region in image space that this mask covers.
 */
typedef struct vit_mask {
	//! In pixels.
	float x, y, w, h;
} vit_mask_t;

/*!
 * Image sample type feed into VIT tracker
 *
 * Can easily be converted into an OpenCV Matrix for processing.
 */
typedef struct vit_img_sample {
	uint32_t cam_index;

	//! In nanoseconds, must increase monotonically.
	int64_t timestamp;

	// !Image data
	uint8_t *data;
	uint32_t width, height;
	uint32_t stride, size;
	vit_image_format_t format;

	//! Regions to ignore
	uint32_t mask_count;
	vit_mask_t *masks;
} vit_img_sample_t;

/*!
 * Data that is always returned from tracker.
 */
typedef struct vit_pose_data {
	//! In nanoseconds, must increase monotonically.
	int64_t timestamp;

	//! Position vector.
	float px, py, pz;

	//! Orientation quaternion.
	float ox, oy, oz, ow;

	//! Linear velocity.
	float vx, vy, vz;
} vit_pose_data_t;

/*!
 * Result of pose timing request function.
 */
typedef struct vit_pose_timing {
	uint32_t count;
	const int64_t *timestamps;
} vit_pose_timing_t;

/*!
 * One single feature, element of @ref vit_pose_features result.
 */
typedef struct vit_pose_feature {
	int64_t id;
	float u, v, depth;
} vit_pose_feature_t;

/*!
 * Result of pose feature request function.
 */
typedef struct vit_pose_features {
	uint32_t count;
	const struct vit_pose_feature *features;
} vit_pose_features_t;

/*!
 * Container of parameters for a pinhole camera calibration (fx, fy, cx, cy)
 * with an optional distortion.
 *
 *`distortion_model` and its corresponding `distortion` parameters are not
 * standardized in this struct to facilitate implementation prototyping.
 */
typedef struct vit_camera_calibration {
	uint32_t camera_index; // <! For multi-camera setup. For stereo 0 ~ left, 1 ~ right.

	int width, height; //<! Resolution
	double frequency;  //<! Frames per second
	double fx, fy;	   //<! Focal point
	double cx, cy;	   //<! Principal point
	enum vit_camera_distortion model;
	uint32_t distortion_count;
	double distortion[VIT_CAMERA_CALIBRATION_DISTORTION_MAX_COUNT]; //!< Parameters for the distortion model
	double transform[4 * 4]; //!< Row-major 4x4 camera transform w.r.t. the IMU (i.e., T_imu_cam)
} vit_camera_calibration_t;

typedef struct vit_inertial_calibration {
	// Calibration intrinsics to apply to each raw measurement.

	//! Row major 3x3 linear transformation for raw measurements alignment and scaling.
	double transform[3 * 3];

	//! Offset to add to raw measurements; called bias in other contexts.
	double offset[3];

	// Parameters for the random processes that model this IMU. See section "2.1
	// Gyro Noise Model" of N. Trawny and S. I. Roumeliotis, "Indirect Kalman
	// Filter for 3D Attitude Estimation". Analogous for accelerometers.
	// http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf#page=15

	//! IMU internal bias ~ wiener process with steps N(0, σ²); this field is σ;
	//! [σ] = U / sqrt(sec³) with U = rad if gyroscope, U = m/s if accelerometer.
	double bias_std[3];

	//! IMU measurement noise ~ N(0, σ²); this field is σ.
	//! [σ] = U / sqrt(sec) with U = rad if gyroscope, U = m/s if accelerometer.
	double noise_std[3];
} vit_inertial_calibration_t;

/*!
 * Calibration for one IMU.
 */
typedef struct vit_imu_calibration {
	uint32_t imu_index; //!< For multi-imu setup, usually just 0.

	double frequency; //!< Samples per second
	struct vit_inertial_calibration accel;
	struct vit_inertial_calibration gyro;
} vit_imu_calibration_t;

/*
 *
 * Function prototypes.
 *
 */

typedef vit_result_t (*PFN_vit_api_get_version)(uint32_t *out_major, uint32_t *out_minor, uint32_t *out_patch);
typedef vit_result_t (*PFN_vit_tracker_create)(const vit_config_t *config, vit_tracker_t **out_tracker);
typedef void (*PFN_vit_tracker_destroy)(vit_tracker_t *tracker);
typedef vit_result_t (*PFN_vit_tracker_has_image_format)(const vit_tracker_t *tracker, vit_image_format_t image_format,
														 bool *out_supported);
typedef vit_result_t (*PFN_vit_tracker_get_supported_extensions)(const vit_tracker_t *tracker,
																 vit_tracker_extension_set_t *out_exts);
typedef vit_result_t (*PFN_vit_tracker_get_enabled_extensions)(const vit_tracker_t *tracker,
															   vit_tracker_extension_set_t *out_exts);
typedef vit_result_t (*PFN_vit_tracker_enable_extension)(vit_tracker_t *tracker, vit_tracker_extension_t ext,
														 bool enable);
typedef vit_result_t (*PFN_vit_tracker_start)(vit_tracker_t *tracker);
typedef vit_result_t (*PFN_vit_tracker_stop)(vit_tracker_t *tracker);
typedef vit_result_t (*PFN_vit_tracker_reset)(vit_tracker_t *tracker);
typedef vit_result_t (*PFN_vit_tracker_is_running)(const vit_tracker_t *tracker, bool *out_bool);
typedef vit_result_t (*PFN_vit_tracker_push_imu_sample)(vit_tracker_t *tracker, const vit_imu_sample_t *sample);
typedef vit_result_t (*PFN_vit_tracker_push_img_sample)(vit_tracker_t *tracker, const vit_img_sample_t *sample);
typedef vit_result_t (*PFN_vit_tracker_add_imu_calibration)(vit_tracker_t *tracker,
															const vit_imu_calibration_t *calibration);
typedef vit_result_t (*PFN_vit_tracker_add_camera_calibration)(vit_tracker_t *tracker,
															   const vit_camera_calibration_t *calibration);
typedef vit_result_t (*PFN_vit_tracker_pop_pose)(vit_tracker_t *tracker, vit_pose_t **out_pose);
typedef vit_result_t (*PFN_vit_tracker_get_timing_titles)(const vit_tracker_t *tracker,
														  vit_tracker_timing_titles *out_titles);
typedef void (*PFN_vit_pose_destroy)(vit_pose_t *pose);
typedef vit_result_t (*PFN_vit_pose_get_data)(const vit_pose_t *pose, vit_pose_data_t *out_data);
typedef vit_result_t (*PFN_vit_pose_get_timing)(const vit_pose_t *pose, vit_pose_timing_t *out_timing);
typedef vit_result_t (*PFN_vit_pose_get_features)(const vit_pose_t *pose, uint32_t camera_index,
												  vit_pose_features_t *out_features);

/*
 *
 * Functions.
 *
 */

#ifdef VIT_INTERFACE_IMPLEMENTATION

/*!
 * Returns the API version implemented by the VIT system.
 */
vit_result_t vit_api_get_version(uint32_t *out_major, uint32_t *out_minor, uint32_t *out_patch);

/*!
 * Creates a new VIT tracker. The caller is responsible of destroying it when done.
 */
vit_result_t vit_tracker_create(const vit_config_t *config, vit_tracker_t **out_tracker);

/*!
 * Destroys the VIT tracker and free all resources allocated.
 */
void vit_tracker_destroy(vit_tracker_t *tracker);

/*!
 * Verifies if the tracker supports a given `vit_image_format_t`.
 */
vit_result_t vit_tracker_has_image_format(const vit_tracker_t *tracker, vit_image_format_t image_format,
										  bool *out_supported);

/*!
 * Returns a set with the extensions supported by this tracker.
 *
 * @see vit_tracker_extension_t
 */

vit_result_t vit_tracker_get_supported_extensions(const vit_tracker_t *tracker, vit_tracker_extension_set_t *out_exts);

/*!
 * Returns a set with the extensions enabled in this tracker.
 *
 * @see vit_tracker_extension_t
 */

vit_result_t vit_tracker_get_enabled_extensions(const vit_tracker_t *tracker, vit_tracker_extension_set_t *out_exts);

/*!
 * Enables or disables a tracker extension.
 *
 * @see vit_tracker_extension_t
 */
vit_result_t vit_tracker_enable_extension(vit_tracker_t *tracker, vit_tracker_extension_t ext, bool value);

/*!
 * Starts the VIT tracker. Image and IMU samples can be pushed and pose can be retrieved.
 *
 * This function must be non blocking. The VIT system implementing it is expected to start its own event loop.
 */
vit_result_t vit_tracker_start(vit_tracker_t *tracker);

/*!
 * Stops the VIT tracker. The tracker wont accept image and IMU samples, and will not return poses.
 */
vit_result_t vit_tracker_stop(vit_tracker_t *tracker);

/*!
 * Resets the VIT tracker. The tracker internal state will be set to its original state.
 */
vit_result_t vit_tracker_reset(vit_tracker_t *tracker);

/*!
 * Verifies if the tracker is running.
 */
vit_result_t vit_tracker_is_running(const vit_tracker_t *tracker, bool *out_bool);

/*!
 * Push an IMU sample into the tracker.
 *
 * There must be a single producer thread pushing samples.
 * Samples must have monotonically increasing timestamps.
 * The implementation must be non-blocking.
 * Thus, a separate consumer thread should process the samples.
 */
vit_result_t vit_tracker_push_imu_sample(vit_tracker_t *tracker, const vit_imu_sample_t *sample);

/*!
 * Push an image sample into the tracker.
 *
 * Same conditions as @ref push_imu_sample apply.
 * When using N>1 cameras, the N frames must be pushed following @ref cam_index order.
 * The bundle of N frames must have the same timestamps.
 */
vit_result_t vit_tracker_push_img_sample(vit_tracker_t *tracker, const vit_img_sample_t *sample);

/*!
 * Adds an inertial measurement unit calibration to the tracker. The tracker must not be started.
 *
 * Returns `VIT_ERROR_NOT_SUPPORTED` if the tracker doesn't offer the extension.
 *
 * @see vit_tracker_get_supported_extensions
 * @see vit_tracker_extension_t
 */
vit_result_t vit_tracker_add_imu_calibration(vit_tracker_t *tracker, const vit_imu_calibration_t *calibration);

/*!
 * Adds a camera calibration to the tracker. The tracker must not be started.
 *
 * Returns `VIT_ERROR_NOT_SUPPORTED` if the tracker doesn't offer the extension.
 *
 * @see vit_tracker_get_supported_extensions
 * @see vit_tracker_extension_t
 */
vit_result_t vit_tracker_add_camera_calibration(vit_tracker_t *tracker, const vit_camera_calibration_t *calibration);

/*!
 * Get the pose from the front of the tracking queue from the VIT tracker
 *
 * This function must be non-blocking and consumed by a single consummer.
 *
 * If @p out_pose is NULL the pose will be immediately destroyed.
 *
 * @param[out] out_pose Pose returned to the caller, NULL if there is not pose available or on error.
 */
vit_result_t vit_tracker_pop_pose(vit_tracker_t *tracker, vit_pose_t **out_pose);

/*!
 * Get the titles of the timestamps measured by the pose timings.
 *
 * Returns `VIT_ERROR_NOT_SUPPORTED` if the tracker doesn't offer the pose timing extension.
 */
vit_result_t vit_tracker_get_timing_titles(const vit_tracker_t *tracker, vit_tracker_timing_titles *out_titles);

/*!
 * Destroys a pose. All of the data, timing and features associated to it will be invalidated.
 */
void vit_pose_destroy(vit_pose_t *pose);

/*!
 * Gets the data form a given `vit_pose_t`.
 *
 * The data becomes invalid when the associated pose gets destroyed.
 */
vit_result_t vit_pose_get_data(const vit_pose_t *pose, vit_pose_data_t *out_data);

/*!
 * Gets the timing form a given `vit_pose_t`.
 *
 * The timing data becomes invalid when the associated pose gets destroyed.
 */
vit_result_t vit_pose_get_timing(const vit_pose_t *pose, vit_pose_timing_t *out_timing);

/*!
 * Gets the features form a given `vit_pose_t`.
 *
 * The features data becomes invalid when the associated pose gets destroyed.
 */
vit_result_t vit_pose_get_features(const vit_pose_t *pose, uint32_t camera_index, vit_pose_features_t *out_features);

#endif

#if defined(__cplusplus)
}
#endif
