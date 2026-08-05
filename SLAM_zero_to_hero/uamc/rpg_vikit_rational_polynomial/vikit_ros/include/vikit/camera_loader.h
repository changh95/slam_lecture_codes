/*
 * camera_loader.h
 *
 *  Created on: Feb 11, 2014
 *      Author: cforster
 *  Update on: Feb 01, 2025
 *      Author: StrangeFly
 */

#ifndef VIKIT_CAMERA_LOADER_H_
#define VIKIT_CAMERA_LOADER_H_

#include <string>
#include <vikit/abstract_camera.h>
#include <vikit/pinhole_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/omni_camera.h>
#include <vikit/equidistant_camera.h>
#include <vikit/polynomial_camera.h>
#include <vikit/rational_polynomial_camera.h>
#include <vikit/params_helper.h>

namespace vk {
namespace camera_loader {

/// Load from ROS Namespace
bool loadFromRosNs(const rclcpp::Node::SharedPtr & nh, const std::string& ns, vk::AbstractCamera*& cam);
bool loadFromRosNs(const rclcpp::Node::SharedPtr & nh, const std::string& ns, std::vector<vk::AbstractCamera*>& cam_list);

} // namespace camera_loader
} // namespace vk

#endif // VIKIT_CAMERA_LOADER_H_
