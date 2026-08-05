/*
 * ros_params_helper.h
 *
 *  Created on: Feb 22, 2013
 *      Author: cforster
 *  Update on: Feb 01, 2025
 *      Author: StrangeFly
 *
 * from libpointmatcher_ros
 */

#ifndef ROS_PARAMS_HELPER_H_
#define ROS_PARAMS_HELPER_H_

#include <sstream>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/srv/get_parameters.hpp>
#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace vk {

template <typename T>
T getParam(const std::string& node, const std::string& name, const T& defaultValue) {
    // Keep popen as a absolute fallback if no Node handle is available
    // But this is very slow!
    try {
        std::string full_node = node;
        if (!full_node.empty() && full_node[0] != '/') full_node = "/" + full_node;
        
        std::string command = "ros2 param get " + full_node + " " + name + " 2>/dev/null";
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
        if (!pipe) {
            return defaultValue;
        }
        std::ostringstream resultStream;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            resultStream << buffer;
        }
        std::string result = resultStream.str();

        size_t pos = result.find(": ");
        if (pos != std::string::npos) {
            std::string valueStr = result.substr(pos + 2);
            if (!valueStr.empty() && valueStr.back() == '\n') valueStr.pop_back();

            if (!valueStr.empty()) {
                std::stringstream ss(valueStr);
                T value;
                if (ss >> value) {
                    return value;
                }
            }
        }
        return defaultValue;
    } catch (...) {
        return defaultValue;
    }
}

template <typename T>
T getParam(const std::string& node, const std::string& name) {
    // override function could has no default value
    if constexpr (std::is_same_v<T, std::string>) {
        return getParam<T>(node, name, ""); // if std::string, default value is ""
    } else if constexpr (std::is_integral_v<T>) {
        return getParam<T>(node, name, 0); // if int. defalt 0
    } else if constexpr (std::is_floating_point_v<T>) {
        return getParam<T>(node, name, 0.0); // if float，default 0.0
    } else {
        throw std::runtime_error("Unsupported type for getParam without default value.");
    }
}

inline
bool hasParam(const rclcpp::Node::SharedPtr &nh, const std::string& name)
{
  return nh->has_parameter(name);
}

template<typename T>
T getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name, const T& defaultValue)
{
  T v;
  if(nh->get_parameter(name, v))
  {
    return v;
  }
  
  // If not found locally, it might be a prefix for a remote node or a local param not yet declared
  if (!nh->has_parameter(name)) {
      nh->declare_parameter(name, defaultValue);
      if (nh->get_parameter(name, v)) return v;
  }

  return defaultValue;
}

// New function for cross-node parameter access
template<typename T>
T getRemoteParam(const rclcpp::Node::SharedPtr &nh, const std::string& remote_node_name, const std::string& param_name, const T& defaultValue)
{
    // Try local first just in case
    std::string full_name = remote_node_name + "." + param_name; // common convention
    T v;
    if (nh->get_parameter(full_name, v)) return v;
    if (nh->get_parameter(param_name, v)) return v;

    // Use SyncParametersClient for high performance cross-node access
    try {
        auto parameters_client = std::make_shared<rclcpp::SyncParametersClient>(nh, remote_node_name);
        // Wait briefly for the service to be available
        if (parameters_client->wait_for_service(std::chrono::milliseconds(100))) {
            auto values = parameters_client->get_parameters({param_name});
            if (!values.empty() && values[0].get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET) {
                return values[0].get_value<T>();
            }
        }
    } catch (...) {
        // Fallback to popen if client fails
        return getParam<T>(remote_node_name, param_name, defaultValue);
    }

    return defaultValue;
}

template<typename T>
T getParam(const rclcpp::Node::SharedPtr &nh, const std::string& name)
{
  T v;
  if (nh->get_parameter(name, v)) {
    RCLCPP_INFO_STREAM(nh->get_logger(), "Found parameter: " << name << ", value: " << v);
    return v;
  }

  // If not found, try to declare it (this might be useful if the parameter is expected to be there)
  // or just return a default constructed T.
  RCLCPP_ERROR_STREAM(nh->get_logger(), "Cannot find value for parameter: " << name << ". Returning default.");
  return T();
}

} // namespace vk

#endif // ROS_PARAMS_HELPER_H_
