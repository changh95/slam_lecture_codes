#ifndef LIVOX_MSG_COMPAT_H_
#define LIVOX_MSG_COMPAT_H_

// Auto-detect available Livox message packages using __has_include
#if __has_include(<livox_ros_driver2/msg/custom_msg.hpp>)
  #include <livox_ros_driver2/msg/custom_msg.hpp>
  #ifndef USE_LIVOX_ROS_DRIVER2
    #define USE_LIVOX_ROS_DRIVER2
  #endif
#endif

#if __has_include(<livox_interfaces/msg/custom_msg.hpp>)
  #include <livox_interfaces/msg/custom_msg.hpp>
  #ifndef USE_LIVOX_INTERFACES
    #define USE_LIVOX_INTERFACES
  #endif
#endif

#if !defined(USE_LIVOX_ROS_DRIVER2) && !defined(USE_LIVOX_INTERFACES)
  #error "No Livox message package found. Install livox_ros_driver2 or livox_interfaces."
#endif

// Default namespace alias: prefer livox_ros_driver2 if available
#if defined(USE_LIVOX_ROS_DRIVER2)
  namespace livox_custom_msg = livox_ros_driver2::msg;
#elif defined(USE_LIVOX_INTERFACES)
  namespace livox_custom_msg = livox_interfaces::msg;
#endif

#endif // LIVOX_MSG_COMPAT_H_
