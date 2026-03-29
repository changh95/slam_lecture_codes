# console_bridge

`console_bridge` is a ROS-independent, pure CMake (i.e. non-catkin and non-rosbuild package) that provides logging calls that mirror those found in rosconsole, but for applications that are not necessarily using ROS.

## Quality Declaration

This package claims to be in the **Quality Level 1** category, see the [Quality Declaration](./QUALITY_DECLARATION.md) for more details.

## Features

This library allows to log messages in the standard output and the standard output error based on the log level:

 - CONSOLE_BRIDGE_LOG_DEBUG
 - CONSOLE_BRIDGE_LOG_INFO
 - CONSOLE_BRIDGE_LOG_WARN
 - CONSOLE_BRIDGE_LOG_ERROR
 - CONSOLE_BRIDGE_LOG_NONE
