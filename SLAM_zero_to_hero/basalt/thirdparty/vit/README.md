<!--
Copyright 2024, Collabora, Ltd.
SPDX-License-Identifier: BSL-1.0

Author: Mateo de Mayo <mateo.demayo@collabora.com>
-->

# Visual-Inertial Tracking Interface

Upstream: <https://gitlab.freedesktop.org/monado/utilities/vit>

A unified interface for visual-inertial tracking (VI-SLAM, VIO, etc) systems to expose. Originally created for [Monado](https://monado.dev), an OpenXR runtime, to communicate with multiple SLAM systems.

This project aims to maintain different parts relevant to the visual-inertial tracking (VIT) interface used in Monado. The interface supersedes a previous C++ version of the interface [`slam_tracker.hpp`](https://gitlab.freedesktop.org/monado/monado/-/blob/20cb556f6d03226f62e6b6cc340a6e6c0c260ee8/src/external/slam_tracker/slam_tracker.hpp).

## Usage

The tracking system needs to implement the [vit_interface.h](./vit_interface.h)
symbols and export them. If the system is written in C++, a class is provided in
[vit_implementation_helper](./vit_implementation_helper.hpp) that you can derive
from to implement the symbols. The consumer can then link against the compiled
object and use the names symbols from the header.

## Examples of VIT systems implementing the interface:

**Public implementations**:

- [Basalt](https://gitlab.freedesktop.org/mateosss/basalt/-/tree/bb5dfb123ede01050f03e38ebb7cc37912263e82)

**Public consumers**:

- [Monado](https://gitlab.freedesktop.org/monado/monado/-/tree/bc8025a6bd9ede8b5f9e02596faa0c08822b760b)
