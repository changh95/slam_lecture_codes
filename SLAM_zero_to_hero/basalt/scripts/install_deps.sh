#!/bin/bash
##
## BSD 3-Clause License
##
## This file is part of the Basalt project.
## https://gitlab.com/VladyslavUsenko/basalt.git
##
## Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
## All rights reserved.
##

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ "$OSTYPE" == "darwin"* ]]; then
	brew install boost opencv cmake pkgconfig lz4 clang-format tbb glew eigen ccache lz4 fmt llvm
else
	DISTRO=$( awk -F= '/^ID/{print $2}' /etc/os-release )
	if [ "$DISTRO" == "fedora" ]; then
		sudo dnf install -y gcc g++ cmake ninja-build mold git tbb-devel eigen3-devel glew-devel ccache libjpeg-turbo-devel libpng-devel lz4-devel bzip2-devel boost-regex boost-filesystem boost-date-time boost-program-options gtest-devel opencv-devel
	else
		sudo apt-get install -y gcc g++ cmake ninja-build mold git libtbb-dev libeigen3-dev libglew-dev ccache libjpeg-dev libpng-dev liblz4-dev libbz2-dev libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev libgtest-dev libopencv-dev libfmt-dev libc++-14-dev libc++1-14 libc++abi1-14 libunwind-14 libunwind-14-dev libwayland-bin libepoxy-dev
	fi
fi
