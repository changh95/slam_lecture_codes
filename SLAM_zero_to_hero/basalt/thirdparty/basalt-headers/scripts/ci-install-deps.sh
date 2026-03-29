#!/usr/bin/env bash
#
# Helper script for ci to separately build and install submodules.
# Assumed to be called from within the main source directory.

set -x
set -e

# pass build type as first argument to script; default is Release
BUILD_TYPE="${1:-Release}"

mkdir -p external

# on macOS we build with Eigen from homebrew
if ! [[ $OSTYPE == 'darwin'* ]]; then
    cmake -B external/build-eigen \
          thirdparty/eigen/ \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_TESTING=OFF \
          -DCMAKE_INSTALL_PREFIX=external/install
    cmake --build external/build-eigen --target install
fi

cmake -B external/build-sophus \
      thirdparty/Sophus/ \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_SOPHUS_TESTS=OFF \
      -DBUILD_SOPHUS_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=external/install \
      -DCMAKE_PREFIX_PATH="$PWD"/external/install
cmake --build external/build-sophus --target install

cmake -B external/build-cereal \
      thirdparty/cereal/ \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DJUST_INSTALL_CEREAL=ON \
      -DCMAKE_INSTALL_PREFIX=external/install
cmake --build external/build-cereal --target install
