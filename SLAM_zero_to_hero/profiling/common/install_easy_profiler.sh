#!/bin/bash
set -e
# Install easy_profiler from source (works on x86_64 and arm64)
apt-get update && apt-get install -y git cmake build-essential
cd /tmp
git clone https://github.com/yse/easy_profiler.git
cd easy_profiler
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DEASY_PROFILER_NO_GUI=ON \
  -DEASY_PROFILER_NO_SAMPLES=ON
make -j$(nproc)
make install
ldconfig
# Build the converter tool for JSON export
cd /tmp/easy_profiler/easy_profiler_converter
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp profiler_converter /usr/local/bin/easy_profiler_converter 2>/dev/null || \
cp easy_profiler_converter /usr/local/bin/easy_profiler_converter 2>/dev/null || \
echo "WARNING: easy_profiler_converter binary not found. JSON conversion will not be available."
cd / && rm -rf /tmp/easy_profiler
