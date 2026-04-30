FROM slam:base

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    libpcl-dev \
    liboctomap-dev \
    octomap-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone Bonxai (header-only library)
RUN git clone https://github.com/facontidavide/Bonxai.git /tmp/bonxai && \
    mkdir -p /workspace/part2_ch03_08/3rdparty/bonxai && \
    cp -r /tmp/bonxai/bonxai_core/include/bonxai/* /workspace/part2_ch03_08/3rdparty/bonxai/ && \
    rm -rf /tmp/bonxai

# Copy project files
COPY . /workspace/part2_ch03_08

# Build the tutorial examples
RUN cd /workspace/part2_ch03_08 && \
    mkdir -p build && cd build && \
    cmake -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        .. && \
    ninja

WORKDIR /workspace/part2_ch03_08/build
