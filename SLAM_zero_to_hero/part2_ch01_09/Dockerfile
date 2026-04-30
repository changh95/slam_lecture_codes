FROM slam:base

# Install DBoW2
RUN git clone https://github.com/dorian3d/DBoW2.git && \
    cd DBoW2 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4 && \
    make install && \
    ldconfig && \
    cd ../..

WORKDIR /workspace
COPY . /workspace/part2_ch01_09

RUN cd /workspace/part2_ch01_09 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch01_09/build
