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
COPY . /workspace/2_9

RUN cd /workspace/2_9 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/2_9/build
