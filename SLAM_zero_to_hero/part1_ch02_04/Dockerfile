FROM slam:base

# easy_profiler GUI requires Qt6, but slam:base only installed Qt5 so the
# profiler_gui target was silently skipped. Install Qt6 here and rebuild it.
RUN apt-get update && apt-get install -y \
        qt6-base-dev \
        libqt6core5compat6-dev \
        qt6-tools-dev && \
    rm -rf /var/lib/apt/lists/* && \
    cd /easy_profiler/build && \
    cmake .. -DEASY_PROFILER_NO_GUI=OFF && \
    make -j4 profiler_gui && \
    install -m 0755 bin/profiler_gui /usr/local/bin/profiler_gui

WORKDIR /workspace
COPY . /workspace/part1_ch02_04

RUN cd /workspace/part1_ch02_04 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

RUN mkdir -p /output

WORKDIR /workspace/part1_ch02_04/build

CMD ["bash", "-c", "./profile_features && profiler_gui profile.prof"]
