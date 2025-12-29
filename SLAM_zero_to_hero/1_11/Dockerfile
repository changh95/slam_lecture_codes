FROM slam:base

WORKDIR /workspace
COPY . /workspace/1_11

RUN cd /workspace/1_11 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

# Create output directory for profile files
RUN mkdir -p /output

WORKDIR /workspace/1_11/build

# Run profiler and open GUI if DISPLAY is set, otherwise just save the profile
CMD ["sh", "-c", "./profile_features && if [ -n \"$DISPLAY\" ]; then profiler_gui profile.prof; else cp profile.prof /output/ && echo 'Profile saved to /output/profile.prof'; fi"]
