FROM slam:base

WORKDIR /workspace

# Build MAGSAC++ (and its bundled Graph-Cut RANSAC) from source so the
# optional ransac_magsac demo compiles. Kept before COPY so the layer caches.
RUN cd /tmp && \
    git clone --recursive --depth 1 https://github.com/danini/magsac.git && \
    cd magsac && mkdir build && cd build && \
    cmake .. -DCREATE_SAMPLE_PROJECT=OFF && \
    make -j$(nproc) MAGSAC GraphCutRANSAC && \
    cp libMAGSAC.a libGraphCutRANSAC.a /usr/local/lib/ && \
    mkdir -p /usr/local/include/pymagsac /usr/local/include/gcransac && \
    cp -r /tmp/magsac/src/pymagsac/include /usr/local/include/pymagsac/ && \
    cp -r /tmp/magsac/graph-cut-ransac/src/pygcransac/include/. /usr/local/include/gcransac/ && \
    rm -rf /tmp/magsac && \
    # Patch upstream header bugs that only surface when consuming the headers:
    # gcransac calls utils::clamp (doesn't exist; std::clamp intended), and
    # magsac's estimators.h uses essential-matrix solvers without including
    # their headers.
    grep -rl "utils::clamp" /usr/local/include/gcransac | \
        xargs -r sed -i -e "s/gcransac::utils::clamp/std::clamp/g" -e "s/utils::clamp/std::clamp/g" && \
    sed -i 's|#include "estimators/essential_estimator.h"|#include "estimators/solver_essential_matrix_five_point_nister.h"\n#include "estimators/solver_essential_matrix_five_point_stewenius.h"\n#include "estimators/solver_essential_matrix_bundle_adjustment.h"\n#include "estimators/essential_estimator.h"|' \
        /usr/local/include/pymagsac/include/estimators.h

COPY . /workspace/part2_ch02_12

# Build the RANSAC/USAC examples
RUN cd /workspace/part2_ch02_12 && \
    mkdir build && cd build && \
    cmake .. && \
    make -j4

WORKDIR /workspace/part2_ch02_12/build
