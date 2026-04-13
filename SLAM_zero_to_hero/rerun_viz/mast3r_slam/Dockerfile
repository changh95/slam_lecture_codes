# MASt3R-SLAM Rerun visualization - built on the profiler image which
# already has all Blackwell/torch 2.6 patches applied.
FROM slam:mast3r_slam-profiler

RUN pip install --no-cache-dir rerun-sdk plyfile && \
    pip install --no-cache-dir "numpy==1.26.4" && \
    find /MASt3R-SLAM -name '*.py' -exec sed -i 's/np\.unicode_/np.str_/g' {} +

COPY track_rerun.py /rerun_viz/track_rerun.py
COPY rerun_patch.py /rerun_viz/rerun_patch.py

RUN mkdir -p /output

EXPOSE 9876 9090

WORKDIR /MASt3R-SLAM

CMD ["python3", "/rerun_viz/track_rerun.py", "--help"]
