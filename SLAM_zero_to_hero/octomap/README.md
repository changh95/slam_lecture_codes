# How to build

```
docker build . -tag slam:octomap
```

# How to run

```
docker run -it --env DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:ro slam:octomap
```

```
docker run -it --env DISPLAY=$DISPLAY -v /kitti:/data -v /tmp/.X11-unix/:/tmp/.X11-unix:ro slam:bonxai

# Inside docker image
cd /Bonxai/build/bonxai_map
./benchmark_kitti --clouds /data/sequences/00/velodyne --calib /data/sequences/00/calib.txt 
```
