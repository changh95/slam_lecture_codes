# Deep Learning based Visual Place Recognition Tutorial

- uses Python codes (Mostly because VPR codes are supported in python natively)
- Switchable descriptors (HDC-DELF, AlexNet-conv, NetVLAD, PatchNetVLAD, CosPlace, EigenPlaces)
- Switchable day/night datasets (GardensPoint, StLucia, SFU)

---

# How to build & run

# Docker build

```
xhost +local:docker
docker build . -t slam:vpr
docker run -it --env DISPLAY=$DISPLAY --privileged --runtime nvidia --gpus all -v /tmp/.X11-unix/:/tmp/.X11-unix:ro slam:vpr

# Inside docker container
cd VPR_Tutorial
source activate vprtutorial
python3 demo.py (--descriptor PatchNetVLAD)
```

## Results

![](output.png)
