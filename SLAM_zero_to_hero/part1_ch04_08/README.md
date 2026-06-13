# Kalibr and Allan Variance ROS

## How to build

```
docker build . -t kalibr
```

## How to run

Kalibr
```
xhost +local:root

docker run -it \
  --net=host \
  --ipc=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --volume=/home/$USER/vio_data:/data \
  --privileged \
  --env="XAUTHORITY=/root/.Xauthority" \
  kalibr
  
# Inside docker container
source ./devel/setup.bash
```

## Dataset

- AR Table Dataset from RPNG
   - Images: https://drive.google.com/file/d/1cmtc24e_W6PFNoviXMbDKId1IwDxpS3O/view?usp=sharing
   - Static IMU: https://drive.google.com/file/d/1hrEO5vq81mvO_lZ_FLU2KOm0bSxQQRQr/view?usp=sharing
   - Aprilgrid yaml: https://drive.google.com/file/d/1aQv0gA_nOOfqYC6_AMskRupcxSZ8GdY0/view?usp=sharing
