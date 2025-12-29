# Cartographer

> We use cartographer to make 2D occupancy grid map

# Dataset

Download the .bag file from from [link to file 1](https://drive.google.com/file/d/1iGdhoPW4hYI-0rPctSQdNKJ7YE9TYMd7/view?usp=sharing) [link to file 2]().

If you want any other sequences than KITTI odometry 05, you can use [kitti2bag](https://github.com/ulterzlw/kitti2bag) to turn any raw KITTI dataset into a ROS bag file.

## How to run
```
docker build -t slam:cartographer

# Open 2 terminals, each terminal should run the code below
docker run -it -e "DISPLAY=$DISPLAY" -e "QT_X11_NO_MITSHM=1" -e "XAUTHORITY=$XAUTH" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v ${PWD}/:/data --net=host --ipc=host --privileged slam:cartographer

# Inside the first container
roslaunch cartographer_ros velodyne_kitti_uamc.launch

# Inside the second container
rosbag play /data/2011_09_30_0018.bag 
rosrun map_server map_saver -f /data/map
```

## Output

![](./output.gif)

## Special thanks

Special thanks to [Juwon Jason Kim](https://github.com/U-AMC)
