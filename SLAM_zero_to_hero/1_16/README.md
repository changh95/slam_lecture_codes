# ROS Fundamentals: ROS1 and ROS2

This tutorial covers Robot Operating System (ROS) fundamentals for both ROS1 (Noetic) and ROS2 (Humble), with examples in Python and C++.

## Topics Covered

| Topic | Description |
|-------|-------------|
| Nodes | Basic execution units in ROS |
| Topics | Publish/Subscribe communication |
| Messages | Data structures for communication |
| Services | Request/Response pattern |
| Parameters | Runtime configuration |
| Launch Files | Multi-node startup |

---

## ROS1 vs ROS2 Comparison

| Feature | ROS1 (Noetic) | ROS2 (Humble) |
|---------|---------------|---------------|
| Master | Required (`roscore`) | Not required (DDS) |
| Build System | catkin | colcon |
| Python | rospy | rclpy |
| C++ | roscpp | rclcpp |
| Launch | XML | Python/XML/YAML |
| OS Support | Ubuntu 20.04 | Ubuntu 22.04 |

---

## Directory Structure

```
1_16/
├── ros1/
│   ├── Dockerfile
│   ├── src/
│   │   └── slam_tutorial/
│   │       ├── CMakeLists.txt
│   │       ├── package.xml
│   │       ├── scripts/
│   │       │   ├── publisher.py
│   │       │   └── subscriber.py
│   │       └── src/
│   │           ├── publisher.cpp
│   │           └── subscriber.cpp
│   └── README.md
│
└── ros2/
    ├── Dockerfile
    ├── src/
    │   └── slam_tutorial/
    │       ├── CMakeLists.txt
    │       ├── package.xml
    │       ├── slam_tutorial/
    │       │   ├── __init__.py
    │       │   ├── publisher.py
    │       │   └── subscriber.py
    │       └── src/
    │           ├── publisher.cpp
    │           └── subscriber.cpp
    └── README.md
```

---

## Quick Start

### ROS1 (Noetic)

```bash
cd ros1
docker build . -t slam_ros1
docker run -it slam_ros1

# Inside container
source /opt/ros/noetic/setup.bash
cd /catkin_ws
catkin_make
source devel/setup.bash

# Terminal 1: Start roscore
roscore

# Terminal 2: Run publisher
rosrun slam_tutorial publisher.py

# Terminal 3: Run subscriber
rosrun slam_tutorial subscriber.py
```

### ROS2 (Humble)

```bash
cd ros2
docker build . -t slam_ros2
docker run -it slam_ros2

# Inside container
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build
source install/setup.bash

# Terminal 1: Run publisher
ros2 run slam_tutorial publisher

# Terminal 2: Run subscriber
ros2 run slam_tutorial subscriber
```

---

## Core Concepts

### 1. Nodes
Nodes are independent processes that perform computation.

**ROS1 Python:**
```python
#!/usr/bin/env python3
import rospy

rospy.init_node('my_node')
rospy.loginfo('Node started')
rospy.spin()
```

**ROS2 Python:**
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node started')

def main():
    rclpy.init()
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### 2. Topics (Pub/Sub)

**ROS1 Publisher (Python):**
```python
import rospy
from std_msgs.msg import String

rospy.init_node('publisher')
pub = rospy.Publisher('chatter', String, queue_size=10)
rate = rospy.Rate(10)  # 10 Hz

while not rospy.is_shutdown():
    pub.publish('Hello ROS1')
    rate.sleep()
```

**ROS2 Publisher (Python):**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.pub = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello ROS2'
        self.pub.publish(msg)
```

### 3. Common Message Types

| Type | ROS Package | Description |
|------|-------------|-------------|
| `String` | std_msgs | Text data |
| `Int32` | std_msgs | Integer |
| `Float64` | std_msgs | Double precision float |
| `Pose` | geometry_msgs | Position + Orientation |
| `Twist` | geometry_msgs | Linear + Angular velocity |
| `Image` | sensor_msgs | Camera image |
| `PointCloud2` | sensor_msgs | 3D point cloud |
| `Imu` | sensor_msgs | IMU data |
| `Odometry` | nav_msgs | Robot odometry |

### 4. Useful Commands

**ROS1:**
```bash
roscore                    # Start master
rosnode list              # List active nodes
rostopic list             # List topics
rostopic echo /topic      # View topic data
rostopic hz /topic        # Check publish rate
rosmsg show std_msgs/String  # Show message definition
roslaunch pkg launch.launch  # Launch file
```

**ROS2:**
```bash
ros2 node list            # List active nodes
ros2 topic list           # List topics
ros2 topic echo /topic    # View topic data
ros2 topic hz /topic      # Check publish rate
ros2 interface show std_msgs/msg/String  # Show message definition
ros2 launch pkg launch.py # Launch file
```

---

## C++ Examples

### ROS1 Publisher (C++)
```cpp
#include <ros/ros.h>
#include <std_msgs/String.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "publisher");
    ros::NodeHandle nh;

    ros::Publisher pub = nh.advertise<std_msgs::String>("chatter", 10);
    ros::Rate rate(10);

    while (ros::ok()) {
        std_msgs::String msg;
        msg.data = "Hello ROS1";
        pub.publish(msg);
        rate.sleep();
    }
    return 0;
}
```

### ROS2 Publisher (C++)
```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class Publisher : public rclcpp::Node {
public:
    Publisher() : Node("publisher") {
        pub_ = create_publisher<std_msgs::msg::String>("chatter", 10);
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&Publisher::timer_callback, this)
        );
    }

private:
    void timer_callback() {
        auto msg = std_msgs::msg::String();
        msg.data = "Hello ROS2";
        pub_->publish(msg);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Publisher>());
    rclcpp::shutdown();
    return 0;
}
```

---

## SLAM-Relevant Topics

| Topic Name | Message Type | Description |
|------------|--------------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Camera images |
| `/camera/camera_info` | sensor_msgs/CameraInfo | Camera intrinsics |
| `/velodyne_points` | sensor_msgs/PointCloud2 | LiDAR point cloud |
| `/imu/data` | sensor_msgs/Imu | IMU measurements |
| `/odom` | nav_msgs/Odometry | Wheel odometry |
| `/tf` | tf2_msgs/TFMessage | Transform tree |

---

## Recommended Learning Path

1. **Start with ROS2** if learning fresh (ROS1 is deprecated)
2. Practice with simple pub/sub examples
3. Learn tf2 for coordinate transforms
4. Explore sensor message types
5. Build a simple robot simulation

---

## References

- [ROS1 Wiki](http://wiki.ros.org/)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
