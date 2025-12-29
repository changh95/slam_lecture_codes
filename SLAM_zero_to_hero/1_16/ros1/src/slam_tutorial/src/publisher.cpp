/**
 * ROS1 C++ Publisher Example
 *
 * Publishes robot pose messages at 10 Hz.
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <cmath>

int main(int argc, char** argv) {
    // Initialize ROS
    ros::init(argc, argv, "pose_publisher_cpp");
    ros::NodeHandle nh;

    // Create publisher
    ros::Publisher pub = nh.advertise<geometry_msgs::PoseStamped>("/robot/pose", 10);

    // Set rate (10 Hz)
    ros::Rate rate(10);

    ROS_INFO("C++ Pose publisher started");

    double t = 0.0;

    while (ros::ok()) {
        // Create message
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "map";

        // Simulate circular motion
        msg.pose.position.x = std::cos(t * 0.1) * 5.0;
        msg.pose.position.y = std::sin(t * 0.1) * 5.0;
        msg.pose.position.z = 0.0;

        // Simple quaternion (yaw only)
        double yaw = t * 0.1;
        msg.pose.orientation.x = 0.0;
        msg.pose.orientation.y = 0.0;
        msg.pose.orientation.z = std::sin(yaw / 2);
        msg.pose.orientation.w = std::cos(yaw / 2);

        // Publish
        pub.publish(msg);

        if (static_cast<int>(t) % 10 == 0) {
            ROS_INFO("Published pose: x=%.2f, y=%.2f",
                     msg.pose.position.x, msg.pose.position.y);
        }

        t += 0.1;
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
