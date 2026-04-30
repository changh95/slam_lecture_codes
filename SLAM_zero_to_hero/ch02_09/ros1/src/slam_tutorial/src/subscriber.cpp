/**
 * ROS1 C++ Subscriber Example
 *
 * Subscribes to robot pose messages and tracks distance traveled.
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <cmath>

class PoseSubscriber {
public:
    PoseSubscriber() : last_x_(0.0), last_y_(0.0), total_distance_(0.0), first_msg_(true) {
        // Subscribe to pose topic
        sub_ = nh_.subscribe("/robot/pose", 10, &PoseSubscriber::callback, this);
        ROS_INFO("C++ Pose subscriber started");
    }

    void callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        double x = msg->pose.position.x;
        double y = msg->pose.position.y;

        // Calculate distance traveled
        if (!first_msg_) {
            double dx = x - last_x_;
            double dy = y - last_y_;
            double distance = std::sqrt(dx * dx + dy * dy);
            total_distance_ += distance;
        }
        first_msg_ = false;

        last_x_ = x;
        last_y_ = y;

        // Log periodically
        static int count = 0;
        if (++count % 50 == 0) {
            ROS_INFO("Received pose: (%.2f, %.2f), total distance: %.2fm",
                     x, y, total_distance_);
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    double last_x_, last_y_;
    double total_distance_;
    bool first_msg_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pose_subscriber_cpp");

    PoseSubscriber subscriber;
    ros::spin();

    return 0;
}
