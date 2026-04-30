/**
 * ROS2 C++ Subscriber Example
 *
 * Subscribes to robot pose messages and tracks distance traveled.
 */

#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

class PoseSubscriber : public rclcpp::Node {
public:
    PoseSubscriber()
        : Node("pose_subscriber_cpp"),
          last_x_(0.0), last_y_(0.0),
          total_distance_(0.0),
          first_msg_(true),
          msg_count_(0) {

        // Create subscriber
        subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/robot/pose", 10,
            std::bind(&PoseSubscriber::callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "C++ Pose subscriber started");
    }

private:
    void callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
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
        msg_count_++;

        // Log periodically
        if (msg_count_ % 50 == 0) {
            RCLCPP_INFO(this->get_logger(),
                "Received pose: (%.2f, %.2f), total distance: %.2fm",
                x, y, total_distance_);
        }
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
    double last_x_, last_y_;
    double total_distance_;
    bool first_msg_;
    int msg_count_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseSubscriber>());
    rclcpp::shutdown();
    return 0;
}
