/**
 * ROS2 C++ Publisher Example
 *
 * Publishes robot pose messages at 10 Hz.
 */

#include <chrono>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

using namespace std::chrono_literals;

class PosePublisher : public rclcpp::Node {
public:
    PosePublisher() : Node("pose_publisher_cpp"), t_(0.0) {
        // Create publisher
        publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/robot/pose", 10);

        // Create timer (10 Hz)
        timer_ = this->create_wall_timer(
            100ms, std::bind(&PosePublisher::timer_callback, this));

        RCLCPP_INFO(this->get_logger(), "C++ Pose publisher started");
    }

private:
    void timer_callback() {
        // Create message
        auto msg = geometry_msgs::msg::PoseStamped();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";

        // Simulate circular motion
        msg.pose.position.x = std::cos(t_ * 0.1) * 5.0;
        msg.pose.position.y = std::sin(t_ * 0.1) * 5.0;
        msg.pose.position.z = 0.0;

        // Simple quaternion (yaw only)
        double yaw = t_ * 0.1;
        msg.pose.orientation.x = 0.0;
        msg.pose.orientation.y = 0.0;
        msg.pose.orientation.z = std::sin(yaw / 2);
        msg.pose.orientation.w = std::cos(yaw / 2);

        // Publish
        publisher_->publish(msg);

        if (static_cast<int>(t_) % 10 == 0) {
            RCLCPP_INFO(this->get_logger(), "Published pose: x=%.2f, y=%.2f",
                        msg.pose.position.x, msg.pose.position.y);
        }

        t_ += 0.1;
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    double t_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PosePublisher>());
    rclcpp::shutdown();
    return 0;
}
