#include <cmath>
#include <memory>
#include <random>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

class OdomCorruptor : public rclcpp::Node {
public:
  OdomCorruptor() : Node("odom_corruptor"), is_first_msg_(true) {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/tesse/odom", 10,
        std::bind(&OdomCorruptor::odom_callback, this, std::placeholders::_1));

    noisy_odom_pub_ =
        this->create_publisher<nav_msgs::msg::Odometry>("/noisy_odom", 10);

    // slam toolbox wants tf
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // mean, std
    yaw_noise_gen_ = std::normal_distribution<double>(0.0, 0.005);
    trans_noise_gen_ = std::normal_distribution<double>(0.0, 0.01);

    RCLCPP_INFO(this->get_logger(), "Odometry Corruptor started! :D");
  }

private:
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    RCLCPP_INFO_ONCE(this->get_logger(), "Recieved first odom msg");
    if (is_first_msg_) {
      last_gt_odom_ = *msg;
      noisy_x_ = msg->pose.pose.position.x;
      noisy_y_ = msg->pose.pose.position.y;
      noisy_yaw_ = quat_to_yaw(msg->pose.pose.orientation);
      is_first_msg_ = false;
      return;
    }

    // gt movement
    double current_gt_yaw = quat_to_yaw(msg->pose.pose.orientation);
    double last_gt_yaw = quat_to_yaw(last_gt_odom_.pose.pose.orientation);

    double dx_gt =
        msg->pose.pose.position.x - last_gt_odom_.pose.pose.position.x;
    double dy_gt =
        msg->pose.pose.position.y - last_gt_odom_.pose.pose.position.y;

    double d_trans = std::hypot(dx_gt, dy_gt);
    double d_yaw = current_gt_yaw - last_gt_yaw;
    d_yaw = std::atan2(std::sin(d_yaw), std::cos(d_yaw));

    // noise if moving
    if (d_trans > 0.001 || std::abs(d_yaw) > 0.001) {
      d_trans += trans_noise_gen_(generator_);
      d_yaw += yaw_noise_gen_(generator_);
    }

    // integrate
    noisy_yaw_ += d_yaw;
    noisy_yaw_ =
        std::atan2(std::sin(noisy_yaw_), std::cos(noisy_yaw_)); // normalize

    noisy_x_ += d_trans * std::cos(noisy_yaw_);
    noisy_y_ += d_trans * std::sin(noisy_yaw_);

    // publish noisy odom
    nav_msgs::msg::Odometry noisy_msg = *msg; // keep og time, twist
    noisy_msg.header.frame_id = "noisy_odom";
    noisy_msg.child_frame_id = "noisy_base_link";

    noisy_msg.pose.pose.position.x = noisy_x_;
    noisy_msg.pose.pose.position.y = noisy_y_;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, noisy_yaw_);
    noisy_msg.pose.pose.orientation = tf2::toMsg(q);

    noisy_odom_pub_->publish(noisy_msg);

    // broadcast noisy tf
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = msg->header.stamp;
    t.header.frame_id = "noisy_odom";
    t.child_frame_id = "noisy_base_link";
    t.transform.translation.x = noisy_x_;
    t.transform.translation.y = noisy_y_;
    t.transform.translation.z = msg->pose.pose.position.z;
    t.transform.rotation = noisy_msg.pose.pose.orientation;

    tf_broadcaster_->sendTransform(t);


    // save state for next call
    last_gt_odom_ = *msg;
  }

  double quat_to_yaw(const geometry_msgs::msg::Quaternion& q_msg) const {
    tf2::Quaternion q;
    tf2::fromMsg(q_msg, q);
    double r,p,y;
    tf2::Matrix3x3(q).getRPY(r,p,y);
    return y;
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr noisy_odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  nav_msgs::msg::Odometry last_gt_odom_;
  bool is_first_msg_;
  double noisy_x_;
  double noisy_y_;
  double noisy_yaw_;

  std::default_random_engine generator_;
  std::normal_distribution<double> yaw_noise_gen_;
  std::normal_distribution<double> trans_noise_gen_;
};


int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<OdomCorruptor>());
  rclcpp::shutdown();
  return 0;
}