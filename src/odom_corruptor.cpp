#include <memory>
#include <random>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"

class OdomCorruptor : public rclcpp::Node
{
public:
  OdomCorruptor()
  : Node("odom_corruptor"), is_first_msg_(true)
  {
    this->declare_parameter<std::string>("input_odom_topic", "/tesse/odom");
    this->declare_parameter<std::string>("noisy_odom_topic", "/noisy_odom");
    this->declare_parameter<std::string>("gt_path_topic", "/gt_path");

    std::string input_odom_topic = this->get_parameter("input_odom_topic").as_string();
    std::string noisy_odom_topic = this->get_parameter("noisy_odom_topic").as_string();
    std::string gt_path_topic = this->get_parameter("gt_path_topic").as_string();

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      input_odom_topic, 10, std::bind(&OdomCorruptor::odom_callback, this, std::placeholders::_1));

    noisy_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(noisy_odom_topic, 10);
    gt_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(gt_path_topic, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    gt_path_.header.frame_id = "world";

    yaw_noise_gen_ = std::normal_distribution<double>(0.0, 0.0002); 
    trans_noise_gen_ = std::normal_distribution<double>(0.0, 0.001); 

    RCLCPP_INFO(this->get_logger(), "Odometry Corruptor started (sub: %s, pub: %s, gt: %s)",
      input_odom_topic.c_str(), noisy_odom_topic.c_str(), gt_path_topic.c_str());
  }

private:
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (is_first_msg_) {
      last_gt_odom_ = *msg;
      noisy_x_ = msg->pose.pose.position.x;
      noisy_y_ = msg->pose.pose.position.y;
      noisy_yaw_ = get_yaw_from_quaternion(msg->pose.pose.orientation);
      start_time_ = msg->header.stamp;
      is_first_msg_ = false;
      return;
    }

    double current_gt_yaw = get_yaw_from_quaternion(msg->pose.pose.orientation);
    double last_gt_yaw = get_yaw_from_quaternion(last_gt_odom_.pose.pose.orientation);

    double dx_gt = msg->pose.pose.position.x - last_gt_odom_.pose.pose.position.x;
    double dy_gt = msg->pose.pose.position.y - last_gt_odom_.pose.pose.position.y;

    double dx_local = dx_gt * std::cos(last_gt_yaw) + dy_gt * std::sin(last_gt_yaw);
    double dy_local = -dx_gt * std::sin(last_gt_yaw) + dy_gt * std::cos(last_gt_yaw);

    double d_yaw = current_gt_yaw - last_gt_yaw;
    d_yaw = std::atan2(std::sin(d_yaw), std::cos(d_yaw));

    double current_sec = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    double start_sec = start_time_.sec + start_time_.nanosec * 1e-9;
    bool past_grace_period = (current_sec - start_sec) > 10.0;

    if (past_grace_period && (std::hypot(dx_local, dy_local) > 0.001 || std::abs(d_yaw) > 0.001)) {
      //0.5% scale error, approx 0.15 deg/s heading drift
      dx_local *= 1.005;
      d_yaw += 0.00005;

      dx_local += trans_noise_gen_(generator_); 
      d_yaw += yaw_noise_gen_(generator_);      
    }

    noisy_yaw_ += d_yaw;
    noisy_yaw_ = std::atan2(std::sin(noisy_yaw_), std::cos(noisy_yaw_)); 

    noisy_x_ += dx_local * std::cos(noisy_yaw_) - dy_local * std::sin(noisy_yaw_);
    noisy_y_ += dx_local * std::sin(noisy_yaw_) + dy_local * std::cos(noisy_yaw_);

    nav_msgs::msg::Odometry noisy_msg = *msg; 
    noisy_msg.header.frame_id = "noisy_odom"; 
    noisy_msg.child_frame_id = "noisy_base_link";
    
    noisy_msg.pose.pose.position.x = noisy_x_;
    noisy_msg.pose.pose.position.y = noisy_y_;
    
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, noisy_yaw_);
    noisy_msg.pose.pose.orientation = tf2::toMsg(q);

    noisy_odom_pub_->publish(noisy_msg);

    // noisy odom TF 
    geometry_msgs::msg::TransformStamped t_noisy;
    t_noisy.header.stamp = msg->header.stamp;
    t_noisy.header.frame_id = "noisy_odom";
    t_noisy.child_frame_id = "noisy_base_link";
    t_noisy.transform.translation.x = noisy_x_;
    t_noisy.transform.translation.y = noisy_y_;
    t_noisy.transform.translation.z = 0.0;
    t_noisy.transform.rotation = noisy_msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(t_noisy);

    // ground truth odom TF
    geometry_msgs::msg::TransformStamped t_gt;
    t_gt.header.stamp = msg->header.stamp;
    t_gt.header.frame_id = "world";
    t_gt.child_frame_id = "base_link";
    t_gt.transform.translation.x = msg->pose.pose.position.x;
    t_gt.transform.translation.y = msg->pose.pose.position.y;
    t_gt.transform.translation.z = 0.0;
    t_gt.transform.rotation = msg->pose.pose.orientation;
    tf_broadcaster_->sendTransform(t_gt);

    // ground truth path
    geometry_msgs::msg::PoseStamped gt_pose;
    gt_pose.header = msg->header;
    gt_pose.header.frame_id = "world";
    gt_pose.pose = msg->pose.pose;
    gt_pose.pose.position.z = 0.0;

    gt_path_.header.stamp = msg->header.stamp;
    gt_path_.header.frame_id = "world";
    gt_path_.poses.push_back(gt_pose);
    gt_path_pub_->publish(gt_path_);

    last_gt_odom_ = *msg;
  }

  double get_yaw_from_quaternion(const geometry_msgs::msg::Quaternion& q_msg) const
  {
    tf2::Quaternion q;
    tf2::fromMsg(q_msg, q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    return yaw;
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr noisy_odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr gt_path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  nav_msgs::msg::Path gt_path_;
  nav_msgs::msg::Odometry last_gt_odom_;
  bool is_first_msg_;
  builtin_interfaces::msg::Time start_time_;

  double noisy_x_;
  double noisy_y_;
  double noisy_yaw_;

  std::default_random_engine generator_;
  std::normal_distribution<double> yaw_noise_gen_;
  std::normal_distribution<double> trans_noise_gen_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdomCorruptor>());
  rclcpp::shutdown();
  return 0;
}