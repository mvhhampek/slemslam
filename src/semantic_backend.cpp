#include <memory>
#include <set>
#include <map>
#include <chrono>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/LossFunctions.h>
#include <gtsam/inference/Symbol.h>

using gtsam::symbol_shorthand::X; 
using gtsam::symbol_shorthand::L; 

class SemanticBackend : public rclcpp::Node
{
public:
  SemanticBackend() : Node("semantic_backend"), pose_idx_(0), first_odom_(true),
                      target_x_(0), target_y_(0), target_yaw_(0),
                      x_smooth_(0), y_smooth_(0), yaw_smooth_(0)
  {
    this->declare_parameter<std::string>("landmarks_topic", "/semantic_landmarks");
    this->declare_parameter<std::string>("optimized_path_topic", "/optimized_path");
    this->declare_parameter<std::string>("uncorrected_path_topic", "/uncorrected_path");
    this->declare_parameter<std::string>("optimized_map_topic", "/optimized_map");

    std::string landmarks_topic = this->get_parameter("landmarks_topic").as_string();
    std::string optimized_path_topic = this->get_parameter("optimized_path_topic").as_string();
    std::string uncorrected_path_topic = this->get_parameter("uncorrected_path_topic").as_string();
    std::string optimized_map_topic = this->get_parameter("optimized_map_topic").as_string();

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(parameters);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&SemanticBackend::update_and_publish, this));

    landmark_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        landmarks_topic, 10, std::bind(&SemanticBackend::landmark_callback, this, std::placeholders::_1));

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(optimized_path_topic, 10);
    uncorrected_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(uncorrected_path_topic, 10);
    optimized_map_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(optimized_map_topic, 10);
    
    optimized_path_.header.frame_id = "world";
    uncorrected_path_.header.frame_id = "world";
    RCLCPP_INFO(this->get_logger(), "Semantic Backend Started (landmarks: %s, path: %s)",
      landmarks_topic.c_str(), optimized_path_topic.c_str());
  }

private:
  void update_and_publish()
  {
    geometry_msgs::msg::TransformStamped t_m_b;
    try {
      t_m_b = tf_buffer_->lookupTransform("noisy_odom", "noisy_base_link", tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      return; 
    }

    tf2::Quaternion q_mb;
    tf2::fromMsg(t_m_b.transform.rotation, q_mb);
    tf2::Matrix3x3 m_mb(q_mb);
    double roll, pitch, yaw;
    m_mb.getRPY(roll, pitch, yaw);

    gtsam::Pose2 current_slam_pose(t_m_b.transform.translation.x, t_m_b.transform.translation.y, yaw);

    //dead reckoning path publishing (uncorrected
    publish_uncorrected_path(t_m_b.header.stamp, current_slam_pose);

    
    if (first_odom_) {
      gtsam::Vector3 prior_sigmas(0.01, 0.01, 0.01);
      auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
      
      graph_.addPrior(X(0), current_slam_pose, priorNoise);
      initial_estimates_.insert(X(0), current_slam_pose);
      
      prev_slam_pose_ = current_slam_pose;
      optimized_pose_ = current_slam_pose;
      target_x_ = current_slam_pose.x();
      target_y_ = current_slam_pose.y();
      target_yaw_ = current_slam_pose.theta();
      x_smooth_ = target_x_;
      y_smooth_ = target_y_;
      yaw_smooth_ = target_yaw_;
      pose_timestamps_.push_back(t_m_b.header.stamp);
      first_odom_ = false;
      return;
    }

    // keyframe insertion (>0.25m translation OR >0.10rad rotation)
    gtsam::Pose2 delta = prev_slam_pose_.between(current_slam_pose);
    bool is_keyframe = (delta.translation().norm() > 0.25 || std::abs(delta.theta()) > 0.10);

    if (is_keyframe) {
      pose_idx_++;
      pose_timestamps_.push_back(t_m_b.header.stamp);

      // odom noise model
      gtsam::Vector3 odom_sigmas(0.05, 0.05, 0.03);
      auto odomNoise = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);
      
      graph_.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2>>(X(pose_idx_ - 1), X(pose_idx_), delta, odomNoise);

      optimized_pose_ = optimized_pose_.compose(delta);
      initial_estimates_.insert(X(pose_idx_), optimized_pose_);
      prev_slam_pose_ = current_slam_pose;

      isam_->update(graph_, initial_estimates_);
      isam_->update();
      graph_.resize(0);
      initial_estimates_.clear();

      optimized_pose_ = isam_->calculateEstimate<gtsam::Pose2>(X(pose_idx_));
      
      publish_optimized_path(t_m_b.header.stamp);
      publish_optimized_map(t_m_b.header.stamp);
    }

    // T_world_odom = T_world_base * T_odom_base^-1
    gtsam::Pose2 t_world_map = optimized_pose_.compose(current_slam_pose.inverse());

    target_x_ = t_world_map.x();
    target_y_ = t_world_map.y();
    target_yaw_ = t_world_map.theta();

    x_smooth_ += 0.2 * (target_x_ - x_smooth_);
    y_smooth_ += 0.2 * (target_y_ - y_smooth_);
    double dyaw = atan2(sin(target_yaw_ - yaw_smooth_), cos(target_yaw_ - yaw_smooth_));
    yaw_smooth_ += 0.2 * dyaw;

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = this->now();
    t.header.frame_id = "world";
    t.child_frame_id = "noisy_odom";

    t.transform.translation.x = x_smooth_;
    t.transform.translation.y = y_smooth_;
    t.transform.translation.z = 0.0; 
    
    tf2::Quaternion q_out;
    q_out.setRPY(0.0, 0.0, yaw_smooth_);
    t.transform.rotation = tf2::toMsg(q_out);
    
    tf_broadcaster_->sendTransform(t);
  }

  void landmark_callback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
  {
    if (first_odom_) return;
    auto now = this->now();

    for (const auto& marker : msg->markers) {
      if (marker.ns.empty() || marker.ns == "optimized_landmarks" || marker.ns == "optimized_labels") continue;
      
      int l_id = marker.id;
      std::string l_class = marker.ns;

      gtsam::Point2 p_rel(marker.pose.position.x, marker.pose.position.y);
      gtsam::Rot2 bearing = gtsam::Rot2::atan2(p_rel.y(), p_rel.x());
      double range = p_rel.norm();
      
      // landmark measurement noise model
      auto baseNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(0.06, 0.20));
      auto robustNoise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(1.345), baseNoise);

      // if detector assigned a new id, check if an existing landmark of the same class is nearby in ISAM2
      if (seen_landmarks_.find(l_id) == seen_landmarks_.end()) {
        gtsam::Point2 cand_opt = optimized_pose_.transformFrom(p_rel);
        for (int e_id : seen_landmarks_) {
          if (landmark_class_map_.count(e_id) && landmark_class_map_[e_id] == l_class) {
            try {
              gtsam::Point2 e_opt = isam_->calculateEstimate<gtsam::Point2>(L(e_id));
              double dist = (e_opt - cand_opt).norm();
              int dk = pose_idx_ - landmark_last_keyframe_[e_id];
              if (dist < 2.5 && dk > 10) {
                RCLCPP_INFO(this->get_logger(), "Re-associated new ID %d (%s) -> existing landmark %d (dist=%.2fm, dk=%d)",
                  l_id, l_class.c_str(), e_id, dist, dk);
                l_id = e_id; // existing landmark ID
                break;
              }
            } catch (...) {}
          }
        }
      }

      landmark_class_map_[l_id] = l_class;

      int keyframe_diff = 9999;
      if (landmark_last_keyframe_.count(l_id)) {
        keyframe_diff = pose_idx_ - landmark_last_keyframe_[l_id];
      }

      if (seen_landmarks_.find(l_id) == seen_landmarks_.end()) {
        // new landmark discovery
        seen_landmarks_.insert(l_id);
        gtsam::Point2 l_opt = optimized_pose_.transformFrom(p_rel);
        initial_estimates_.insert(L(l_id), l_opt);
        
        graph_.emplace_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(X(pose_idx_), L(l_id), bearing, range, robustNoise);
        landmark_last_keyframe_[l_id] = pose_idx_;
        
        RCLCPP_INFO(this->get_logger(), "Landmark %d (%s) initialized at keyframe X_%d",
          l_id, l_class.c_str(), pose_idx_);
        
      } else if (keyframe_diff >= 1 && keyframe_diff <= 5) {
        // driving past a landmark, refine its position
        graph_.emplace_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(X(pose_idx_), L(l_id), bearing, range, robustNoise);
        landmark_last_keyframe_[l_id] = pose_idx_;

        RCLCPP_INFO(this->get_logger(), "Refined landmark %d (%s) at keyframe X_%d",
          l_id, l_class.c_str(), pose_idx_);
        
      } else if (keyframe_diff > 10) {
        // loop closure candidate, evaluate residuals
        bool accept_candidate = false;
        double b_diff = 0.0, r_diff = 0.0;
        try {
          gtsam::Point2 opt_pt = isam_->calculateEstimate<gtsam::Point2>(L(l_id));
          double dx = opt_pt.x() - optimized_pose_.x();
          double dy = opt_pt.y() - optimized_pose_.y();
          double pred_range = std::hypot(dx, dy);
          double pred_bearing = std::atan2(dy, dx) - optimized_pose_.theta();
          pred_bearing = std::atan2(std::sin(pred_bearing), std::cos(pred_bearing));

          double meas_bearing = bearing.theta();
          b_diff = std::fabs(std::atan2(std::sin(meas_bearing - pred_bearing), std::cos(meas_bearing - pred_bearing)));
          r_diff = std::fabs(range - pred_range);

          // probabilistic residual gate (bearing <= 0.35 rad approx 20 deg, range <= 4.0m)
          accept_candidate = (b_diff <= 0.35 && r_diff <= 4.0);
        } catch (...) {
          accept_candidate = true;
        }

        RCLCPP_INFO(this->get_logger(),
          "Landmark %d (%s): dk=%d keyframes, b_diff=%.3f rad, r_diff=%.2fm -> %s",
          l_id, l_class.c_str(), keyframe_diff, b_diff, r_diff, accept_candidate ? "accepted" : "rejected");

        landmark_last_keyframe_[l_id] = pose_idx_;

        if (accept_candidate) {
          // Huber loss (k=1.0) automatically clamps outliers
          auto loopClosureNoise = gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(1.0), baseNoise);

          graph_.emplace_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Point2>>(X(pose_idx_), L(l_id), bearing, range, loopClosureNoise);
          RCLCPP_INFO(this->get_logger(), "Correcting trajectory via Landmark %d (%s) at X_%d",
            l_id, l_class.c_str(), pose_idx_);

          // ISAM2 update iterations
          isam_->update(graph_, initial_estimates_);
          isam_->update();
          isam_->update();
          graph_.resize(0);
          initial_estimates_.clear();
          optimized_pose_ = isam_->calculateEstimate<gtsam::Pose2>(X(pose_idx_));
          publish_optimized_path(now);
          publish_optimized_map(now);
        }
      }
    }
  }

  void publish_uncorrected_path(rclcpp::Time stamp, const gtsam::Pose2& pose)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "world";
    pose_msg.pose.position.x = pose.x();
    pose_msg.pose.position.y = pose.y();
    pose_msg.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, pose.theta());
    pose_msg.pose.orientation = tf2::toMsg(q);

    uncorrected_path_.poses.push_back(pose_msg);
    uncorrected_path_pub_->publish(uncorrected_path_);
  }

  void publish_optimized_path(rclcpp::Time stamp)
  {
    optimized_path_.poses.clear();
    optimized_path_.header.stamp = stamp;
    
    gtsam::Values current_estimates = isam_->calculateEstimate();
    for (int i = 0; i <= pose_idx_; ++i) {
      if (current_estimates.exists(X(i))) {
        gtsam::Pose2 p = current_estimates.at<gtsam::Pose2>(X(i));
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = (i < (int)pose_timestamps_.size()) ? pose_timestamps_[i] : stamp;
        pose_msg.header.frame_id = "world";
        pose_msg.pose.position.x = p.x();
        pose_msg.pose.position.y = p.y();
        pose_msg.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, p.theta());
        pose_msg.pose.orientation = tf2::toMsg(q);
        optimized_path_.poses.push_back(pose_msg);
      }
    }
    path_pub_->publish(optimized_path_);
  }

  void publish_optimized_map(rclcpp::Time stamp)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    for (int l_id : seen_landmarks_) {
      try {
        gtsam::Point2 opt_pt = isam_->calculateEstimate<gtsam::Point2>(L(l_id));
        
        visualization_msgs::msg::Marker sphere;
        sphere.header.frame_id = "world";
        sphere.header.stamp = stamp;
        sphere.ns = "optimized_landmarks";
        sphere.id = l_id;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.pose.position.x = opt_pt.x();
        sphere.pose.position.y = opt_pt.y();
        sphere.pose.position.z = 0.05; 
        sphere.scale.x = 0.5; sphere.scale.y = 0.5; sphere.scale.z = 0.5;
        sphere.color.g = 1.0; sphere.color.a = 0.9;
        marker_array.markers.push_back(sphere);

        visualization_msgs::msg::Marker text;
        text.header = sphere.header;
        text.ns = "optimized_labels";
        text.id = l_id;
        text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text.pose.position = sphere.pose.position;
        text.pose.position.z += 0.5;
        text.scale.z = 0.4;
        text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 1.0;
        char buf[64];
        std::string cls = landmark_class_map_.count(l_id) ? landmark_class_map_[l_id] : "";
        snprintf(buf, sizeof(buf), "%s_%d", cls.c_str(), l_id);
        text.text = buf;
        marker_array.markers.push_back(text);
      } catch (...) {}
    }
    optimized_map_pub_->publish(marker_array);
  }

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimates_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;

  int pose_idx_;
  bool first_odom_;
  gtsam::Pose2 prev_slam_pose_;
  gtsam::Pose2 optimized_pose_;
  
  double target_x_, target_y_, target_yaw_;
  double x_smooth_, y_smooth_, yaw_smooth_;

  std::vector<rclcpp::Time> pose_timestamps_;
  std::set<int> seen_landmarks_;
  std::map<int, int> landmark_last_keyframe_;
  std::map<int, std::string> landmark_class_map_;

  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr uncorrected_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr optimized_map_pub_;
  
  nav_msgs::msg::Path optimized_path_;
  nav_msgs::msg::Path uncorrected_path_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticBackend>());
  rclcpp::shutdown();
  return 0;
}