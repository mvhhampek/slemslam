#include <memory>
#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/LinearMath/Vector3.h"
#include "tf2/LinearMath/Transform.h"

class LidarMerger : public rclcpp::Node {
public:
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::LaserScan, sensor_msgs::msg::LaserScan>
      SyncPolicy;

  LidarMerger() : Node("lidar_merger") {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 10), qos_profile);

    front_sub_.subscribe(this, "/tesse/front_lidar/scan", qos.get_rmw_qos_profile());
    rear_sub_.subscribe(this, "/tesse/rear_lidar/scan", qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), front_sub_, rear_sub_);
    sync_->registerCallback(std::bind(&LidarMerger::scan_callback, this,
                                      std::placeholders::_1, std::placeholders::_2));

    merged_pub_drifting_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan_drifting", 10);
    merged_pub_gt_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan_gt", 10);
    
    RCLCPP_INFO(this->get_logger(), "Lidar Merger started!");
  }

private:
  void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& front_msg,
                     const sensor_msgs::msg::LaserScan::ConstSharedPtr& rear_msg) {
    
    // Empty 360 scan
    sensor_msgs::msg::LaserScan empty_scan;
    empty_scan.header.stamp = front_msg->header.stamp;
    empty_scan.angle_min = -M_PI;
    empty_scan.angle_max = M_PI;
    empty_scan.angle_increment = front_msg->angle_increment;
    empty_scan.time_increment = front_msg->time_increment;
    empty_scan.scan_time = front_msg->scan_time;
    empty_scan.range_min = front_msg->range_min;
    empty_scan.range_max = front_msg->range_max;
    
    int num_readings = std::ceil((empty_scan.angle_max - empty_scan.angle_min) / empty_scan.angle_increment);
    empty_scan.ranges.assign(num_readings, std::numeric_limits<float>::infinity());

    // Ground truth scan
    sensor_msgs::msg::LaserScan gt_scan = empty_scan;
    gt_scan.header.frame_id = "base_link";

    // Drifting scan
    sensor_msgs::msg::LaserScan drifting_scan = empty_scan;
    drifting_scan.header.frame_id = "noisy_merged_lidar"; // Attach this frame to noisy_base_link via static TF

    // Merge scans into the two output scans
    process_scan(front_msg, gt_scan);
    process_scan(rear_msg, gt_scan);
    
    process_scan(front_msg, drifting_scan);
    process_scan(rear_msg, drifting_scan);

    merged_pub_gt_->publish(gt_scan);
    merged_pub_drifting_->publish(drifting_scan);
  }

  void process_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                    sensor_msgs::msg::LaserScan& merged_scan) {
    std::string target_frame = "base_link";

    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buffer_->lookupTransform(target_frame, scan_msg->header.frame_id, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      return;
    }

    tf2::Transform transform;
    tf2::fromMsg(t.transform, transform);

    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      float r = scan_msg->ranges[i];
      if (std::isinf(r) || std::isnan(r) || r < scan_msg->range_min || r > scan_msg->range_max) {
        continue;
      }

      float angle = scan_msg->angle_min + i * scan_msg->angle_increment;
      tf2::Vector3 point_local(r * std::cos(angle), r * std::sin(angle), 0.0);
      tf2::Vector3 point_base = transform * point_local;

      float merged_range = std::hypot(point_base.x(), point_base.y());
      float merged_angle = std::atan2(point_base.y(), point_base.x());

      if (merged_angle >= merged_scan.angle_min && merged_angle <= merged_scan.angle_max) {
        int merged_index = std::round((merged_angle - merged_scan.angle_min) /
                                      merged_scan.angle_increment);
        
        if (merged_index >= 0 && merged_index < (int)merged_scan.ranges.size()) {
          if (std::isinf(merged_scan.ranges[merged_index]) || merged_range < merged_scan.ranges[merged_index]) {
            merged_scan.ranges[merged_index] = merged_range;
          }
        }
      }
    }
  }

  message_filters::Subscriber<sensor_msgs::msg::LaserScan> front_sub_;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> rear_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr merged_pub_drifting_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr merged_pub_gt_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarMerger>());
  rclcpp::shutdown();
  return 0;
}