#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include <cmath>
#include <limits>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class LidarMerger : public rclcpp::Node {
public:
  LidarMerger() : Node("lidar_merger") {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    merged_pub_ =
        this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);

    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 10),
                           qos_profile);

    front_sub_.subscribe(this, "/tesse/front_lidar/scan",
                         qos.get_rmw_qos_profile());
    rear_sub_.subscribe(this, "/tesse/rear_lidar/scan",
                        qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), front_sub_, rear_sub_);
    sync_->registerCallback(std::bind(&LidarMerger::sync_callback, this,
                                      std::placeholders::_1,
                                      std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Lidar Merger started! :D");
  }

private:
  void
  sync_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr &front_msg,
                const sensor_msgs::msg::LaserScan::ConstSharedPtr &rear_msg) {
    sensor_msgs::msg::LaserScan merged_scan;

    merged_scan.header.stamp = front_msg->header.stamp;
    merged_scan.header.frame_id = "base_link_gt"; // "base_link";
 
    merged_scan.angle_min = -M_PI;
    merged_scan.angle_max = M_PI;
    merged_scan.angle_increment = front_msg->angle_increment;
    merged_scan.time_increment = 0.0;
    merged_scan.scan_time = front_msg->scan_time;
    merged_scan.range_min = front_msg->range_min;
    merged_scan.range_max = front_msg->range_max;

    size_t array_size = std::ceil((merged_scan.angle_max - merged_scan.angle_min) / merged_scan.angle_increment);
    merged_scan.ranges.assign(array_size, std::numeric_limits<float>::infinity());

    process_scan(front_msg, merged_scan);
    process_scan(rear_msg, merged_scan);

    merged_pub_->publish(merged_scan);
  }

  void
  process_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
               sensor_msgs::msg::LaserScan& merged_scan) {
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
      transform_stamped = tf_buffer_->lookupTransform(
          merged_scan.header.frame_id, scan_msg->header.frame_id,
          tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_SKIPFIRST(this->get_logger(), "TF Error '%s'", ex.what());
      return;
    }
    tf2::Transform transform;
    tf2::fromMsg(transform_stamped.transform, transform);

    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      float r = scan_msg->ranges[i];

      if (std::isinf(r) ||
          std::isnan(r) || r<scan_msg->range_min || r> scan_msg->range_max) {
        continue;
      }

      float angle = scan_msg->angle_min + i * scan_msg->angle_increment;

      tf2::Vector3 point_local(r * std::cos(angle), r * std::sin(angle), 0.0);
      tf2::Vector3 point_base = transform * point_local;

      float merged_range = std::hypot(point_base.x(), point_base.y());
      float merged_angle = std::atan2(point_base.y(), point_base.x());

      int merged_index = std::round((merged_angle - merged_scan.angle_min) /
                                    merged_scan.angle_increment);

      if (merged_index >= 0 && merged_index < (int)merged_scan.ranges.size()) {
        if (merged_range < merged_scan.ranges[merged_index]) {
          merged_scan.ranges[merged_index] = merged_range;
        }
      }
    }
    // RCLCPP_INFO(this->get_logger(), "scan processed...");
  }

  message_filters::Subscriber<sensor_msgs::msg::LaserScan> front_sub_;
  message_filters::Subscriber<sensor_msgs::msg::LaserScan> rear_sub_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::LaserScan, sensor_msgs::msg::LaserScan>
      SyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr merged_pub_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<LidarMerger>());
  rclcpp::shutdown();
  return 0;
}