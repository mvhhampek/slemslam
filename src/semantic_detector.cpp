#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/marker_array.hpp"


#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <onnxruntime_cxx_api.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

class SemanticDetector : public rclcpp::Node {
public:
  struct Landmark {
    int id;
    std::string label;
    geometry_msgs::msg::Point position;
    int obs_count;
  };

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>
      SyncPolicy;

  SemanticDetector() : Node("semantic_detector"), has_intrinsics_(false) {
    this->declare_parameter<std::string>("rgb_topic", "/tesse/left_cam/rgb/image_raw");
    this->declare_parameter<std::string>("depth_topic", "/tesse/depth_cam/mono/image_raw");
    this->declare_parameter<std::string>("camera_info_topic", "/tesse/left_cam/camera_info");
    this->declare_parameter<std::string>("landmarks_topic", "/semantic_landmarks");

    std::string rgb_topic = this->get_parameter("rgb_topic").as_string();
    std::string depth_topic = this->get_parameter("depth_topic").as_string();
    std::string camera_info_topic = this->get_parameter("camera_info_topic").as_string();
    std::string landmarks_topic = this->get_parameter("landmarks_topic").as_string();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        landmarks_topic, 10);

    std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("slemslam");
    std::string model_path = pkg_share_dir + "/weights/yolov8n.onnx";

    try {
      env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(2);
      session_ = std::make_shared<Ort::Session>(*env_, model_path.c_str(), session_options);
      RCLCPP_INFO(this->get_logger(), "ONNX Runtime: YOLOv8 model loaded successfully");
    } catch (const Ort::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load ONNX model: %s", e.what());
    }

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, 10,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          if (!has_intrinsics_) {
            fx_ = msg->k[0];
            cx_ = msg->k[2];
            fy_ = msg->k[4];
            cy_ = msg->k[5];
            has_intrinsics_ = true;
          }
        });

    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 10), qos_profile);

    rgb_sub_.subscribe(this, rgb_topic, qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, depth_topic, qos.get_rmw_qos_profile());

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), rgb_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&SemanticDetector::image_callback, this,
                                      std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Semantic Detector started (rgb: %s, depth: %s, info: %s, landmarks: %s)",
      rgb_topic.c_str(), depth_topic.c_str(), camera_info_topic.c_str(), landmarks_topic.c_str());
  }

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_msg,
                 const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg) {
    if (!has_intrinsics_ || !session_) return;

    cv_bridge::CvImagePtr cv_rgb, cv_depth;
    try {
      cv_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
      cv_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception &e) {
      return;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(cv_rgb->image, blob, 1.0 / 255.0, cv::Size(640, 640),
                           cv::Scalar(), true, false);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* out_data = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat out_mat(84, 8400, CV_32F, out_data);
    cv::Mat predictions = out_mat.t(); 

    float x_factor = cv_rgb->image.cols / 640.0;
    float y_factor = cv_rgb->image.rows / 640.0;

    visualization_msgs::msg::MarkerArray current_obs;

    for (int i = 0; i < predictions.rows; ++i) {
      float *data = (float *)predictions.row(i).data;

      cv::Mat class_scores(1, 80, CV_32F, &data[4]);
      cv::Point class_id_pt;
      double max_score;
      cv::minMaxLoc(class_scores, nullptr, &max_score, nullptr, &class_id_pt);

      if (max_score > 0.6) {
        int best_class = class_id_pt.x;
        std::string class_name = coco_classes_[best_class];

        if (class_name == "person" || class_name == "car" || class_name == "dog" ||
            class_name == "cat" || class_name == "bird" || class_name == "cell phone") {
            continue; 
        }

        float cx = data[0] * x_factor;
        float cy = data[1] * y_factor;
        int u = std::round(cx);
        int v = std::round(cy);

        if (u < 0 || u >= cv_depth->image.cols || v < 0 || v >= cv_depth->image.rows) continue;

        float z = cv_depth->image.at<float>(v, u);
        if (std::isnan(z) || z < 0.5 || z > 15.0) continue;

        float x = (u - cx_) * z / fx_;
        float y = (v - cy_) * z / fy_;

        geometry_msgs::msg::PointStamped pt_cam, pt_world, pt_base;
        pt_cam.header.frame_id = rgb_msg->header.frame_id;
        pt_cam.header.stamp = rgb_msg->header.stamp; 
        pt_cam.point.x = x;
        pt_cam.point.y = y;
        pt_cam.point.z = z;

        std::string target_frame = "base_link";

        try {
          auto t_cam_to_base = tf_buffer_->lookupTransform(target_frame, pt_cam.header.frame_id, tf2::TimePointZero);
          tf2::doTransform(pt_cam, pt_base, t_cam_to_base);
          
          pt_base.header.frame_id = "noisy_base_link";

          auto t_base_to_world = tf_buffer_->lookupTransform("world", target_frame, tf2::TimePointZero);
          tf2::doTransform(pt_base, pt_world, t_base_to_world);
        } catch (const tf2::TransformException &ex) {
          continue;
        }

        int l_id = associate_landmark(pt_world.point, class_name);

        visualization_msgs::msg::Marker obs;
        obs.header.frame_id = "noisy_base_link";
        obs.header.stamp = rgb_msg->header.stamp;
        obs.ns = class_name;
        obs.id = l_id;
        obs.type = visualization_msgs::msg::Marker::SPHERE;
        obs.action = visualization_msgs::msg::Marker::ADD;
        
        obs.pose.position.x = pt_base.point.x;
        obs.pose.position.y = pt_base.point.y;
        obs.pose.position.z = 0.5;

        obs.pose.orientation.w = 1.0;
        obs.scale.x = 0.3; obs.scale.y = 0.3; obs.scale.z = 0.3;
        obs.color.r = 1.0; obs.color.g = 0.0; obs.color.b = 1.0; obs.color.a = 1.0;
        obs.lifetime = rclcpp::Duration::from_seconds(0.5); 

        current_obs.markers.push_back(obs);
      }
    }

    if (!current_obs.markers.empty()) {
      marker_pub_->publish(current_obs);
    }
  }

  int associate_landmark(const geometry_msgs::msg::Point& pt, const std::string& label) {
    double min_dist = 1.2; // 1.2m radius to prevent merging objects into the same landmark ID
    int best_idx = -1;

    for (size_t i = 0; i < landmark_db_.size(); ++i) {
      if (landmark_db_[i].label == label) {
        // 2d distance (xy)!
        double dx = landmark_db_[i].position.x - pt.x;
        double dy = landmark_db_[i].position.y - pt.y;
        double dist = std::hypot(dx, dy);

        if (dist < min_dist) {
          min_dist = dist;
          best_idx = i;
        }
      }
    }

    if (best_idx != -1) {
      landmark_db_[best_idx].obs_count++;
      // refine position with moving average
      landmark_db_[best_idx].position.x += 0.1 * (pt.x - landmark_db_[best_idx].position.x);
      landmark_db_[best_idx].position.y += 0.1 * (pt.y - landmark_db_[best_idx].position.y);
      return landmark_db_[best_idx].id;
    } else {
      Landmark lm;
      lm.id = next_landmark_id_++;
      lm.label = label;
      lm.position = pt;
      lm.obs_count = 1;
      landmark_db_.push_back(lm);
      RCLCPP_INFO(this->get_logger(), "New landmark discovered: %s_%d", label.c_str(), lm.id);
      return lm.id;
    }
  }

  std::vector<Landmark> landmark_db_;
  int next_landmark_id_ = 0;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::shared_ptr<Ort::Env> env_;
  std::shared_ptr<Ort::Session> session_;

  bool has_intrinsics_;
  float fx_, fy_, cx_, cy_;

  const std::vector<std::string> coco_classes_ = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
  };
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticDetector>());
  rclcpp::shutdown();
  return 0;
}