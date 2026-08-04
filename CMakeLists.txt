cmake_minimum_required(VERSION 3.8)
project(slemslam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(visualization_msgs REQUIRED)
# find_package(<dependency> REQUIRED)


add_executable(lidar_merger src/lidar_merger.cpp)
ament_target_dependencies(lidar_merger
  rclcpp
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
)


add_executable(odom_corruptor src/odom_corruptor.cpp)
ament_target_dependencies(odom_corruptor 
  rclcpp 
  nav_msgs 
  geometry_msgs 
  tf2 
  tf2_ros 
  tf2_geometry_msgs
)


add_executable(semantic_detector src/semantic_detector.cpp)
set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime)
target_include_directories(semantic_detector PRIVATE ${ONNXRUNTIME_DIR}/include)
ament_target_dependencies(semantic_detector 
rclcpp
sensor_msgs
visualization_msgs
cv_bridge
message_filters
tf2
tf2_ros
tf2_geometry_msgs
geometry_msgs
ament_index_cpp
)

add_executable(semantic_backend src/semantic_backend.cpp)
target_include_directories(semantic_backend PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(semantic_backend gtsam tbb)
target_link_directories(semantic_detector PRIVATE ${ONNXRUNTIME_DIR}/lib)
ament_target_dependencies(semantic_backend
rclcpp
nav_msgs
tf2_geometry_msgs
tf2_ros
visualization_msgs
Eigen3
)

target_link_libraries(semantic_backend gtsam)
target_link_directories(semantic_detector PRIVATE ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(semantic_detector onnxruntime ${OpenCV_LIBRARIES})

set_target_properties(semantic_detector PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib")




install(DIRECTORY
  config
  launch
  weights
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  lidar_merger
  odom_corruptor
  semantic_detector
  semantic_backend
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
