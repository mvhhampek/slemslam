#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Path

class ATEEvaluator(Node):
    def __init__(self):
        super().__init__('ate_evaluator')

        self.gt_path = []
        self.uncorrected_path = []
        self.optimized_path = []

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.create_subscription(Path, '/gt_path', self.gt_callback, qos_profile=qos)
        self.create_subscription(Path, '/uncorrected_path', self.uncorrected_callback, qos_profile=qos)
        self.create_subscription(Path, '/optimized_path', self.optimized_callback, qos_profile=qos)

        self.get_logger().info("ATE Evaluator started...")

    def gt_callback(self, msg):
        self.gt_path = [(p.header.stamp.sec + p.header.stamp.nanosec * 1e-9, p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def uncorrected_callback(self, msg):
        self.uncorrected_path = [(p.header.stamp.sec + p.header.stamp.nanosec * 1e-9, p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def optimized_callback(self, msg):
        self.optimized_path = [(p.header.stamp.sec + p.header.stamp.nanosec * 1e-9, p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def evaluate(self):
        if not self.gt_path or not self.uncorrected_path or not self.optimized_path:
            print("\n Incomplete path data received. One path topic missing (/gt_path, /uncorrected_path, /optimized_path)")
            return

        gt_arr = np.array(self.gt_path)
        uncorr_arr = np.array(self.uncorrected_path)
        opt_arr = np.array(self.optimized_path)

        errors_uncorr = compute_trajectory_errors(gt_arr, uncorr_arr)
        errors_opt = compute_trajectory_errors(gt_arr, opt_arr)

        rmse_uncorr = np.sqrt(np.mean(errors_uncorr**2)) if len(errors_uncorr) > 0 else 0.0
        max_uncorr = np.max(errors_uncorr) if len(errors_uncorr) > 0 else 0.0
        mean_uncorr = np.mean(errors_uncorr) if len(errors_uncorr) > 0 else 0.0

        rmse_opt = np.sqrt(np.mean(errors_opt**2)) if len(errors_opt) > 0 else 0.0
        max_opt = np.max(errors_opt) if len(errors_opt) > 0 else 0.0
        mean_opt = np.mean(errors_opt) if len(errors_opt) > 0 else 0.0

        improvement = ((rmse_uncorr - rmse_opt) / rmse_uncorr * 100.0) if rmse_uncorr > 0 else 0.0

        print("\nATE Results:")
        print(f"Dead-Reckoning Odom       : RMSE = {rmse_uncorr:.4f} m, Mean = {mean_uncorr:.4f} m, Max = {max_uncorr:.4f} m")
        print(f"GTSAM Semantic SLAM       : RMSE = {rmse_opt:.4f} m, Mean = {mean_opt:.4f} m, Max = {max_opt:.4f} m")
        print(f"Trajectory Error Reduction: {improvement:.2f}%\n")

def compute_trajectory_errors(gt_arr, est_arr):
    if len(gt_arr) == 0 or len(est_arr) == 0:
        return np.array([0.0])

    unique_timestamps = len(np.unique(est_arr[:, 0]))
    use_timestamps = unique_timestamps > 0.8 * len(est_arr)

    errors = []
    if use_timestamps:
        for t_est, x_est, y_est in est_arr:
            time_diffs = np.abs(gt_arr[:, 0] - t_est)
            min_idx = np.argmin(time_diffs)
            if time_diffs[min_idx] < 1.0:
                x_gt, y_gt = gt_arr[min_idx, 1], gt_arr[min_idx, 2]
                err = np.hypot(x_est - x_gt, y_est - y_gt)
                errors.append(err)

    if not errors:
        N_est = len(est_arr)
        N_gt = len(gt_arr)
        for i, (t_est, x_est, y_est) in enumerate(est_arr):
            gt_idx = int(round(i * (N_gt - 1) / max(1, N_est - 1)))
            gt_idx = min(max(0, gt_idx), N_gt - 1)
            x_gt, y_gt = gt_arr[gt_idx, 1], gt_arr[gt_idx, 2]
            err = np.hypot(x_est - x_gt, y_est - y_gt)
            errors.append(err)

    return np.array(errors) if errors else np.array([0.0])

def evaluate_bag_file(bag_path):
    print(f"\nParsing bag file: {bag_path}...")
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        storage_options.storage_id = "mcap"
        reader.open(storage_options, converter_options)

    gt_path = []
    uncorrected_path = []
    optimized_path = []

    while reader.has_next():
        (topic, data, stamp) = reader.read_next()
        if topic == '/gt_path':
            msg = deserialize_message(data, Path)
            gt_path = [(p.header.stamp.sec + p.header.stamp.nanosec * 1e-9, p.pose.position.x, p.pose.position.y) for p in msg.poses]
        elif topic == '/uncorrected_path':
            msg = deserialize_message(data, Path)
            uncorrected_path = [(p.header.stamp.sec + p.header.stamp.nanosec * 1e-9, p.pose.position.x, p.pose.position.y) for p in msg.poses]
        elif topic == '/optimized_path':
            msg = deserialize_message(data, Path)
            optimized_path = [(p.header.stamp.sec + p.header.stamp.nanosec * 1e-9, p.pose.position.x, p.pose.position.y) for p in msg.poses]

    evaluator = ATEEvaluator()
    evaluator.gt_path = gt_path
    evaluator.uncorrected_path = uncorrected_path
    evaluator.optimized_path = optimized_path
    evaluator.evaluate()

def main():
    parser = argparse.ArgumentParser(description="ATE Trajectory Evaluator")
    parser.add_argument("--bag", "-b", type=str, help="rosbag path", default=None)
    args = parser.parse_args()

    if args.bag:
        rclpy.init()
        evaluate_bag_file(args.bag)
        if rclpy.ok():
            rclpy.shutdown()
        return

    rclpy.init()
    evaluator = ATEEvaluator()
    try:
        rclpy.spin(evaluator)
    except BaseException:
        pass
    finally:
        evaluator.evaluate()
        try:
            evaluator.destroy_node()
        except BaseException:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except BaseException:
                pass

if __name__ == '__main__':
    main()
