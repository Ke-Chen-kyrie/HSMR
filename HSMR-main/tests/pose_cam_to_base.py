"""将 FoundationPose 输出的相机系位姿转换到机器人 base 系。

通过 tf2_ros 查询完整的 camera→BASE 变换链 (包含静态手眼标定和动态关节变换)。
静态手眼标定由 common 容器的 static_camera_tf.py 发布到 TF 树。

对旋转对称物体 (如碗), 可通过 fixed_axis 参数消除指定轴的自转分量。

Usage:
    python3 pose_cam_to_base.py \
        --input /fdpose/pose \
        --output /fdpose/pose_base \
        --namespace fdpose \
        --camera-frame realsense_head_color_optical_frame \
        --base-frame BASE
"""

import argparse

import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
import tf2_ros
from vision_msgs.msg import Detection3DArray

from erban_utils.monitor_logger import MonitorLogger
from erban_utils.tf import lookup_transform_safe


class PoseCamToBaseNode(Node):
    """通过 tf2_ros 将 Detection3DArray 位姿从相机系转换到 base 系。"""

    def __init__(self, input_topic, output_topic, namespace, camera_frame, base_frame):
        super().__init__("pose_cam_to_base", namespace=namespace)
        self._camera_frame = camera_frame
        self._base_frame = base_frame

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ROS2 parameter: 'x', 'y', or 'z' to remove rotation around that axis; '' to disable
        self.declare_parameter("fixed_axis", "")
        self._fixed_axis = self.get_parameter("fixed_axis").get_parameter_value().string_value
        self.add_on_set_parameters_callback(self._on_parameter_change)

        self._publisher = self.create_publisher(Detection3DArray, output_topic, 10)
        # 位姿转换状态: 向 object_manager 同步 TF 失败等原因, 避免服务空等超时
        self._status_publisher = self.create_publisher(String, output_topic + "_status", 10)
        self.create_subscription(Detection3DArray, input_topic, self._callback, 10)

        self._monitor = MonitorLogger("pose_cam_to_base")
        self._monitor.info(
            "node_started",
            camera_frame=camera_frame,
            base_frame=base_frame,
            fixed_axis=self._fixed_axis,
        )

        self.get_logger().info(
            f"cam->base: ({camera_frame})->({base_frame}) via tf2_ros (fixed_axis={self._fixed_axis!r})"
        )

    def _on_parameter_change(self, params):
        for param in params:
            if param.name == "fixed_axis":
                val = param.value
                if val not in ("", "x", "y", "z"):
                    self.get_logger().warning(f"Invalid fixed_axis value: {val!r}, ignoring")
                    return SetParametersResult(successful=False)
                self._fixed_axis = val
                self.get_logger().info(f"fixed_axis updated to: {val!r}")
        return SetParametersResult(successful=True)

    def _get_cam2base(self, stamp):
        """通过 tf2_ros 查询 camera→base 变换, 返回 (4x4矩阵, None) 或 (None, reason)。"""
        tf_stamped, tf_error = lookup_transform_safe(
            self._tf_buffer, self._base_frame, self._camera_frame, stamp, self.get_logger()
        )
        if tf_stamped is None:
            return None, tf_error

        t = tf_stamped.transform.translation
        r = tf_stamped.transform.rotation
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat, None

    def _callback(self, msg):
        T_cam2base, tf_error = self._get_cam2base(msg.header.stamp)
        if T_cam2base is None:
            self._monitor.warn("tf_lookup_failed", reason=tf_error)
            # 同步通知 object_manager 立刻失败, 不要等 pose_base 超时
            self._status_publisher.publish(String(data=f"tf_lookup_failed:{tf_error}"))
            return

        out_msg = Detection3DArray()
        out_msg.header = msg.header
        out_msg.header.frame_id = self._base_frame

        for detection in msg.detections:
            pose = detection.bbox.center

            # 构建 4x4 ob_in_cam
            ob_in_cam = np.eye(4)
            ob_in_cam[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            ob_in_cam[:3, :3] = Rotation.from_quat(q).as_matrix()

            # 变换到 base 系
            ob_in_base = T_cam2base @ ob_in_cam

            if self._fixed_axis:
                ob_in_base = self._remove_axis_rotation(ob_in_base, self._fixed_axis)

            # 写回
            pose.position.x = float(ob_in_base[0, 3])
            pose.position.y = float(ob_in_base[1, 3])
            pose.position.z = float(ob_in_base[2, 3])

            quat = Rotation.from_matrix(ob_in_base[:3, :3]).as_quat()  # [x,y,z,w]
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            out_msg.detections.append(detection)

        self._publisher.publish(out_msg)
        self._status_publisher.publish(String(data="ok"))
        self._monitor.info("pose_transformed", count=len(out_msg.detections))

    @staticmethod
    def _remove_axis_rotation(ob_in_base, axis: str):
        """保留物体指定轴朝向, 消除绕 base 系对应轴的自转。

        axis: 'x', 'y', or 'z'
        """
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        R = ob_in_base[:3, :3].copy()

        # 要保留的物体轴
        preserved = R[:, axis_index]
        preserved = preserved / np.linalg.norm(preserved)

        # 选一个不平行的参考轴
        ref_candidates = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        ref = ref_candidates[0]
        if abs(np.dot(ref, preserved)) > 0.9:
            ref = ref_candidates[1]

        # Gram-Schmidt 正交化
        u = ref - np.dot(ref, preserved) * preserved
        u /= np.linalg.norm(u)
        v = np.cross(preserved, u)
        v /= np.linalg.norm(v)

        result = ob_in_base.copy()
        cols = [None, None, None]
        cols[axis_index] = preserved
        others = [i for i in range(3) if i != axis_index]
        cols[others[0]] = u
        cols[others[1]] = v
        result[:3, 0] = cols[0]
        result[:3, 1] = cols[1]
        result[:3, 2] = cols[2]
        return result


def main():
    parser = argparse.ArgumentParser(description="通过 tf2_ros 将 Detection3DArray 位姿从相机系转换到 base 系。")
    parser.add_argument("--input", default="/fdpose/pose", help="输入 Detection3DArray 话题。")
    parser.add_argument("--output", default="/fdpose/pose_base", help="输出 Detection3DArray 话题。")
    parser.add_argument("--namespace", default="fdpose", help="节点命名空间。")
    parser.add_argument(
        "--camera-frame",
        default="realsense_head_color_optical_frame",
        help="相机 TF frame (默认: realsense_head_color_optical_frame)。",
    )
    parser.add_argument("--base-frame", default="BASE", help="目标 TF frame (默认: BASE)。")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = PoseCamToBaseNode(
        input_topic=args.input,
        output_topic=args.output,
        namespace=args.namespace,
        camera_frame=args.camera_frame,
        base_frame=args.base_frame,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
