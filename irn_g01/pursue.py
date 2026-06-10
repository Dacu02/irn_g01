#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from .literals import EMPTY_MESSAGE, is_aruco_pose_empty

class Positions:
    def __init__(self, max_size: int = 10):
        self._poses: list[PoseStamped] = []
        self._size = 0
        self._max_size = max_size

    def add_pose(self, pose: PoseStamped):
        if self._size >= self._max_size:
            self._poses.pop(0)
        self._poses.append(pose)
        self._size += 1

    def get_last_pose(self) -> PoseStamped | None:
        if not self._poses or len(self._poses) == 0:
            return None
        return self._poses[-1]
    
    def get_poses(self) -> list[PoseStamped]:
        return self._poses

    def clear(self) -> bool:
        self._poses.clear()
        self._size = 0
        return True

    def __len__(self) -> int:
        return self._size


class Pursue(Node):
    def __init__(self):
        super().__init__('pursue')
        self._pose_subscription = self.create_subscription(PoseStamped, '/aruco/pose', self.pose_callback, 10)
        self._last_marker_poses = Positions(max_size=10)

    def pose_callback(self, msg: PoseStamped):
        if is_aruco_pose_empty(msg):
            self.get_logger().info('Marker perso: messaggio vuoto ricevuto.')
            self._last_marker_poses.clear()
            return
        
        self._last_marker_poses.add_pose(msg)

def main(args=None):
	rclpy.init(args=args)
	node = Pursue()
	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()
