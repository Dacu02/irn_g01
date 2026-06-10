#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class Roam(Node):
	def __init__(self):
		super().__init__('roam')
		self._pose_subscription = self.create_subscription(PoseStamped, '/aruco/pose', self.pose_callback, 10 )

	def pose_callback(self, _: PoseStamped):
		self.get_logger().info('Ricevuto messaggio su /aruco/pose, termino il nodo.')
		rclpy.shutdown()


def main(args=None):
	rclpy.init(args=args)
	node = Roam()
	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == '__main__':
	main()
