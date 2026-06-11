import math
from typing import Literal
from rclpy.parameter import Parameter
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, Twist, Pose, TransformStamped, Transform
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav2_msgs.action import NavigateToPose
from .literals import EMPTY_MESSAGE, is_aruco_pose_empty, CAMERA_FRAME, move_towards_angle, quaternion_to_rpy, linear_angle_distances
from copy import deepcopy #TODO Valutarne l'utilizzo
MAX_TRANSFORM_WAIT_TIME:int = 2  #s

TARGET_DISTANCE = 1.5
TARGET_OFFSET = TARGET_DISTANCE / 3
ANGLE_OFFSET = math.radians(20)

FRAME_MAX_TIME = 2 #s

class TransformException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class Follow(Node):

    # ========================================================== #
    #                       Initialization                       #
    # ========================================================== #

    def __init__(self):
        super().__init__('follow')
        self.get_logger().info('Follow Node iniziato.')
        self._navigator = BasicNavigator()
        
        # Subscriptions
        self._aruco_listener = self.create_subscription(PoseStamped, '/aruco/pose', self._aruco_callback, 10)

        # TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Variables
        self._last_aruco_map_pose: Pose | None = None
        self._goal_pose: Pose | None = None


    # ========================================================== #
    #                       Callbacks                            #
    # ========================================================== #

    def _aruco_callback(self, msg: PoseStamped):


        if is_aruco_pose_empty(msg):
            self.get_logger().info('Marker perso: messaggio vuoto ricevuto.')
            self._last_aruco_map_pose = None
            return
        
        try:
            map_pose = self._transform_marker_pose_to_map(msg.pose)
        except TransformException as e:
            self.get_logger().error(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
            return
        self.get_logger().info(f'Posizione del marker in mappa: {map_pose}')

        if self._last_aruco_map_pose is not None:
            self.get_logger().info('Marker già visto in precedenza, confronto con la posizione precedente.')
            linear_distance, angle_distance = linear_angle_distances(self._last_aruco_map_pose, msg.pose)
            if linear_distance > TARGET_DISTANCE:
                self.get_logger().info('Nuova posizione marker in mappa: distanza dal target sufficiente, invio nuovo goal.')
                _, _, angle = quaternion_to_rpy(map_pose.orientation)
                self.get_logger().info(f'Angolo del marker: {math.degrees(angle)}')
                self.get_logger().info(f'Movimento verso il marker, distanza lineare: {linear_distance}, distanza angolare: {math.degrees(angle_distance)}')
                front_pose = move_towards_angle(map_pose, TARGET_DISTANCE, -angle)
                self._reach_goal(front_pose)
                
            elif angle_distance > ANGLE_OFFSET:
                self.get_logger().info(f'Rotazione verso il marker, distanza angolare: {math.degrees(angle_distance)}')
                self._reach_rotation_goal(map_pose.orientation)
        else:
            self.get_logger().info('Prima volta che vedo il marker, invio goal alla posizione attuale del marker.')
            self._reach_goal(map_pose)

        self._last_aruco_map_pose = map_pose
        self.get_logger().info(f'Nuova posizione ricevuta {msg.pose}')

    # ========================================================== #
    #                       Utility                              #
    # ========================================================== #
    def _transform_marker_pose_to_map(self, marker_pose: Pose) -> Pose:
        """Transforms the marker pose from CAMERA_FRAME to map frame using TF."""
        try:
            if self._tf_buffer.can_transform('map', CAMERA_FRAME, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                tf = self._tf_buffer.lookup_transform("map", CAMERA_FRAME, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME))
                return do_transform_pose(marker_pose, tf)

        except Exception as e:
            self.get_logger().warn(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
            raise TransformException(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
        self.get_logger().error('Transform map -> camera non disponibile, impossibile trasformare la posa del marker in mappa!')
        raise TransformException('Transform map -> camera non disponibile, impossibile trasformare la posa del marker in mappa!')
    
    def get_position(self) -> Pose:
        """
            Returns the current position of the robot in the map using TF. 
            Raises an exception if the transform is not available.
        """
        try:
            if self._tf_buffer.can_transform('map', 'base_link', Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                transform = self._tf_buffer.lookup_transform('map', 'base_link', Time()) 
                pose = Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = transform.transform.translation.z
                pose.orientation = transform.transform.rotation
                return pose
            else:
                self.get_logger().error(f'Transform map -> base_link non disponibile')
                raise RuntimeError(f'Transform map -> base_link non disponibile')
        except Exception as e:
            self.get_logger().error(f'Errore TF: {str(e)}')
            raise RuntimeError('Impossibile ottenere la posizione del robot')
        
    # ========================================================== #
    #                       Navigation                           #
    # ========================================================== #
        
    def _reach_rotation_goal(self, target_quaternion:Quaternion) -> bool:
        """Funzione per inviare un goal di rotazione senza interferire nel caso il turtlebot si stia già dirigendo verso una direzione"""
        # If a pose is already present, only change angle
        if self._goal_pose is None:
            new_goal_pose = self.get_position()
            return self._reach_goal(new_goal_pose)

        new_goal_pose = Pose()
        new_goal_pose.position = self._goal_pose.position
        new_goal_pose.orientation = target_quaternion
        return self._reach_goal(new_goal_pose)

    def _reach_goal(self, target_pose: Pose) -> bool:
        """
            Funzione per inviare un goal
        """

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose = Pose()
        goal_msg.pose.position = target_pose.position
        goal_msg.pose.orientation = target_pose.orientation
        if self._goal_pose is not None:
            linear_distance, angle_distance = linear_angle_distances(self._goal_pose, target_pose)
            if linear_distance < TARGET_OFFSET and angle_distance < ANGLE_OFFSET:
                self.get_logger().info('Già abbastanza vicino al goal.')
                return True
        self._goal_pose = target_pose
        return navigate_to_pose(self._navigator, goal_msg)

def navigate_to_pose(navigator: BasicNavigator, pose: PoseStamped) -> bool:
    navigator.get_logger().info(f"Navigating to {pose}")
    return navigator.goToPose(pose)


def main(args=None):
    rclpy.init(args=args)
    param = rclpy.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, False)
    node = Follow()
    node.set_parameters([param])
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()      