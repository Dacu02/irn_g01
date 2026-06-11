import math
from typing import Literal
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, Twist, Pose, TransformStamped, Transform
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from nav2_msgs.action import NavigateToPose
from .literals import EMPTY_MESSAGE, is_aruco_pose_empty, CAMERA_FRAME, linear_angle_distances, quaternion_to_rpy, linear_angle_distances
from copy import deepcopy #TODO Valutarne l'utilizzo
MAX_TRANSFORM_WAIT_TIME:int = 10  #s

TARGET_DISTANCE = 0.05
TARGET_OFFSET = TARGET_DISTANCE / 3
ANGLE_OFFSET = math.radians(15)

FRAME_MAX_TIME = 2 #s

class Follow(Node):

    # ========================================================== #
    #                       Initialization                       #
    # ========================================================== #

    def __init__(self):
        super().__init__('follow')
        self.get_logger().info('Follow Node iniziato.')
        
        # Subscriptions
        self._aruco_listener = self.create_subscription(PoseStamped, '/aruco/pose', self._aruco_callback, 10)
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

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

        # TODO DEBUG
        self.get_logger().info(f'Nuova posizione ricevuta {msg.pose}')
        map_pose = self._transform_marker_pose_to_map(msg.pose)
        self.get_logger().info(f'Posizione del marker in mappa: {map_pose}')
        # TODO DEBUG


        if is_aruco_pose_empty(msg):
            self.get_logger().info('Marker perso: messaggio vuoto ricevuto.')
            self._last_aruco_map_pose = None
            # TODO fermare il robot e lasciar lavorare PURSUE.PY fino a nuovo marker
            return

        map_pose = self._transform_marker_pose_to_map(msg.pose)

        if self._last_aruco_map_pose is not None:
            linear_distance, angle_distance = linear_angle_distances(self._last_aruco_map_pose, msg.pose)
            if linear_distance > TARGET_DISTANCE:
                self.get_logger().info('Nuova posizione marker in mappa: distanza dal target sufficiente, invio nuovo goal.')
                self._reach_goal(map_pose)
                
            elif angle_distance > ANGLE_OFFSET:
                self.get_logger().info('Nuova posizione marker vicina ma con angolo distante, invio rotazione')
                self._reach_rotation_goal(map_pose.orientation)

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
            raise e
        self.get_logger().error('Transform map -> camera non disponibile, impossibile trasformare la posa del marker in mappa!')
        raise RuntimeError('Transform map -> camera non disponibile, impossibile trasformare la posa del marker in mappa!')
    
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
        
    def _goal_response_callback(self, future):
        """Risposta dall'invio del goal al nodo di nav2"""
        goal_handle: ClientGoalHandle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rifiutato dal server Nav2')
            self._goal_handle = None
            return
        
        self.get_logger().info('Goal accettato dal server Nav2')
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, result):
        """Chiamato quando Nav2 completa (o fallisce) il goal."""
        self.get_logger().info('Goal Nav2 completato.')
        self._goal_handle = None
        self._goal_position = None

    def _reach_goal(self, target_pose: Pose) -> bool:
        """
            Funzinone per inviare un goal
        """
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server navigate_to_pose non disponibile!')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = Pose()
        goal_msg.pose.pose.position = target_pose.position
        goal_msg.pose.pose.orientation = target_pose.orientation

        self._goal_position = goal_msg.pose.pose
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(f'Goal inviato al Nav2: ({target_pose})')
        return True

def main(args=None):
    rclpy.init(args=args)
    node = Follow()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()      