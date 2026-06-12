import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from nav2_simple_commander.robot_navigator import BasicNavigator
from .literals import is_aruco_pose_empty, CAMERA_FRAME, quaternion_to_rpy, linear_angle_distances
MAX_TRANSFORM_WAIT_TIME:int = 2  #s

TARGET_DISTANCE = 1
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
        self.declare_parameter('use_sim_time', False)
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
            map_pose = self._transform_marker_pose_to_map(msg)
        except TransformException as e:
            self.get_logger().error(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
            return
        self.get_logger().debug(f'Posizione del marker in mappa: {map_pose}')

        # Compares turtlebot4 position with respect to the marker
        linear_distance, angle_distance = linear_angle_distances(self.get_position(), map_pose)
        if linear_distance < TARGET_DISTANCE and abs(angle_distance) < ANGLE_OFFSET:
            self.get_logger().info('Marker troppo vicino o già ben orientato, non invio nuovi goal.')
            self._last_aruco_map_pose = map_pose
            return

        # Compares last marker position with the new one when available
        if self._last_aruco_map_pose is not None:
            linear_distance, angle_distance = linear_angle_distances(self._last_aruco_map_pose, map_pose)
            if linear_distance > TARGET_OFFSET:
                self.get_logger().info('Nuova posizione marker in mappa: distanza dal target sufficiente, invio nuovo goal.')
                _, _, angle = quaternion_to_rpy(map_pose.orientation)
                self.get_logger().info(f'Angolo del marker: {math.degrees(angle)}')
                self.get_logger().info(f'Movimento verso il marker, distanza lineare: {linear_distance}, distanza angolare: {math.degrees(angle_distance)}')
                #front_pose = move_towards_angle(map_pose, TARGET_DISTANCE, angle)
                front_pose = self._compute_front_pose(map_pose, TARGET_DISTANCE)
                self._reach_goal(front_pose)
                
            elif abs(angle_distance) > ANGLE_OFFSET:
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
    
    def _transform_marker_pose_to_map(self, marker_pose: PoseStamped) -> Pose:
        """Transforms the marker pose from CAMERA_FRAME to map frame using TF."""
        source_frame = marker_pose.header.frame_id or CAMERA_FRAME
        try:
            if self._tf_buffer.can_transform('map', source_frame, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                tf = self._tf_buffer.lookup_transform("map", source_frame, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME))
                return do_transform_pose(marker_pose.pose, tf)

        except Exception as e:
            self.get_logger().warn(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
            raise TransformException(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
        self.get_logger().error(f'Transform map -> {source_frame} non disponibile, impossibile trasformare la posa del marker in mappa!')
        raise TransformException(f'Transform map -> {source_frame} non disponibile, impossibile trasformare la posa del marker in mappa!')
    
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
        
    def _compute_front_pose(self, marker_map_pose: Pose, distance: float) -> Pose:
        """
        Calcola la posa davanti al marker lungo il suo asse Z nel frame mappa.
        Il robot viene orientato a guardare il marker.
        
        L'asse Z del marker ArUco punta verso la telecamera (robot),
        quindi il target è: marker_pos + Z_marker_in_mappa * distance.
        """
        q = marker_map_pose.orientation

        # Colonna 2 della matrice di rotazione da quaternione → asse Z del marker in mappa
        # R * [0,0,1] = [2(qx*qz + qw*qy),  2(qy*qz - qw*qx),  1-2(qx²+qy²)]
        fwd_x = 2.0 * (q.x * q.z + q.w * q.y)
        fwd_y = 2.0 * (q.y * q.z - q.w * q.x)

        # Proietta sul piano XY e normalizza
        norm = math.hypot(fwd_x, fwd_y)
        if norm > 1e-6:
            fwd_x /= norm
            fwd_y /= norm
        else:
            self.get_logger().warn('Asse Z del marker quasi verticale, direzione incerta.')

        front = Pose()
        front.position.x = marker_map_pose.position.x + fwd_x * distance
        front.position.y = marker_map_pose.position.y + fwd_y * distance
        front.position.z = 0.0

        # Orientamento: il robot deve guardare VERSO il marker (direzione opposta a fwd)
        yaw = math.atan2(-fwd_y, -fwd_x)
        front.orientation.x = 0.0
        front.orientation.y = 0.0
        front.orientation.z = math.sin(yaw / 2.0)
        front.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'Front pose: ({front.position.x:.2f}, {front.position.y:.2f}) '
            f'fwd=({fwd_x:.2f},{fwd_y:.2f}) yaw={math.degrees(yaw):.1f}°'
        )
        return front

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
        """Funzione per inviare un goal"""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose = Pose()
        goal_msg.pose.position = target_pose.position
        goal_msg.pose.orientation = target_pose.orientation
        if self._goal_pose is not None:
            linear_distance, angle_distance = linear_angle_distances(self._goal_pose, target_pose)
            if linear_distance < TARGET_OFFSET and abs(angle_distance) < ANGLE_OFFSET:
                self.get_logger().info('Già abbastanza vicino al goal.')
                return True
            else:
                self.get_logger().info(f'Nuovo goal molto distante dal precedente, invio nuovo goal')
                self._navigator.cancelTask()

        self._goal_pose = target_pose
        return navigate_to_pose(self._navigator, goal_msg)

def navigate_to_pose(navigator: BasicNavigator, pose: PoseStamped) -> bool:
    navigator.get_logger().info(f"Navigating to {pose}")
    return navigator.goToPose(pose)


def main(args=None):
    rclpy.init(args=args)
    node = Follow()
    try:
        rclpy.spin(node)
    finally:
        node._navigator.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()