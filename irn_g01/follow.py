import math
from typing import Literal
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Twist, Twist, Pose, TransformStamped, Transform
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from .literals import EMPTY_MESSAGE, is_aruco_pose_empty, CAMERA_FRAME, quaternion_to_rpy
MAX_TRANSFORM_WAIT_TIME:int = 10  #s

TARGET_DISTANCE = 0.05
TARGET_OFFSET = TARGET_DISTANCE / 3
ANGLE_OFFSET = math.radians(15)

FRAME_MAX_TIME = 2 #s

class Follow(Node):
    def __init__(self):
        super().__init__('follow')
        self.get_logger().info('Follow Node iniziato.')
        
        # Subscriptions
        self._aruco_listener = self.create_subscription(
            PoseStamped, '/aruco/pose', self._aruco_callback, 10)


        # TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Variables
        self._last_aruco_map_pose: Pose | None = None


    # ========================================================== 
    #                       Callbacks
    # ========================================================== 
    def _aruco_callback(self, msg: PoseStamped):
        if is_aruco_pose_empty(msg):
            self.get_logger().info('Marker perso: messaggio vuoto ricevuto.')
            self._last_aruco_map_pose = None
            # TODO fermare il robot e lasciar lavorare PURSUE.PY fino a nuovo marker
            return
              
        self._last_aruco_map_pose = self._transform_marker_pose_to_map(msg.pose)
        q = self._last_aruco_map_pose.orientation
        r, p, y = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        self.get_logger().info(f'Posizione marker in mappa aggiornata: x={self._last_aruco_map_pose.position.x:.2f}, y={self._last_aruco_map_pose.position.y:.2f}, roll={math.degrees(r):.2f}°, pitch={math.degrees(p):.2f}°, yaw={math.degrees(y):.2f}°')

    # ==========================================================
    #                       Utility
    # ==========================================================
    def _transform_marker_pose_to_map(self, marker_pose: Pose) -> Pose:
        """
            Transforms the marker pose from CAMERA_FRAME to map frame using TF.
        """
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
        """Returns the current position of the robot in the map using TF. """
        try:
            if self._tf_buffer.can_transform('map', CAMERA_FRAME, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                transform = self._tf_buffer.lookup_transform('map', CAMERA_FRAME, Time()) 
                pose = Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = transform.transform.translation.z
                pose.orientation = transform.transform.rotation
                return pose
            else:
                self.get_logger().error(f'Transform map -> CAMERA_FRAME {CAMERA_FRAME} non disponibile')
                raise RuntimeError(f'Transform map -> CAMERA_FRAME {CAMERA_FRAME} non disponibile')
        except Exception as e:
            self.get_logger().error(f'Errore TF: {str(e)}')
            raise RuntimeError('Impossibile ottenere la posizione del robot')
