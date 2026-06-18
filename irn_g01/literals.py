import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import cv2
import numpy as np
from rclpy.logging import get_logger
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
import tf2_ros
EMPTY_MESSAGE = PoseStamped()
EMPTY_MESSAGE.pose.position.x = 0.0
EMPTY_MESSAGE.pose.position.y = 0.0
EMPTY_MESSAGE.pose.position.z = 0.0
EMPTY_MESSAGE.pose.orientation.x = 0.0
EMPTY_MESSAGE.pose.orientation.y = 0.0
EMPTY_MESSAGE.pose.orientation.z = 0.0
EMPTY_MESSAGE.pose.orientation.w = 1.0
#CAMERA_FRAME = "turtlebot4/oakd_rgb_camera_frame/rgbd_camera"
#CAMERA_FRAME = "oakd_rgb_camera_optical_frame"
CAMERA_FRAME = 'base_link'
MAX_TRANSFORM_WAIT_TIME:int = 2  #s

AVOID_ARUCO_ANGLE = False
TARGET_DISTANCE = .7
TARGET_OFFSET = TARGET_DISTANCE / 3
ANGLE_OFFSET = math.radians(25)

class TransformException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np
from rclpy.time import Time as RclpyTime
from geometry_msgs.msg import Pose, PoseStamped

CAMERA_POSITION_UNCERTAINTY = 0.05        # m   – deviazione standard
CAMERA_ANGLE_UNCERTAINTY    = math.radians(35)  # rad
SIGMA_ACCEL  = 1.5   # m/s²   – accelerazione massima stimata per un umano
SIGMA_ALPHA  = 1.5   # rad/s² – accelerazione angolare massima


def is_aruco_pose_empty(msg:PoseStamped) -> bool:
    return msg.pose.position.x == EMPTY_MESSAGE.pose.position.x and \
        msg.pose.position.y == EMPTY_MESSAGE.pose.position.y and \
        msg.pose.position.z == EMPTY_MESSAGE.pose.position.z and \
        msg.pose.orientation.x == EMPTY_MESSAGE.pose.orientation.x and \
        msg.pose.orientation.y == EMPTY_MESSAGE.pose.orientation.y and \
        msg.pose.orientation.z == EMPTY_MESSAGE.pose.orientation.z and \
        msg.pose.orientation.w == EMPTY_MESSAGE.pose.orientation.w
    
def quaternion_to_rpy(q:Quaternion) -> tuple[float, float, float]:
    """
    Converte un quaternione (x, y, z, w)
    in roll, pitch, yaw (radianti).
    """
    # Roll (X)
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # gimbal lock
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def yaw_to_quaternion(yaw) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)

def linear_angle_distances(old_pose: Pose, new_pose: Pose) -> tuple[float, float]:
        """
            Determines the linear and angular distances $[-pi, pi]$ between two poses.
        """
        dx = new_pose.position.x - old_pose.position.x
        dy = new_pose.position.y - old_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)
        _, _, old_yaw = quaternion_to_rpy(old_pose.orientation)
        _, _, new_yaw = quaternion_to_rpy(new_pose.orientation)
        angle_diff = new_yaw - old_yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi # normalizza a [-pi, pi]
        return distance, angle_diff


class KalmanTracker:
    """
    Stato: [x, vx, y, vy, yaw, omega]
    Misura: [x, y, yaw] dall'ArUco
    Modello a velocità costante, lineare: niente EKF.
    """

    def __init__(self):
        self._kf = self._build_filter()
        self._last_stamp: RclpyTime | None = None

    def _build_filter(self) -> KalmanFilter:
        dim_x = 4 if AVOID_ARUCO_ANGLE else 6
        dim_z = 2 if AVOID_ARUCO_ANGLE else 3
        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        if AVOID_ARUCO_ANGLE:
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]])
            kf.R = np.diag([CAMERA_POSITION_UNCERTAINTY**2,
                            CAMERA_POSITION_UNCERTAINTY**2])
            kf.P = np.diag([1.0, 10.0, 1.0, 10.0])
        else:
            kf.H = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0]])
            kf.R = np.diag([CAMERA_POSITION_UNCERTAINTY**2,
                            CAMERA_POSITION_UNCERTAINTY**2,
                            CAMERA_ANGLE_UNCERTAINTY**2])
            kf.P = np.diag([1.0, 10.0, 1.0, 10.0, 0.5, 5.0])

        # Placeholder – sovrascritto a ogni step da _do_predict
        kf.F = np.eye(dim_x)
        kf.Q = np.eye(dim_x)
        return kf

    def _F(self, dt: float) -> np.ndarray:
        """Matrice di transizione a velocità costante."""
        if not AVOID_ARUCO_ANGLE:
            return np.array([
                [1, dt, 0,  0,  0,  0 ],
                [0, 1,  0,  0,  0,  0 ],
                [0, 0,  1,  dt, 0,  0 ],
                [0, 0,  0,  1,  0,  0 ],
                [0, 0,  0,  0,  1,  dt],
                [0, 0,  0,  0,  0,  1 ],
            ])
        else:
            return np.array([
                [1, dt, 0,  0 ],
                [0, 1,  0,  0 ],
                [0, 0,  1,  dt],
                [0, 0,  0,  1 ],
            ])

    def _Q(self, dt: float) -> np.ndarray:
        """
        Rumore di processo: modello a accelerazione come rumore bianco.
        Q_discrete_white_noise produce la matrice [[dt⁴/4, dt³/2], [dt³/2, dt²]] * σ²
        che scala correttamente con dt.
        """
        q_xy = Q_discrete_white_noise(dim=2, dt=dt, var=SIGMA_ACCEL ** 2)
        if AVOID_ARUCO_ANGLE:
            return block_diag(q_xy, q_xy)                                   # 4×4
        else:
            q_ang = Q_discrete_white_noise(dim=2, dt=dt, var=SIGMA_ALPHA ** 2)
            return block_diag(q_xy, q_xy, q_ang)                            # 6×6

    def _do_predict(self, dt: float) -> Pose:
        self._kf.F = self._F(dt)
        self._kf.Q = self._Q(dt)
        self._kf.predict()
        pose = self.estimated_pose
        return pose

    def update(self, pose: PoseStamped) -> None:
        """Chiamare a ogni frame ArUco valido: predict(dt) + update."""
        stamp = RclpyTime.from_msg(pose.header.stamp)
        if not AVOID_ARUCO_ANGLE:
            yaw = quaternion_to_rpy(pose.pose.orientation)[2]
            saved_yaw = self._kf.x[4] if self._last_stamp is not None else yaw

            # Corregge il salto di discontinuità del yaw (es. da +pi a -pi)
            if self._last_stamp is not None:
                yaw = saved_yaw + ((yaw - saved_yaw + math.pi) % (2 * math.pi) - math.pi)

            
            z = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                yaw,
            ])

            if self._last_stamp is None:
                # Prima misura: inizializza la posizione, velocità = 0
                self._kf.x = np.array([z[0], 0.0, z[1], 0.0, yaw, 0.0])
                self._last_stamp = stamp
                return
        else:
            z = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
            ])
            if self._last_stamp is None:
                self._kf.x = np.array([z[0], 0.0, z[1], 0.0])
                self._last_stamp = stamp
                return

        dt = (stamp - self._last_stamp).nanoseconds * 1e-9
        if dt <= 0.0:
            get_logger("irn_g01").error(f"Stamp non crescente: {self._last_stamp} → {stamp}")
            return

        self._do_predict(dt)
        self._kf.update(z)
        self._last_stamp = stamp

    def predict_only(self, current_time: RclpyTime) -> Pose:
        """
        Chiamare quando il marker non è visibile.
        Propaga lo stato usando le velocità stimate, senza correzione.
        L'incertezza (P) cresce con il tempo trascorso.
        """
        if self._last_stamp is None:
            raise ValueError("Impossibile fare la predizione: nessun timestamp valido presente.")
        
        dt = (current_time - self._last_stamp).nanoseconds * 1e-9
        if dt <= 0.0:
            get_logger("irn_g01").error(f"Stamp non crescente: {self._last_stamp} → {current_time}")
            raise ValueError("Timestamp non crescente: impossibile predire.")
        
        prediction = self._do_predict(dt)
        self._last_stamp = current_time
        return prediction

    @property
    def estimated_pose(self) -> Pose:
        if self._last_stamp is None:
            raise ValueError("Impossibile stimare la posizione: nessun timestamp valido presente.")
        pose = Pose()
        pose.position.x = float(self._kf.x[0])
        pose.position.y = float(self._kf.x[2])
        pose.position.z = 0.0
        if not AVOID_ARUCO_ANGLE:
            yaw = float(self._kf.x[4])
            pose.orientation.x, pose.orientation.y, \
                pose.orientation.z, pose.orientation.w = yaw_to_quaternion(yaw)
        else:
            pose.orientation.x = pose.orientation.y = pose.orientation.z = pose.orientation.w = 0.0
        return pose

    def reset(self) -> None:
        self._kf = self._build_filter()
        self._last_stamp = None

def get_position(tf_buffer: tf2_ros.Buffer) -> Pose:
    """
        Returns the current position of the robot in the map using TF. 
        Raises an exception if the transform is not available.
    """
    try:
        tf = tf_buffer.lookup_transform('map', 'base_link', tf2_ros.Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME))
    except Exception as e:
        raise TransformException(str(e))
    pose = Pose()
    pose.position.x = tf.transform.translation.x
    pose.position.y = tf.transform.translation.y
    pose.position.z = tf.transform.translation.z
    pose.orientation = tf.transform.rotation
    return pose