import math

import cv2
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
EMPTY_MESSAGE = PoseStamped()
EMPTY_MESSAGE.pose.position.x = 0.0
EMPTY_MESSAGE.pose.position.y = 0.0
EMPTY_MESSAGE.pose.position.z = 0.0
EMPTY_MESSAGE.pose.orientation.x = 0.0
EMPTY_MESSAGE.pose.orientation.y = 0.0
EMPTY_MESSAGE.pose.orientation.z = 0.0
EMPTY_MESSAGE.pose.orientation.w = 1.0
CAMERA_FRAME = "oakd_rgb_camera_frame"


def is_aruco_pose_empty(msg:PoseStamped) -> bool:
    return msg.pose.position.x == EMPTY_MESSAGE.pose.position.x and \
        msg.pose.position.y == EMPTY_MESSAGE.pose.position.y and \
        msg.pose.position.z == EMPTY_MESSAGE.pose.position.z and \
        msg.pose.orientation.x == EMPTY_MESSAGE.pose.orientation.x and \
        msg.pose.orientation.y == EMPTY_MESSAGE.pose.orientation.y and \
        msg.pose.orientation.z == EMPTY_MESSAGE.pose.orientation.z and \
        msg.pose.orientation.w == EMPTY_MESSAGE.pose.orientation.w
        
def rvec_to_quaternion(rvec) -> tuple[float, float, float, float]:
    """Converte rotation vector (Rodrigues) → quaternione (x, y, z, w)."""
    R, _ = cv2.Rodrigues(rvec)
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s

def quaternion_to_rpy(q:Quaternion) -> tuple[float, float, float]:
    """
    Converte un quaternione (x, y, z, w)
    in roll, pitch, yaw (radianti).
    """
    return q.x, q.y, q.z
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

def move_towards_angle(pose:Pose, range: float, angle: float) -> Pose:
    """
        Moves the pose towards a certain direction of range
        Args:
            - pose (Pose) : The pose to change
            - range (float) : The range of movement
            - angle (float)  : The angle (axis x is 0, axis y is pi/2) from [-pi, pi]
    """
    new_pose = Pose()
    new_pose.position.x = pose.position.x + range * math.cos(angle)
    new_pose.position.y = pose.position.y + range * math.sin(angle)
    new_pose.position.z = pose.position.z
    
    return new_pose