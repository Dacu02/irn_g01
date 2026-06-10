import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
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

def quaternion_to_rpy(x, y, z, w) -> tuple[float, float, float]:
    """
    Converte un quaternione (x, y, z, w)
    in roll, pitch, yaw (radianti).
    """

    # Roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # gimbal lock
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw