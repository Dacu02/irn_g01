import math
from typing import Literal
import rclpy
from rclpy.node import Node, Parameter
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Quaternion
from .literals import EMPTY_MESSAGE, rvec_to_quaternion, CAMERA_FRAME

# Marker info
MARKER_LENGTH = .2  # es. 20 cm
ARUCO_ID = 372 # ID marker
ARUCO_DICT = cv2.aruco.DICT_4X4_1000

# Lost mechanic
MAX_MARKER_LOST_TIME:int = 5 #s
MAX_MARKER_LOST_FRAMES:int = 5
LOST_CONDITION: Literal['frame', 'time', 'AND', 'OR'] = 'OR'

# Activation mechanic
DETECT_FRAMES_THRESHOLD: int = 3 
SHOULD_BE_CONSECUTIVE_FRAMES: bool = True
USE_ACTIVATION_MECHANIC_WHEN_LOST: bool = True

# Other
ALWAYS_CHECK_CAMERA_PARAMETERS = False

class ArucoReader(Node):
    def __init__(self):
        super().__init__('aruco_reader')

        # CV parameters
        self.bridge = CvBridge()
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        # Camera parameters
        self._camera_matrix = None  # K  (3x3)
        self._dist_coeffs = None    # D  (1xN)


        # ArUco corners: top-left, top-right, bottom-right, bottom-left
        h = MARKER_LENGTH / 2
        self._obj_points = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

        # Publisher
        self._pose_publisher = self.create_publisher(PoseStamped, '/aruco/pose', 10)

        # State variables
        self._timer = self.get_clock().now()
        self._lost_frames = 0
        self._seen_frames = 0
        self._lost_flag = False
        self.get_logger().info('Nodo ArucoReader avviato.')
        self._activation_flag = False # se attivare l'inseguimento dei marker


    # ------------------------------------------------------------------ #

    def set_subscriber(self, use_sim_time:bool):
        self._image_subscription = self.create_subscription(Image if use_sim_time else CompressedImage, '/oakd/rgb/preview/image_raw' if use_sim_time else '/oakd/rgb/preview/image_raw/compressed', self._general_image_callback, 10)  #TODO metti a inactive e check distruzione callback
        self._camera_info_subscription = self.create_subscription(CameraInfo, '/oakd/rgb/preview/camera_info', self._camera_info_callback, 10)

    def _general_image_callback(self, msg:Image|CompressedImage):
        if self._camera_matrix is None:
            self.get_logger().warn('CameraInfo non ancora ricevuta, salto il frame.', throttle_duration_sec=2.0)
            return
        if self._activation_flag:
            self._image_callback(msg)
        else:
            self._inactive_image_callback(msg)

    # ------------------------------------------------------------------ #

    def _camera_info_callback(self, msg: CameraInfo):
        """
        Salva camera matrix e distorsione alla PRIMA ricezione.
        K è un array flat [9] → reshape (3,3)
        D è un array di lunghezza variabile (tipicamente 5 o 8 elementi)


        Removes the subscription when parameters are received, unless ALWAYS_CHECK_CAMERA_PARAMETERS is True.
        """
        if self._camera_matrix is not None:
            if not np.array_equal(self._camera_matrix, np.array(msg.k).reshape(3, 3)):
                self.get_logger().warn('Camera matrix cambiata! Aggiorno i parametri.')

        # K: [ fx,  0, cx,
        #       0, fy, cy,
        #       0,  0,  1 ]
        self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        # D: [k1, k2, p1, p2, k3]  (plumb_bob)  o più valori (fisheye)
        self._dist_coeffs = np.array(msg.d, dtype=np.float64)

        self.get_logger().info(
            f'Camera matrix ricevuta:\n{self._camera_matrix}\n'
            f'Distorsione: {self._dist_coeffs}'
        )

        if not ALWAYS_CHECK_CAMERA_PARAMETERS:
            self._camera_info_subscription.destroy()  

    # ------------------------------------------------------------------ #

    def _inactive_image_callback(self, msg: CompressedImage|Image):
        """
            Initial callback, used until the marker is assumed to be detected (DETECT_FRAMES_THRESHOLD).
            Counts the number of frames received and activates the real callback after the threshold is reached.
        """
        if self._camera_matrix is None:
            self.get_logger().warn('CameraInfo non ancora ricevuta, salto il frame.', throttle_duration_sec=2.0)
            return
        
        frame = image_to_frame(msg, self.bridge)
        if frame is None:
            self.get_logger().error("Decodifica JPEG fallita")
            return

        corners, ids, _ = self._aruco_detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0 and ARUCO_ID in ids:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            self._seen_frames += 1
            if self._seen_frames >= DETECT_FRAMES_THRESHOLD:
                self.get_logger().info(f'Rilevati {self._seen_frames} frame, attivo la lettura marker!')
                self._activation_flag = True

        elif SHOULD_BE_CONSECUTIVE_FRAMES:
            self._seen_frames = 0  # reset se vogliamo frame consecutivi

        cv2.imshow('ArUco Camera View', frame)
        cv2.waitKey(1)

            
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: CompressedImage|Image):
        """
            Active callback, used after the marker is assumed to be detected (DETECT_FRAMES_THRESHOLD).
            Detects the marker, estimates its pose, and publishes it. If the marker is lost, counts lost frames and time, and publishes EMPTY_MESSAGE if the lost condition is met.
        """
        if self._camera_matrix is None:
            self.get_logger().warn('CameraInfo non ancora ricevuta, salto il frame.', throttle_duration_sec=2.0)
            return


        frame = image_to_frame(msg, self.bridge)
        if frame is None:
            self.get_logger().error("Decodifica JPEG fallita")
            return

        try:
            self._timer = self.get_clock().now() 
            self._lost_frames = 0
        except Exception as e:
            self.get_logger().error(f'Conversione immagine fallita: {e}')
            return

        corners, ids, _ = self._aruco_detector.detectMarkers(frame)

        if ids is not None and len(ids) > 0 and ARUCO_ID in ids:
            idx = np.where(ids == ARUCO_ID)[0][0]  # indice del marker con ID desiderato
            corner = corners[idx][0]  # shape (4, 2)
            img_points = corner.astype(np.float32)  # proiezioni 2D dei 4 angoli del marker

            # ---- STIMA DELLA POSA con solvePnP ----
            # Risolve: dati punti 3D noti (obj_points) e loro
            # proiezioni 2D (img_points), trova R e t della camera
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_points,
                img_points,
                self._camera_matrix,
                self._dist_coeffs, # type: ignore
                flags=cv2.SOLVEPNP_IPPE_SQUARE  # ottimale per marker quadrati
            ) # type: ignore

            if not ok:
                self.get_logger().error('solvePnP ha fallito!')
                return
            
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)   # contorno verde + ID
            if self._dist_coeffs is None:
                raise ValueError("Distortion coefficients are not available")
            cv2.drawFrameAxes(frame, self._camera_matrix, self._dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.5)

            # tvec → posizione [m] nel frame camera
            # rvec → orientamento (Rodrigues) → quaternione
            qx, qy, qz, qw = rvec_to_quaternion(rvec)
            pose_msg = PoseStamped()
            pose_msg.pose.position.x = float(tvec[0][0])
            pose_msg.pose.position.y = float(tvec[1][0])
            pose_msg.pose.position.z = float(tvec[2][0])
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw

            self._pose_publisher.publish(pose_msg)   

            self.get_logger().info(
                f'Marker {str(ARUCO_ID)}: x={tvec[0][0]:.3f}m  y={tvec[1][0]:.3f}m  z={tvec[2][0]:.3f}m'
            )
        else:
            self.get_logger().debug('Marker non rilevato.')
            #self._pose_publisher.publish(EMPTY_MESSAGE)
            
            self._lost_frames += 1
            elapsed_time = (self.get_clock().now() - self._timer).nanoseconds  / 1e9
            
            lost_by_time = elapsed_time > MAX_MARKER_LOST_TIME
            lost_by_frames = self._lost_frames > MAX_MARKER_LOST_FRAMES

            if LOST_CONDITION == 'time' and lost_by_time:
                self.get_logger().warn(f'Marker perso da {elapsed_time:.1f}s (soglia {MAX_MARKER_LOST_TIME}s)')
                self.aruco_lost()
            elif LOST_CONDITION == 'frame' and lost_by_frames:
                self.get_logger().warn(f'Marker perso da {self._lost_frames} frame (soglia {MAX_MARKER_LOST_FRAMES} frame)')
                self.aruco_lost()
            elif LOST_CONDITION == 'AND' and lost_by_time and lost_by_frames:
                self.get_logger().warn(f'Marker perso da {elapsed_time:.1f}s e {self._lost_frames} frame (soglie: {MAX_MARKER_LOST_TIME}s e {MAX_MARKER_LOST_FRAMES} frame)')
                self.aruco_lost()
            elif LOST_CONDITION == 'OR' and (lost_by_time or lost_by_frames):
                self.get_logger().warn(f'Marker perso da {elapsed_time:.1f}s o {self._lost_frames} frame (soglie: {MAX_MARKER_LOST_TIME}s o {MAX_MARKER_LOST_FRAMES} frame)')
                self.aruco_lost()
            
        cv2.imshow('ArUco Camera View', frame)
        cv2.waitKey(1)

    def aruco_lost(self):
        """
            Called when the ArUco marker is lost.
                - Publishes EMPTY_MESSAGE to indicate loss.
                - If USE_ACTIVATION_MECHANIC_WHEN_LOST is True, resets the activation mechanic to allow re-detection.
        """
        if not self._lost_flag:
            self._pose_publisher.publish(EMPTY_MESSAGE)
            self._lost_flag = True
            self.get_logger().info('Marker perso: pubblicato messaggio vuoto.')
        if USE_ACTIVATION_MECHANIC_WHEN_LOST:
            self.get_logger().info('Riattivo meccanismo di attivazione dopo perdita marker.')
            self._activation_flag = False
            self._seen_frames = 0
            
def image_to_frame(msg:Image|CompressedImage, bridge:CvBridge) -> np.ndarray:
    if isinstance(msg, Image):
        return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    else:  # CompressedImage
        return bridge.compressed_imgmsg_to_cv2(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoReader()
    
    node.set_subscriber(node.get_parameter('use_sim_time').value) # type: ignore
    node.get_logger().info('Sim a tempo reale: ' + str(node.get_parameter('use_sim_time').value)) # type: ignore
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Chiusura...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()