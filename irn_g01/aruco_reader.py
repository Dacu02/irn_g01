import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

# Dimensione fisica REALE del marker stampato (in metri!)
MARKER_LENGTH = 0.20  # es. 20 cm
ARUCO_ID = 372 # ID del marker da rilevare
EMPTY_MESSAGE = PoseStamped()
EMPTY_MESSAGE.pose.position.x = 0.0
EMPTY_MESSAGE.pose.position.y = 0.0
EMPTY_MESSAGE.pose.position.z = 0.0
EMPTY_MESSAGE.pose.orientation.x = 0.0
EMPTY_MESSAGE.pose.orientation.y = 0.0
EMPTY_MESSAGE.pose.orientation.z = 0.0
EMPTY_MESSAGE.pose.orientation.w = 1.0


def rvec_to_quaternion(rvec):
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


class ArucoReader(Node):
    def __init__(self):
        super().__init__('aruco_reader')
        self.bridge = CvBridge()
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        # Parametri camera — popolati dal callback CameraInfo
        self._camera_matrix = None  # K  (3x3)
        self._dist_coeffs = None    # D  (1xN)

        # Punti 3D del marker nel suo sistema di riferimento locale
        # ArUco corners: top-left, top-right, bottom-right, bottom-left
        h = MARKER_LENGTH / 2
        self._obj_points = np.array([
            [-h,  h, 0],
            [ h,  h, 0],
            [ h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float32)

        self._image_subscription = self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self._image_callback, 10)
        self._camera_info_subscription = self.create_subscription(
            CameraInfo, '/oakd/rgb/preview/camera_info', self._camera_info_callback, 10)

        # PoseStamped invece di Pose: include header con frame e timestamp
        self._pose_publisher = self.create_publisher(PoseStamped, '/aruco/pose', 10)

        self.get_logger().info('Nodo ArucoReader avviato.')

    # ------------------------------------------------------------------ #
    def _camera_info_callback(self, msg: CameraInfo):
        """
        Salva camera matrix e distorsione alla PRIMA ricezione.
        K è un array flat [9] → reshape (3,3)
        D è un array di lunghezza variabile (tipicamente 5 o 8 elementi)
        """
        if self._camera_matrix is not None:
            return  # già ricevuti, non serve aggiornarli ogni frame

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

    # ------------------------------------------------------------------ #
    def _image_callback(self, msg: Image):
        if self._camera_matrix is None:
            self.get_logger().warn('CameraInfo non ancora ricevuta, salto il frame.', throttle_duration_sec=2.0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE  # ottimale per marker quadrati
            ) # type: ignore

            if not ok:
                self.get_logger().error('solvePnP ha fallito!')
                return

            # Disegna gli assi XYZ sul marker (debug visivo)
            cv2.drawFrameAxes(
                frame, self._camera_matrix, self._dist_coeffs,
                rvec, tvec, MARKER_LENGTH * 0.5
            ) # type: ignore

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
            self.get_logger().info('Marker non rilevato.')
            self._pose_publisher.publish(EMPTY_MESSAGE)

        cv2.imshow('ArUco Camera View', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoReader()
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