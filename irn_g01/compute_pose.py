import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
import tf2_ros
from tf2_ros import Time
from .literals import EMPTY_MESSAGE, ANGLE_OFFSET, MAX_TRANSFORM_WAIT_TIME, KalmanTracker, is_aruco_pose_empty, TransformException, CAMERA_FRAME, quaternion_to_rpy, yaw_to_quaternion, TARGET_OFFSET, TARGET_DISTANCE, linear_angle_distances, get_position, AVOID_ARUCO_ANGLE
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
RETREAT_WHEN_TOO_CLOSE = False
POSES_TO_LOST = 1
class ComputePose(Node):
    def __init__(self):
        super().__init__('compute_pose')
        self._tracker = KalmanTracker()
        self._subscription = self.create_subscription(PoseStamped, '/aruco/pose', self.pose_callback, 10)
        self._publisher = self.create_publisher(PoseStamped, '/estimated_pose', 10)
        self._lost_counter = 0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.get_logger().info('ComputePose node initialized, waiting for ArUco poses...')
        self._timer_tf = self.create_timer(MAX_TRANSFORM_WAIT_TIME + 2.0, self._check_transform_available)
        self._tf_ready = False

    def _check_transform_available(self):
        try:
            if self._tf_buffer.can_transform('map', CAMERA_FRAME, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                self.get_logger().info('TF map -> {} disponibile, pronto a trasformare pose.'.format(CAMERA_FRAME))
                self._tf_ready = True
                self._timer_tf.cancel()  # Stop checking once the transform is available
            else:
                self.get_logger().warn('TF map -> {} non disponibile, ritentando...'.format(CAMERA_FRAME))
        except Exception as e:
            self.get_logger().warn(f'Errore durante l\'attesa del TF: {str(e)}. Ritentando...')

    def pose_callback(self, msg:PoseStamped):
        if not self._tf_ready:
            return  # Skip processing until TF is ready
        self.get_logger().info(f'Received ArUco pose: x{msg.pose.position.x:.2f}, y{msg.pose.position.y:.2f}, z{msg.pose.position.z:.2f}')
        
        if is_aruco_pose_empty(msg):
            try:
                front_pose = PoseStamped()
                front_pose.pose = self._tracker.predict_only(RclpyTime.from_msg(msg.header.stamp))
                front_pose.header = msg.header
                self._lost_counter += 1
                if self._lost_counter >= POSES_TO_LOST:
                    self.get_logger().warn('Marker lost, publishing empty pose.')
                    empty_mex = EMPTY_MESSAGE
                    empty_mex.header = msg.header
                    self._publisher.publish(empty_mex)
                else:

                    self._publisher.publish(front_pose)
            except:
                self._publisher.publish(msg)
                return
            return
        self.get_logger().info('Marker visibile, aggiornando tracker con la nuova posa.')
        robot_pose = get_position(self._tf_buffer)
        map_pose = PoseStamped()
        map_pose.pose = self._transform_marker_pose_to_map(msg)
        map_pose.header = msg.header
        if AVOID_ARUCO_ANGLE:
            yaw = math.atan2(map_pose.pose.position.y, map_pose.pose.position.x)
            qx, qy, qz, qw = yaw_to_quaternion(math.pi + yaw)
            map_pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        front_pose = PoseStamped()
        front_pose.pose = compute_front_pose(map_pose.pose, distance=TARGET_DISTANCE)
        self.get_logger().info(f'Calcolando posa frontale: marker at ({map_pose.pose.position.x:.2f}, {map_pose.pose.position.y:.2f}), yaw {math.degrees(quaternion_to_rpy(map_pose.pose.orientation)[2]):.2f}°, front at ({front_pose.pose.position.x:.2f}, {front_pose.pose.position.y:.2f}, yaw {math.degrees(quaternion_to_rpy(front_pose.pose.orientation)[2]):.2f}°)')
        front_pose.header = msg.header
        self._tracker.update(front_pose)
        estimated_pose = self._tracker.estimated_pose

        linear_distance, angle_distance = linear_angle_distances(robot_pose, estimated_pose)
        if linear_distance < TARGET_DISTANCE + TARGET_OFFSET and abs(angle_distance) < ANGLE_OFFSET:
            self.get_logger().info('Marker troppo vicino e già ben orientato, non invio nuovi goal.')
            self.get_logger().debug(f'Distanza lineare: {linear_distance:.2f} m, distanza angolare: {math.degrees(angle_distance):.1f}°')
            return
        
        if not RETREAT_WHEN_TOO_CLOSE and linear_angle_distances(estimated_pose, map_pose.pose)[0] > linear_angle_distances(robot_pose, map_pose.pose)[0]:
            self.get_logger().info('Marker dietro il robot, distanza: {}, non invio nuovi goal.'.format(linear_angle_distances(estimated_pose, map_pose.pose)[0]))
            return
        
        output_pose = PoseStamped()
        output_pose.pose = estimated_pose
        output_pose.header = msg.header
        self.get_logger().info('Marker visibile, pubblicando posa stimata: x:{:.2f}, y:{:.2f}, yaw:{:.2f}'.format(estimated_pose.position.x, estimated_pose.position.y, math.degrees(quaternion_to_rpy(estimated_pose.orientation)[2]))) 
        self._publisher.publish(output_pose)

    def _transform_marker_pose_to_map(self, marker_pose: PoseStamped|Pose) -> Pose:
        """Transforms the marker pose from frame to map frame using TF."""
        if isinstance(marker_pose, PoseStamped):
            source_frame = CAMERA_FRAME # TODO: Use marker_pose.header.frame_id
            pose = marker_pose.pose
        else:
            source_frame = CAMERA_FRAME
            pose = marker_pose
        try:
            if self._tf_buffer.can_transform('map', source_frame, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                tf = self._tf_buffer.lookup_transform("map", source_frame, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME))
                return do_transform_pose(pose, tf)

        except Exception as e:
            self.get_logger().warn(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
            raise TransformException(f'Errore durante la trasformazione del marker in mappa: {str(e)}')
        self.get_logger().error(f'Transform map -> {source_frame} non disponibile, impossibile trasformare la posa del marker in mappa!')
        raise TransformException(f'Transform map -> {source_frame} non disponibile, impossibile trasformare la posa del marker in mappa!')
    
def compute_front_pose(marker_map_pose: Pose, distance: float) -> Pose:
        """
        Calcola la posa davanti al marker lungo il suo asse Z nel frame mappa.
        Il robot viene orientato a guardare il marker.
        
        L'asse Z del marker ArUco punta verso la telecamera (robot),
        quindi il target è: marker_pos + Z_marker_in_mappa * distance.
        """
        if not AVOID_ARUCO_ANGLE:
            q = marker_map_pose.orientation
            _, _, yaw = quaternion_to_rpy(q)
        else:
            yaw = math.atan2(marker_map_pose.position.y, marker_map_pose.position.x)
        fwd_x = math.cos(yaw)
        fwd_y = math.sin(yaw)

        front = Pose()
        front.position.x = marker_map_pose.position.x + fwd_x * distance
        front.position.y = marker_map_pose.position.y + fwd_y * distance
        front.position.z = 0.0

        # Orientamento: il robot deve guardare VERSO il marker (direzione opposta a fwd)
        oppposed_yaw = (yaw + math.pi) % (2 * math.pi) - math.pi  # Normalizza tra -pi e pi
        qx, qy, qz, qw = yaw_to_quaternion(oppposed_yaw)
        front.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        return front

def main(args=None):
    rclpy.init(args=args)
    node = ComputePose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()