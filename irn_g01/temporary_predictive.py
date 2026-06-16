import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
import tf2_ros
from rclpy.time import Time as RclpyTime
from nav2_simple_commander.robot_navigator import FollowPath, SmoothPath, ComputePathToPose
from rclpy.action import ActionClient
from std_msgs.msg import String
from .literals import is_aruco_pose_empty, CAMERA_FRAME, quaternion_to_rpy, linear_angle_distances, KalmanTracker, yaw_to_quaternion, TransformException, get_position, TARGET_DISTANCE, TARGET_OFFSET
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

FULL_QOS = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

ANGLE_OFFSET = math.radians(25)
BUFFER_SIZE = 10
PLANNER = 'Smac2D'
CONTROLLER = 'SmoothPursuit'
TIMEOUT_THRESHOLD = 2#s

class PredictiveFollower(Node):

    # ========================================================== #
    #                       Initialization                       #
    # ========================================================== #

    def __init__(self):
        super().__init__('PredictiveFollower')  
        self.get_logger().info('PredictiveFollower Node iniziato.')
        
        # Action Clients
        self._compute_path_to_pose_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self._smoother_client = ActionClient(self, SmoothPath, 'smooth_path')
        self._follow_path_client = ActionClient(self, FollowPath, 'follow_path')

        while not self._compute_path_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('ComputePathToPose action server non disponibile!')

        while not self._smoother_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('SmoothPath action server non disponibile!')
        
        while not self._follow_path_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowPath action server non disponibile!')


        # Publishers
        self._controller_pub = self.create_publisher(String, 'controller_selector', FULL_QOS)
        self._planner_pub    = self.create_publisher(String, 'planner_selector',    FULL_QOS)
        self._controller_pub.publish(String(data=CONTROLLER))
        self._planner_pub.publish(String(data=PLANNER))

        # Subscription
        self._aruco_listener = self.create_subscription(PoseStamped, '/estimated_pose', self._aruco_callback, 10)

        # TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Variables
        self._goal_pose: Pose | None = None
        self._next_goal_pose: Pose | None = None
        self._path = None
        self._next_path = None
        self._computing_path: bool = False
        #self._is_lost = False

    # ========================================================== #
    #                     Pose Callback                          #
    # ========================================================== #

    def _aruco_callback(self, msg: PoseStamped):
        if is_aruco_pose_empty(msg):
            self.get_logger().warn('Marker perso, non invio nuovi goal.')
            if self._goal_pose is None:
                self.get_logger().warn('Marker perso e nessun goal attivo, il robot rimarrà fermo in attesa di nuovi goal.')
            #self._is_lost = True
            return

        # Compares turtlebot4 position with respect to the marker
        if self._goal_pose is None:
            linear_distance, angle_distance = linear_angle_distances(get_position(self._tf_buffer), msg.pose)
            if linear_distance < TARGET_OFFSET and abs(angle_distance) < ANGLE_OFFSET:
                self.get_logger().info('Marker troppo vicino e già ben orientato, non invio nuovi goal.')
                self.get_logger().debug(f'Distanza lineare: {linear_distance:.2f} m, distanza angolare: {math.degrees(angle_distance):.1f}°')
                return

        # Compares goal position with respect to the marker
        else:
            linear_distance, angle_distance = linear_angle_distances(self._goal_pose, msg.pose)
            if linear_distance < TARGET_OFFSET and abs(angle_distance) < ANGLE_OFFSET:
                self.get_logger().info('Nuova posizione marker troppo vicina al goal attuale, non invio nuovi goal.')
                self.get_logger().debug(f'Distanza lineare: {linear_distance:.2f} m, distanza angolare: {math.degrees(angle_distance):.1f}°')
                return

        self.get_logger().info('Nuova posizione marker in mappa: distanza sufficiente')
        if self._goal_pose is None:
            self.get_logger().info('Nuova posizione marker in mappa: distanza dal target sufficiente, invio nuovo goal.')
            self.send_goal(msg.pose, get_position(self._tf_buffer))
            self._goal_pose = msg.pose
        else:
            self.get_logger().info('Nuova posizione marker in mappa: distanza dal target sufficiente, prenoto nuovo goal.')
            self._next_goal_pose = msg.pose

    # ========================================================== #
    #                      Navigation Chain                      #
    # ========================================================== #
        
    def send_goal(self, target: Pose, initial_pose: Pose):
        self.get_logger().info(f'Processing navigation pipeline for pose: x:{target.position.x:.2f} y:{target.position.y:.2f} yaw:{math.degrees(quaternion_to_rpy(target.orientation)[2]):.1f}°')

        time_now = self.get_clock().now().to_msg()

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = time_now
        goal_msg.pose = target

        start_msg = PoseStamped()
        start_msg.header.frame_id = 'map'
        start_msg.header.stamp = time_now
        start_msg.pose = initial_pose

        compute_path_goal = ComputePathToPose.Goal(
            use_start=True,
            start=start_msg,
            goal=goal_msg,
            planner_id=PLANNER
        )
        self._compute_path_to_pose_client.send_goal_async(compute_path_goal).add_done_callback(self._compute_path_response_callback)

    # ========================================================== #
    #                   Compute Path Callbacks                   #
    # ========================================================== #
    # The following section contains the callbacks for the ComputePathToPose, whose output is the path to be smoothed and then followed

    def _compute_path_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('ComputePathToPose goal rejected.')
            self._computing_path = False
            return

        self.get_logger().info('ComputePathToPose goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._compute_path_result_callback)

    def _compute_path_result_callback(self, future):
        result = future.result().result
        if result is None or not result.path.poses:
            self.get_logger().error('ComputePathToPose failed to compute a path.')
            self._computing_path = False
            return

        self.get_logger().info('Path computed successfully, sending to smoother...')
        smooth_path_goal = SmoothPath.Goal(path=result.path)
        self._smoother_client.send_goal_async(smooth_path_goal).add_done_callback(self._smooth_path_response_callback)


    # ========================================================== #
    #                    Smooth Path Callbacks                   #
    # ========================================================== #
    # The following section contains the callbacks for the SmoothPath, whose output is the smoothed path to be followed

    def _smooth_path_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('SmoothPath goal rejected.')
            self._computing_path = False
            return

        self.get_logger().info('SmoothPath goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._smooth_path_result_callback)

    def _smooth_path_result_callback(self, future):
        result = future.result().result
        if result is None or not result.path.poses:
            self.get_logger().error('SmoothPath failed to smooth the path.')
            self._computing_path = False
            return

        if self._path is None:
            self.get_logger().info('Path smoothed successfully, sending to FollowPath...')
            self._path = result.path
            follow_path_goal = FollowPath.Goal(path=self._path, controller_id=CONTROLLER)
            self._follow_path_client.send_goal_async(
                follow_path_goal, 
                self._follow_path_feedback_callback
            ).add_done_callback(self._follow_path_response_callback)
        elif self._next_path is None:
            self.get_logger().info('Path smoothed ma coda con più path da seguire, sostituzione del path in coda con il nuovo path smoothed.')
            self._next_path = result.path
        else:
            self.get_logger().info('Path smoothed ma attesa fine FollowPath precedente. Non invio il nuovo path.')
            self._next_path = result.path

    # ========================================================== #
    #                   Follow Path Callbacks                    #
    # ========================================================== #
    # The following section contains the callbacks for the FollowPath, whose output is the followed path
    # The feedback callback is used to check if there is a new goal available for the robot to follow, and if so, it sends a new goal while the robot is still following the current path.

    def _follow_path_feedback_callback(self, feedback_msg):
        distance = feedback_msg.feedback.distance_to_goal
        speed    = feedback_msg.feedback.speed

        if speed > 1e-3:
            approx_time_left = distance / speed
            self.get_logger().info(f'... time left: {approx_time_left:.2f}s')
            if approx_time_left > TIMEOUT_THRESHOLD:
                return  # too early
        elif distance >= TARGET_OFFSET:
            return  # far away and not moving

        # Qui siamo "vicini alla fine": pre-computiamo il prossimo percorso
        self.get_logger().debug('Checking for new goals...')
        if self._next_goal_pose is not None and not self._computing_path:
            self._computing_path = True
            if self._goal_pose is None:
                raise RuntimeError('Unexpected state: next_goal_pose is set but goal_pose is None')
            self.send_goal(self._next_goal_pose, self._goal_pose)
            self._goal_pose = self._next_goal_pose
            self._next_goal_pose = None

    def _follow_path_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('FollowPath goal rejected.')
            self._computing_path = False
            return

        self.get_logger().info('FollowPath goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._follow_path_result_callback)

    def _follow_path_result_callback(self, future):
        result = future.result().result
        if result is None:
            self.get_logger().error('FollowPath failed to follow the path.')
            self._computing_path = False
            return

        self.get_logger().info('FollowPath completed successfully.')

        if self._next_path is not None:
            follow_path_goal = FollowPath.Goal(path=self._next_path, controller_id=CONTROLLER)
            self._follow_path_client.send_goal_async(
                follow_path_goal, 
                self._follow_path_feedback_callback
            ).add_done_callback(self._follow_path_response_callback)
        else:
            self.get_logger().info('No new path to follow, waiting for new goals...')
            self._goal_pose = None
            # TODO Politica di timeout: se non arriva un nuovo goal entro un certo tempo, fermare il robot e aspettare nuovi goal

        
        self._path = self._next_path
        self._next_path = None
        self._computing_path = False

        #if self._is_lost:
        #    self.get_logger().warn('Marker perso durante il follow path, attivazione modalità lost. Il robot rimarrà fermo in attesa di nuovi goal.')

def main(args=None):
    rclpy.init(args=args)
    node = PredictiveFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()