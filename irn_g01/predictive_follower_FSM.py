import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from rclpy.action.client import ClientGoalHandle
from action_msgs.msg import GoalStatus
from rclpy.time import Time as RclpyTime
from nav2_simple_commander.robot_navigator import FollowPath, SmoothPath, ComputePathToPose
from rclpy.action import ActionClient
from std_msgs.msg import String
from .literals import is_aruco_pose_empty, ANGLE_OFFSET, CAMERA_FRAME, quaternion_to_rpy, linear_angle_distances, yaw_to_quaternion, TransformException, get_position, TARGET_DISTANCE, TARGET_OFFSET
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from rclpy.task import Future
from enum import Enum

FULL_QOS = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

PLANNER = 'Smac2D'
CONTROLLER = 'SmoothPursuit'
SMOOTHER = 'GolaySmoother'

class State(Enum):
    IDLE = 0
    COMPUTING_PATH = 1
    SMOOTHING_PATH = 2
    FOLLOWING_PATH = 3
    PARALLEL_COMPUTATION = 4
    PARALLEL_SMOOTHING = 5
    SUBSTITUTE_PATH = 6

class FSMException(Exception):
    def __init__(self, message):
        super().__init__(message)

def compare_poses(pose1: Pose, pose2: Pose, target_offset: float = TARGET_DISTANCE, angle_offset: float = ANGLE_OFFSET) -> bool:
        """
            Compare two poses and return True if they are within the specified offsets.
        """
        linear_distance, angle_distance = linear_angle_distances(pose1, pose2)
        return linear_distance < target_offset and abs(angle_distance) < angle_offset    

class PredictiveFollowerFSM(Node):

    # ========================================================== #
    #                       Initialization                       #
    # ========================================================== #

    def __init__(self):
        super().__init__('PredictiveFollowerFSM')  
        self.get_logger().info('PredictiveFollowerFSM Node iniziato.')
        
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
        self._smoother_pub   = self.create_publisher(String, 'smoother_selector',   FULL_QOS)
        self._controller_pub.publish(String(data=CONTROLLER))
        self._planner_pub.publish(String(data=PLANNER))
        self._smoother_pub.publish(String(data=SMOOTHER))

        # Subscription
        self._aruco_listener = self.create_subscription(PoseStamped, '/estimated_pose', self._aruco_callback, 10)

        # Variables
        self._goal_pose: Pose | None = None
        self._next_goal_pose: Pose | None = None
        self._path = None
        self._next_path = None
        self._uuid: int = 0
        self._state: State = State.IDLE
        self._current_follow_goal_handle: ClientGoalHandle | None = None

    def set_state(self, new_state: State):
        self.get_logger().info(f'State transition: {self._state.name} -> {new_state.name}')
        self._state = new_state

    # ========================================================== #
    #                     Pose Callback                          #
    # ========================================================== #

    def _aruco_callback(self, msg: PoseStamped):

        if is_aruco_pose_empty(msg):
            self.get_logger().warn('Marker perso, non invio nuovi goal.')
            if self._state in [State.IDLE]:
                self.get_logger().warn('Marker perso e nessun goal attivo, il robot rimarrà fermo in attesa di nuovi goal.')
            return
    
        # FSM Logic
        # -------------------------------------------------------------------------------------------------------------------------------------------
        if self._state in [State.IDLE]:
            if self._goal_pose is not None and compare_poses(self._goal_pose, msg.pose):
                return
            
            self.get_logger().info('Nuovo goal ricevuto, creando percorso...')
            self._goal_pose = msg.pose
            self.create_path(self._goal_pose)
        # -------------------------------------------------------------------------------------------------------------------------------------------
        elif self._state in [State.COMPUTING_PATH, State.SMOOTHING_PATH, State.PARALLEL_COMPUTATION, State.PARALLEL_SMOOTHING, State.SUBSTITUTE_PATH]:
            return # No new goal is accepted during computation, smoothing or substitute path
        #elif self._state in [State.COMPUTING_PATH, State.SMOOTHING_PATH]:
        #    if self._goal_pose is None: 
        #        raise FSMException('Unexpected state: COMPUTING_PATH or SMOOTHING_PATH with no active goal.')
        #    
        #    if compare_poses(self._goal_pose, msg.pose): 
        #        return
        #    if (self._next_goal_pose is not None) and compare_poses(self._next_goal_pose, msg.pose): 
        #            return
        #        
        #    self.get_logger().info('Nuovo goal ricevuto durante la computazione/smoothing, prenotando nuovo goal...')
        #    self._next_goal_pose = msg.pose
        #    self.create_path(self._next_goal_pose)
        # -------------------------------------------------------------------------------------------------------------------------------------------
        #elif self._state in [State.PARALLEL_COMPUTATION, State.PARALLEL_SMOOTHING, State.SUBSTITUTE_PATH]:
        #    return  # No new goal is accepted during parallel computation or smoothing
        # -------------------------------------------------------------------------------------------------------------------------------------------
        elif self._state in [State.FOLLOWING_PATH] and self._next_goal_pose is None: 
            # No new goal is accepted during FOLLOWING_PATH when a new goal is already booked
            if self._goal_pose is None: 
                raise FSMException('Unexpected state: FOLLOWING_PATH with no active goal.')
            
            if compare_poses(self._goal_pose, msg.pose, target_offset=TARGET_OFFSET): 
                return
                
            self.get_logger().info('Nuovo goal ricevuto durante il following, prenotando nuovo goal...')
            self._next_goal_pose = msg.pose
            self.create_path(self._next_goal_pose)
        # -------------------------------------------------------------------------------------------------------------------------------------------
        else:
            return # In all other states, we do not accept new goals


    # ========================================================== #
    #                        Create Path                         #
    # ========================================================== #
    
    def create_path(self, target: Pose):
        """Creates a path to the given target pose using the ComputePathToPose action."""
        self.get_logger().info(f'Processing navigation pipeline for pose: x:{target.position.x:.2f} y:{target.position.y:.2f} yaw:{math.degrees(quaternion_to_rpy(target.orientation)[2]):.1f}°')
        time_now = self.get_clock().now().to_msg()

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = time_now
        goal_msg.pose = target

        compute_path_goal = ComputePathToPose.Goal(
            use_start=False, # We are using the current robot pose as the start, so we set use_start to False
            goal=goal_msg,
            planner_id=PLANNER
        )

        if self._state in [State.IDLE]:
            self.set_state(State.COMPUTING_PATH)
            self._goal_pose = target
        elif self._state in [State.FOLLOWING_PATH]:
            self.set_state(State.PARALLEL_COMPUTATION)
            self._next_goal_pose = target
        else:
            return
        
        #self._goal_pose = target
        self._compute_path_to_pose_client.send_goal_async(compute_path_goal).add_done_callback(self.create_path_callback)

    def create_path_callback(self, goal:Future):
        """Callback for the ComputePathToPose action."""
        goal_handle: ClientGoalHandle = goal.result() # type: ignore
        if not goal_handle:
            raise FSMException('Unexpected data received in create_path_callback.')
        elif not goal_handle.accepted:
            self.get_logger().error('Create path rejected by server.')
            if self._state in [State.PARALLEL_COMPUTATION]:
                self.set_state(State.FOLLOWING_PATH) 
                self._next_goal_pose = None
            elif self._state in [State.COMPUTING_PATH]:
                self.set_state(State.IDLE)
                self._goal_pose = None
            return


        if self._state in [State.COMPUTING_PATH]: 
            self.set_state(State.SMOOTHING_PATH)
        elif self._state in [State.PARALLEL_COMPUTATION]:
            self.set_state(State.PARALLEL_SMOOTHING)
        else:
            return # Callback was too late
        
        self.get_logger().info('Create path goal accepted, waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.smooth_path)

    # ========================================================== #
    #                        Smooth Path                         #
    # ========================================================== #
    
    def smooth_path(self, path: Future):
        self.get_logger().info('Smoothing path...')
        response: ComputePathToPose.Impl.GetResultService.Response = path.result() # type: ignore
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Smooth path failed, cannot follow path.')
            self.get_logger().error(f'Error code: {response.status}')
            self.get_logger().error(f'Error message: {response.result}')
            if self._state in [State.PARALLEL_SMOOTHING]:
                self.set_state(State.FOLLOWING_PATH) 
                self._next_goal_pose = None
            elif self._state in [State.SMOOTHING_PATH]:
                self.set_state(State.IDLE)
                self._goal_pose = None
            return
        
        result: ComputePathToPose.Result = response.result
        
        smooth_path_goal = SmoothPath.Goal(
            path=result.path,
            smoother_id=SMOOTHER
        )
        
        self._smoother_client.send_goal_async(smooth_path_goal).add_done_callback(self.smooth_path_callback)

    def smooth_path_callback(self, goal: Future):
        goal_handle: ClientGoalHandle = goal.result() # type: ignore
        if not goal_handle:
            raise FSMException('Unexpected data received in smooth_path_callback.')
        elif not goal_handle.accepted:
            self.get_logger().error('Smooth path rejected by server.')
            if self._state in [State.PARALLEL_SMOOTHING]:
                self.set_state(State.FOLLOWING_PATH) 
                self._next_goal_pose = None
            elif self._state in [State.SMOOTHING_PATH]:
                self.set_state(State.IDLE)
                self._goal_pose = None
            return

        self.get_logger().info('Smooth path goal accepted, waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.follow_path)

    # ========================================================== #
    #                        Follow Path                         #
    # ========================================================== #

    def follow_path(self, path: Future):
        self.get_logger().info('Following path...')
        response: FollowPath.Impl.GetResultService.Response = path.result() # type: ignore
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Smooth path failed, cannot follow path.')
            self.get_logger().error(f'Error code: {response.status}')
            self.get_logger().error(f'Error message: {response.result}')
            if self._state in [State.PARALLEL_SMOOTHING]:
                self.set_state(State.FOLLOWING_PATH) 
                self._next_goal_pose = None
            elif self._state in [State.SMOOTHING_PATH]:
                self.set_state(State.IDLE)
                self._goal_pose = None
            return

        result: SmoothPath.Result = response.result # type: ignore
        follow_path_goal = FollowPath.Goal(
            path=result.path,
            controller_id=CONTROLLER
        )
        if self._state in [State.PARALLEL_SMOOTHING]:
            self.set_state(State.SUBSTITUTE_PATH)
        elif self._state not in [State.SMOOTHING_PATH]:        
            return # Callback was too late
        
        self._uuid = (self._uuid + 1) % (2**16)
        uuid = self._uuid
        self._follow_path_client.send_goal_async(follow_path_goal).add_done_callback(lambda goal: self.follow_path_callback(goal, uuid))
        
    def follow_path_callback(self, goal: Future, uuid: int):
        goal_handle: ClientGoalHandle = goal.result() # type: ignore
        if not goal_handle:
            raise FSMException('Unexpected data received in follow_path_callback.')
        if goal_handle.accepted:
            self.get_logger().info('Follow path goal accepted')

            if self._state in [State.SMOOTHING_PATH] and uuid == self._uuid:
                self.set_state(State.FOLLOWING_PATH)
                self._current_follow_goal_handle = goal_handle
            elif self._state in [State.FOLLOWING_PATH] and uuid == self._uuid: 
                self.get_logger().info('Parallel path goal accepted')
            elif self._state in [State.SUBSTITUTE_PATH] and uuid == self._uuid:
                self.get_logger().info('Substitute path goal accepted')
                self.set_state(State.FOLLOWING_PATH)
                self._goal_pose = self._next_goal_pose
                self._next_goal_pose = None
                if self._current_follow_goal_handle is not None:
                    self.get_logger().info('Canceling previous follow path goal...')
                self._current_follow_goal_handle = goal_handle
        else:
            self.get_logger().error('Follow path goal rejected')
            if uuid != self._uuid:
                if self._state in [State.SUBSTITUTE_PATH]:
                    self.get_logger().warn('Parallel path goal rejected.')
                    self._next_goal_pose = None
                    self.set_state(State.FOLLOWING_PATH)
            else:
                match self._state:
                    case State.FOLLOWING_PATH:
                        self.get_logger().info('Follow path goal canceled, no new goal booked, switching to IDLE.')
                        self._goal_pose = None
                        self.set_state(State.IDLE)
                    case State.PARALLEL_COMPUTATION:
                        self.get_logger().info('Follow path goal canceled, switching to computation since new goal is already booked.')
                        self._goal_pose = self._next_goal_pose
                        self._next_goal_pose = None
                        self.set_state(State.COMPUTING_PATH)
                    case State.PARALLEL_SMOOTHING:
                        self.get_logger().info('Follow path goal canceled, switching to smoothing since new goal is already booked.')
                        self._goal_pose = self._next_goal_pose
                        self._next_goal_pose = None
                        self.set_state(State.SMOOTHING_PATH)
            return

        goal_handle.get_result_async().add_done_callback(lambda f: self.follow_path_result_callback(f, uuid))

    def follow_path_result_callback(self, result: Future, uuid: int):
        status = result.result().status # type: ignore
        if status is None:
            raise FSMException('Follow path result has no status, something went wrong.')
        if uuid != self._uuid:
            self.get_logger().warn('Received follow path result for an old goal, ignoring.')
            return
        match status:
            case GoalStatus.STATUS_ABORTED:
                self.get_logger().error('Follow path goal aborted')
                if self._state in [State.SUBSTITUTE_PATH] and uuid != self._uuid:
                    self.get_logger().warn('Follow path goal aborted, but a new goal is already booked, switching to it.')
                    self._goal_pose = self._next_goal_pose
                    self._next_goal_pose = None
                    self.set_state(State.FOLLOWING_PATH)

                elif self._state in [State.FOLLOWING_PATH] and uuid == self._uuid:
                    self.get_logger().error('Follow path goal aborted and no new goal booked, switching to IDLE.')
                    self._goal_pose = None
                    self._current_follow_goal_handle = None
                    self.set_state(State.IDLE)
                
            case GoalStatus.STATUS_CANCELED:
                self.get_logger().error('Follow path goal canceled')
                if uuid != self._uuid:
                    if self._state in [State.SUBSTITUTE_PATH]:
                        self.get_logger().warn('Follow path goal canceled, but a new goal is already booked, switching to it.')
                        self._goal_pose = self._next_goal_pose
                        self._next_goal_pose = None
                        self.set_state(State.FOLLOWING_PATH)
                else:
                    match self._state:
                        case State.FOLLOWING_PATH:
                            self.get_logger().info('Follow path goal canceled, no new goal booked, switching to IDLE.')
                            self._goal_pose = None
                            self.set_state(State.IDLE)
                        case State.PARALLEL_COMPUTATION:
                            self.get_logger().info('Follow path goal canceled, switching to computation since new goal is already booked.')
                            self._goal_pose = self._next_goal_pose
                            self._next_goal_pose = None
                            self.set_state(State.COMPUTING_PATH)
                        case State.PARALLEL_SMOOTHING:
                            self.get_logger().info('Follow path goal canceled, switching to smoothing since new goal is already booked.')
                            self._goal_pose = self._next_goal_pose
                            self._next_goal_pose = None
                            self.set_state(State.SMOOTHING_PATH)

            case GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Follow path goal succeeded')
                if uuid != self._uuid:
                    if self._state in [State.SUBSTITUTE_PATH]:
                        self.get_logger().warn('Follow path goal succeeded, but a new goal is already booked, switching to it.')
                        self._goal_pose = self._next_goal_pose
                        self._next_goal_pose = None
                        self.set_state(State.FOLLOWING_PATH)
                else:
                    match self._state:
                        case State.FOLLOWING_PATH:
                            self.get_logger().info('Follow path goal succeeded, no new goal booked, switching to IDLE.')
                            self._goal_pose = None
                            self.set_state(State.IDLE)
                        case State.PARALLEL_COMPUTATION:
                            self.get_logger().info('Follow path goal succeeded, switching to computation since new goal is already booked.')
                            self._goal_pose = self._next_goal_pose
                            self._next_goal_pose = None
                            self.set_state(State.COMPUTING_PATH)
                        case State.PARALLEL_SMOOTHING:
                            self.get_logger().info('Follow path goal succeeded, switching to smoothing since new goal is already booked.')
                            self._goal_pose = self._next_goal_pose
                            self._next_goal_pose = None
                            self.set_state(State.SMOOTHING_PATH)
                        case State.SUBSTITUTE_PATH:
                            self.get_logger().info('Follow path goal succeeded, switching to FOLLOWING_PATH since new goal is already booked.')
                            self._goal_pose = self._next_goal_pose
                            self._next_goal_pose = None
                            self.set_state(State.FOLLOWING_PATH)

def main(args=None):
    rclpy.init(args=args)
    predictive_follower_fsm = PredictiveFollowerFSM()
    rclpy.spin(predictive_follower_fsm)
    predictive_follower_fsm.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 