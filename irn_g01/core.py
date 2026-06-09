import math
from typing import Literal
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from enum import Enum
from geometry_msgs.msg import PoseStamped, Twist, Twist
import tf2_ros
from sensor_msgs.msg import CameraInfo, LaserScan
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
class State(Enum):
    ROAM = 0 
    FOLLOW = 1 
    PURSUE = 2 

MAX_TRANSFORM_WAIT_TIME:int = 10  #s

MAX_MARKER_LOST_TIME:int = 5      #s
MAX_MARKER_LOST_FRAMES:int = 2
LOST_CONDITION: Literal['frame', 'time', 'AND', 'OR'] = 'OR'

TARGET_DISTANCE = 0.35
TARGET_OFFSET = 0.15

KP_LINEAR = 0.4
KP_ANGULAR = 1.2
STOP_CMD = Twist() # comando di stop (tutti i campi a zero)
STOP_CMD.linear.x = 0.0
STOP_CMD.angular.z = 0.0

ANGLE_OFFSET = math.radians(25)

MAX_LINEAR_SPEED = .5
MAX_ANGULAR_SPEED = 1

class Core(Node):
    def __init__(self):
        super().__init__('core')
        self._state = State.ROAM
        self.get_logger().info('Core Node iniziato in stato ROAM.')
        
        # Sottoscrizioni
        self._aruco_listener = self.create_subscription(
            PoseStamped, '/aruco/pose', self._aruco_callback, 10)
        
        self._camera_info_subscription = self.create_subscription(
            CameraInfo, '/oakd/rgb/preview/camera_info', self._camera_info_callback, 10)
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self._lidar_subscription = self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, 10)
        
        # TF e Variabili di Stato
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._camera_frame: str | None = None
        self._is_moving = None
        self._latest_scan: LaserScan | None = None
        
        # Tracciamento Marker
        self._last_aruco_pose_map: PoseStamped | None = None
        self._aruco_pose_wrt_camera: PoseStamped | None = None
        self._last_seen_time: Time | None = None
        self._marker_lost_frames = 0
        
        # Timer principale di controllo (FSM e Filtri) - 10 Hz
        self._control_timer = self.create_timer(0.1, self._control_loop)
        
    def get_position(self) -> PoseStamped:
        """Restituisce la posizione attuale del robot in mappa usando TF. """
        try:
            if self._tf_buffer.can_transform('map', 'base_link', Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                transform = self._tf_buffer.lookup_transform('map', 'base_link', Time())
                pose = PoseStamped()
                pose.header.stamp = transform.header.stamp
                pose.header.frame_id = 'map'
                pose.pose.position.x = transform.transform.translation.x
                pose.pose.position.y = transform.transform.translation.y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                return pose
            else:
                self.get_logger().error('Transform map -> base_link non disponibile')
                raise RuntimeError('Transform map -> base_link non disponibile')
        except Exception as e:
            self.get_logger().error(f'Errore TF: {str(e)}')
            raise RuntimeError('Impossibile ottenere la posizione del robot')
        
    def _reach_goal(self, target_pose: PoseStamped) -> bool:
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server navigate_to_pose non disponibile!')
            return False
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        if self._is_moving:
            self.get_logger().info('Annullamento goal precedente...')
            #self._is_moving.cancel_goal()
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if not result.accepted:
            self.get_logger().error('Goal non accettato!')
            return False
        else:
            self.get_logger().info('Goal accettato, in movimento...')
            self._is_moving = future
            #future.add_done_callback(lambda: self.get_logger().info('Arrivato a destinazione!') if future.result().status == 4 else self.get_logger().warn('Goal annullato o fallito!'))
            return True
        
    def _lidar_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _camera_info_callback(self, msg: CameraInfo):
        if not self._camera_frame:
            self._camera_frame = msg.header.frame_id
            self.get_logger().info(f'Camera frame impostato a: {self._camera_frame}')
            self.destroy_subscription(self._camera_info_subscription)

    def _aruco_callback(self, msg: PoseStamped):        
        if not (msg.pose.position.x == 0.0 and msg.pose.position.y == 0.0 and msg.pose.position.z == 0.0):
            self._marker_lost_frames = 0
            self._last_seen_time = self.get_clock().now()
            
            # non usare tf2 fino a quando non è strettamente necessario
            #pose_map = self._transform_marker_pose_to_map(msg)
            #if pose_map:
            #    self._last_aruco_pose_map = pose_map
            #    #TODO [QUI] Passo di UPDATE / CORRECTION del Kalman/Particle Filter
            self._aruco_pose_wrt_camera = msg
            
            # Se eravamo in ROAM o PURSUE e vediamo il marker, passiamo a FOLLOW
            if self._state != State.FOLLOW:
                self._transition_to(State.FOLLOW)
        else:
            if self._state == State.FOLLOW:
                self._marker_lost_frames += 1

    def _control_loop(self):
        """ Loop principale a frequenza fissa per gestione FSM e predizioni filtri. """
        now = self.get_clock().now()
        
        # Se siamo in FOLLOW, verifichiamo se abbiamo perso il marker
        if self._state == State.FOLLOW:
            time_lost = False
            frame_lost = self._marker_lost_frames >= MAX_MARKER_LOST_FRAMES
            
            if self._last_seen_time is not None:
                elapsed_time = (now - self._last_seen_time).nanoseconds / 1e9
                time_lost = elapsed_time >= MAX_MARKER_LOST_TIME
            
            # Valutazione condizione di perdita
            should_pursue = False
            if LOST_CONDITION == 'OR' and (time_lost or frame_lost):
                should_pursue = True
            elif LOST_CONDITION == 'AND' and (time_lost and frame_lost):
                should_pursue = True
            elif LOST_CONDITION == 'time' and time_lost:
                should_pursue = True
            elif LOST_CONDITION == 'frame' and frame_lost:
                should_pursue = True
                
            if should_pursue:
                self.get_logger().warn('Marker perso. Transizione a PURSUE.')
                self._transition_to(State.PURSUE)
                #TODO  [QUI] Inizializza il Filtro di Kalman / Particle Filter con l'ultima posa nota
                
        # Esecuzione dei comportamenti in base allo stato
        if self._state == State.ROAM:
            self._execute_roam()
        elif self._state == State.FOLLOW:
            self._execute_follow()
        elif self._state == State.PURSUE:
            self._execute_pursue(now)


    def _execute_roam(self):
        pass # nav2

    def _is_path_clear(self, angle1: float, angle2: float, max_distance: float):
        min_angle = min(angle1, angle2)
        max_angle = max(angle1, angle2)

        if self._latest_scan is None:
            return False
        
        if self._latest_scan.angle_min > max_angle or self._latest_scan.angle_max < min_angle:
            self.get_logger().warn('LIDAR non copre l\'intervallo di angoli richiesto per il controllo del percorso.')
            return False
        
        if self._latest_scan.range_max < max_distance:
            self.get_logger().warn('LIDAR range max troppo corto per il controllo del percorso.')
            return False
        
        if self._latest_scan.range_min > max_distance:
            self.get_logger().warn('LIDAR range min troppo lungo per il controllo del percorso.')
            return False
        
        angle_increment = self._latest_scan.angle_increment
        starting_angle = (angle_increment // min_angle) - 1
        ending_angle = (angle_increment // max_angle) + 1
        starting_index = 16 - starting_angle / angle_increment  
        ending_index = 16 + ending_angle / angle_increment
        for i in range(int(starting_index), int(ending_index) + 1):
            if i < 0 or i >= len(self._latest_scan.ranges):
                continue
            if self._latest_scan.ranges[i] < max_distance:
                return False
        return True
    
    def _execute_follow(self):
        if self._aruco_pose_wrt_camera is None:
            raise RuntimeError('Nessuna posizione del marker disponibile durante il FOLLOW')

        x = self._aruco_pose_wrt_camera.pose.position.x
        z = self._aruco_pose_wrt_camera.pose.position.z

        distance = z # math.hypot(x, z) # suggerimento chatgpt

        angle = math.atan2(x, z)

        cmd = Twist()

        if distance > (TARGET_DISTANCE + TARGET_OFFSET):
            distance_error = distance - (TARGET_DISTANCE + TARGET_OFFSET)
        elif distance < (TARGET_DISTANCE - TARGET_OFFSET):
            distance_error = distance - (TARGET_DISTANCE - TARGET_OFFSET)
        else:
            distance_error = 0.0

        cmd.linear.x = saturate_value(KP_LINEAR * distance_error, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)

        if distance_error != 0.0 or abs(angle) > ANGLE_OFFSET:
            angle_error = -angle 
            cmd.angular.z = saturate_value(KP_ANGULAR * angle_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        if self._is_path_clear(min(angle, -ANGLE_OFFSET), max(angle, ANGLE_OFFSET), distance - 0.15):
            self._cmd_vel_publisher.publish(cmd)
        else:
            self.get_logger().warn('Percorso non chiaro, fermo il robot.')
            self._cmd_vel_publisher.publish(STOP_CMD)
            # TODO go to pose with nav2

    def _execute_pursue(self, now: Time):
        # TODO [QUI] Passo di PREDICTION del Filtro di Kalman / Particle Filter
        # Se il filtro diverge o passa troppo tempo (es. altri 10 secondi in pursue), torna a ROAM
        if self._last_seen_time is not None:
            total_elapsed = (now - self._last_seen_time).nanoseconds / 1e9
            if total_elapsed > (MAX_MARKER_LOST_TIME + 10.0): # Timeout di ricerca fallita
                self.get_logger().error('Inseguimento predittivo fallito. Torno in ROAM.')
                self._transition_to(State.ROAM)
                return
        
        # Muovi il robot verso la posizione *predetta* dal filtro
        pass

    def _transform_marker_pose_to_map(self, marker_pose: PoseStamped) -> PoseStamped | None:
        if not self._camera_frame:
            return None
        try:
            if self._tf_buffer.can_transform('map', self._camera_frame, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                return self._tf_buffer.transform(marker_pose, 'map', new_type=PoseStamped) # type: ignore
        except Exception as e:
            self.get_logger().warn(f'Errore TF: {str(e)}')
        return None

    def _transition_to(self, new_state: State) -> None:
        if self._state != new_state:
            self.get_logger().info(f'FSM: {self._state.name} -> {new_state.name}')
            self._state = new_state

def saturate_value(value: float, min_value: float, max_value: float) -> float:
    """Satura un valore entro i limiti specificati."""
    return max(min(value, max_value), min_value)

def main(args=None):
    rclpy.init(args=args)
    node = Core()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()