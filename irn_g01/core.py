import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from enum import Enum
from geometry_msgs.msg import PoseStamped, Twist, Twist, Pose, TransformStamped, Transform
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from sensor_msgs.msg import CameraInfo, LaserScan
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rclpy.action.client import ClientGoalHandle
class State(Enum):
    ROAM = 0 
    FOLLOW = 1 
    PURSUE = 2 

CAMERA_FRAME = "oakd_rgb_camera_frame"
MAX_TRANSFORM_WAIT_TIME:int = 10  #s

TARGET_DISTANCE = 0.05
TARGET_OFFSET = TARGET_DISTANCE / 3

KP_LINEAR = 30
KP_ANGULAR = 35
STOP_CMD = Twist() # comando di stop (tutti i campi a zero)
STOP_CMD.linear.x = 0.0
STOP_CMD.angular.z = 0.0

ANGLE_OFFSET = math.radians(15)

MAX_LINEAR_SPEED: float = 5.0
MAX_ANGULAR_SPEED: float = 3.0

class Core(Node):
    def __init__(self):
        super().__init__('core')
        self._state = State.ROAM
        self.get_logger().info('Core Node iniziato in stato ROAM.')
        
        # Sottoscrizioni
        self._aruco_listener = self.create_subscription(
            PoseStamped, '/aruco/pose', self._aruco_callback, 10)
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self._lidar_subscription = self.create_subscription(
            LaserScan, '/scan', self._lidar_callback, 10)
        
        # TF e Variabili di Stato
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._goal_handle: ClientGoalHandle | None = None
        self._goal_position: Pose | None = None
        self._latest_scan: LaserScan | None = None
        
        # Tracciamento Marker
        self._last_aruco_pose_map: Pose  | None = None
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
        
    def _reach_goal(self, target_pose: Pose) -> bool:
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server navigate_to_pose non disponibile!')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = Pose()
        goal_msg.pose.pose.position.x = target_pose.position.x
        goal_msg.pose.pose.position.y = target_pose.position.y
        goal_msg.pose.pose.position.z = target_pose.position.z
        goal_msg.pose.pose.orientation.w = target_pose.orientation.w
        goal_msg.pose.pose.orientation.x = target_pose.orientation.x
        goal_msg.pose.pose.orientation.y = target_pose.orientation.y
        goal_msg.pose.pose.orientation.z = target_pose.orientation.z

        if self._goal_handle is not None and self._goal_position is not None:
            dx = goal_msg.pose.pose.position.x - self._goal_position.position.x
            dy = goal_msg.pose.pose.position.y - self._goal_position.position.y
            dz = goal_msg.pose.pose.position.z - self._goal_position.position.z
            distance_to_new_goal = math.sqrt(dx**2 + dy**2 + dz**2)

            current_yaw = self._quaternion_to_yaw(self._goal_position.orientation)
            new_yaw = self._quaternion_to_yaw(goal_msg.pose.pose.orientation)
            angle_diff = abs(new_yaw - current_yaw)
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            if distance_to_new_goal > TARGET_OFFSET or abs(angle_diff) > ANGLE_OFFSET:
                self.get_logger().info('Annullamento goal precedente...')
                self._goal_handle.cancel_goal_async()   # ← corretto: metodo sull'handle
                self._goal_handle = None
            else:
                # Goal già attivo e sufficientemente vicino al target: non inviare di nuovo
                self.get_logger().debug('Goal Nav2 già attivo e vicino al target, skip.')
                return True  # ← questa riga mancava: causa del flooding

        self._goal_position = goal_msg.pose.pose
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(
            f'Goal inviato al Nav2: ({goal_msg.pose.pose.position.x:.2f}, {goal_msg.pose.pose.position.y:.2f})'
        )
        return True
    
    def _lidar_callback(self, msg: LaserScan):
        self._latest_scan = msg

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

    def _quaternion_to_yaw(self, orientation) -> float:
        """Estrae l'angolo yaw da un quaternione."""
        import math
        # Formula standard per estrarre yaw da quaternione
        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w
        
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return yaw

    def _goal_response_callback(self, future):
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rifiutato dal server Nav2')
            self._goal_handle = None
            return
        self.get_logger().info('Goal accettato dal server Nav2')
        self._goal_handle = goal_handle  # ← salva il ClientGoalHandle, non la Future
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        """Chiamato quando Nav2 completa (o fallisce) il goal."""
        self.get_logger().info('Goal Nav2 completato.')
        self._goal_handle = None
        self._goal_position = None

    def _control_loop(self):
        """ Loop principale a frequenza fissa per gestione FSM e predizioni filtri. """
        now = self.get_clock().now()
        self.get_logger().debug(f'Controllo FSM - Stato attuale: {self._state.name}')
        # Se siamo in FOLLOW, verifichiamo se abbiamo perso il marker
        if self._state == State.FOLLOW:
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
        return False # TODO Implementare controllo LIDAR
        min_angle = min(angle1, angle2)
        max_angle = max(angle1, angle2)

        if self._latest_scan is None:
            return False
        
        if self._latest_scan.angle_min > max_angle or self._latest_scan.angle_max < min_angle:
            self.get_logger().warn('LIDAR non copre l\'intervallo di angoli richiesto per il controllo del percorso.')
            return False
        
        # Sono impossibili i controlli se la distanza è negativa, ovvero deve arretrare

        #if self._latest_scan.range_max < max_distance:
        #    self.get_logger().warn('LIDAR range max troppo corto per il controllo del percorso.')
        #    return False
        
        #if self._latest_scan.range_min > max_distance:
        #    self.get_logger().warn('LIDAR range min troppo lungo per il controllo del percorso.')
        #    return False
        

        angle_increment = self._latest_scan.angle_increment
        starting_index = int((min_angle - self._latest_scan.angle_min) / angle_increment)
        ending_index   = int((max_angle - self._latest_scan.angle_min) / angle_increment)
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

        distance = z

        angle = math.atan2(x, z)
        q = self._aruco_pose_wrt_camera.pose.orientation
        yaw = self._quaternion_to_yaw(q)
        self.get_logger().info(f'Angolo marker (yaw): {math.degrees(yaw):.1f}°')
        cmd = Twist()

        if distance > (TARGET_DISTANCE + TARGET_OFFSET) or distance < (TARGET_DISTANCE - TARGET_OFFSET):
            distance_error = distance - TARGET_DISTANCE # positivo se siamo troppo lontani, negativo se siamo troppo vicini
        else:
            distance_error = 0.0

        cmd.linear.x = saturate_value(KP_LINEAR * distance_error, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)

        if distance_error != 0.0 or abs(angle) > ANGLE_OFFSET:
            angle_error = -angle 
            cmd.angular.z = saturate_value(KP_ANGULAR * angle_error, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        else:
            angle_error = 0.0
            cmd.angular.z = 0.0

        self.get_logger().debug(f'FOLLOW - distance: {distance:.2f}, angle: {math.degrees(angle):.1f}°, cmd.linear.x: {cmd.linear.x:.2f}, cmd.angular.z: {math.degrees(cmd.angular.z):.1f}°/s')
        self.get_logger().debug(f'ERRs - distance_error: {distance_error:.2f}, angle_error: {math.degrees(angle_error) if distance_error != 0.0 or abs(angle) > ANGLE_OFFSET else 0.0:.1f}°')

        if self._is_path_clear(min(angle, -ANGLE_OFFSET), max(angle, ANGLE_OFFSET), distance - 0.15):
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
                self._goal_handle = None
                self._goal_position = None
            self._cmd_vel_publisher.publish(cmd)
        else:
            self.get_logger().warn('Percorso ostruito o vincolato, calcolo posa Nav2...')
            self._cmd_vel_publisher.publish(STOP_CMD)

            # 1. Trasforma la posa del marker nel frame map
            self._last_aruco_pose_map = self._transform_marker_pose_to_map(self._aruco_pose_wrt_camera.pose)
            if self._last_aruco_pose_map is None:
                self.get_logger().error('Impossibile trasformare la posa del marker in mappa!')
                return

            # 2. Ottieni la posizione attuale del robot in mappa
            try:
                robot_pose = self.get_position()
            except RuntimeError:
                return

            # Coordinate correnti in mappa
            xr = robot_pose.pose.position.x
            yr = robot_pose.pose.position.y
            xm = self._last_aruco_pose_map.position.x
            ym = self._last_aruco_pose_map.position.y

            # 3. Calcola il vettore dal robot al marker e la sua distanza reale in mappa
            dx = xm - xr
            dy = ym - yr
            current_map_distance = math.hypot(dx, dy)

            if current_map_distance == 0.0:
                self.get_logger().warn('Robot e Marker sono nella stessa posizione!')
                return

            # 4. Calcola la coordinata desiderata arretrando rispetto al marker lungo la retta d'aria
            desider_pose = Pose()
            # Ci posizioniamo a TARGET_DISTANCE dal marker, lungo il vettore che unisce robot e marker
            desider_pose.position.x = xm - (TARGET_DISTANCE * (dx / current_map_distance))
            desider_pose.position.y = ym - (TARGET_DISTANCE * (dy / current_map_distance))
            desider_pose.position.z = 0.0

            # 5. Calcola l'orientamento assoluto per guardare il marker in mappa
            # L'angolo della retta che unisce la nuova posa al marker è lo stesso del vettore originale
            absolute_target_angle = math.atan2(dy, dx)

            desider_pose.orientation.x = 0.0
            desider_pose.orientation.y = 0.0
            desider_pose.orientation.z = math.sin(absolute_target_angle / 2.0)
            desider_pose.orientation.w = math.cos(absolute_target_angle / 2.0)

            self.get_logger().info(f'Navigazione verso posa calcolata: ({desider_pose.position.x:.2f}, {desider_pose.position.y:.2f})')
            if not self.reach_goal(desider_pose):
                self.get_logger().error('Impossibile inviare il goal a Nav2!')
                return
            

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

    def _transform_marker_pose_to_map(self, marker_pose: Pose) -> Pose | None:
        try:
            if self._tf_buffer.can_transform('map', CAMERA_FRAME, Time(), timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)):
                tf = self._tf_buffer.lookup_transform(
                "map",
                CAMERA_FRAME,
                Time(),
                timeout=Duration(seconds=MAX_TRANSFORM_WAIT_TIME)
            )

                return do_transform_pose(marker_pose, tf)

        except Exception as e:
            self.get_logger().warn(f'Errore TF: {str(e)}')
        self.get_logger().error('Transform map -> camera non disponibile, impossibile trasformare la posa del marker in mappa!')
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
    param = rclpy.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, False)
    node = Core()
    node.set_parameters([param])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()