import time
import enum
import threading
import random
import math

import numpy as np

import rclpy.action
import rclpy.callback_groups
import rclpy.qos
import rclpy.duration
import rclpy.time

import robomaster.robot
import robomaster.gripper
import robomaster.action
import robomaster.protocol
import robomaster.util

import robomaster_msgs.msg
import robomaster_msgs.action
import sensor_msgs.msg

from typing import Any, TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ..client import RoboMasterROS

from .. import Module

_joint_data: np.ndarray = np.array([
    [0.000000, 0.000001, 0.000002, -0.000000, -0.000001, 0.000001, 0.000000, 0.000001, -0.000001, -0.000001, 0.000000, -0.000000, -0.000000],
    [-0.000500, 0.021745, -0.052589, 0.001998, 0.019252, -0.015623, -0.007750, -0.021849, 0.052777, -0.001530, -0.019611, 0.015616, 0.007618],
    [-0.001000, 0.047151, -0.116532, 0.004799, 0.050170, -0.040639, -0.019933, -0.047891, 0.118524, -0.004238, -0.051579, 0.041027, 0.019779],
    [-0.001500, 0.072188, -0.180082, 0.006528, 0.085342, -0.069234, -0.033497, -0.073735, 0.184429, -0.006072, -0.088152, 0.070264, 0.033395],
    [-0.002000, 0.095901, -0.240268, 0.007242, 0.121033, -0.098400, -0.046932, -0.098175, 0.246777, -0.006928, -0.125227, 0.100074, 0.046864],
    [-0.002500, 0.118303, -0.297017, 0.007334, 0.155795, -0.126943, -0.059698, -0.121193, 0.305305, -0.007166, -0.161160, 0.129118, 0.059593],
    [-0.003000, 0.139572, -0.350754, 0.007090, 0.189182, -0.154478, -0.071667, -0.142971, 0.360495, -0.007087, -0.195472, 0.156987, 0.071455],
    [-0.003500, 0.159901, -0.401966, 0.006667, 0.221161, -0.180961, -0.082868, -0.163718, 0.412928, -0.006869, -0.228171, 0.183665, 0.082499],
    [-0.004000, 0.179423, -0.451072, 0.006153, 0.251863, -0.206486, -0.093385, -0.183622, 0.463060, -0.006590, -0.259392, 0.209245, 0.092816],
    [-0.004500, 0.198252, -0.498322, 0.005578, 0.281403, -0.231136, -0.103291, -0.202775, 0.511223, -0.006295, -0.289320, 0.233864, 0.102501],
    [-0.005000, 0.216467, -0.544002, 0.004973, 0.309942, -0.255035, -0.112666, -0.221304, 0.557686, -0.005993, -0.318102, 0.257629, 0.111631],
    [-0.005500, 0.234141, -0.588260, 0.004338, 0.337580, -0.278255, -0.121564, -0.239261, 0.602659, -0.005700, -0.345888, 0.280654, 0.120278],
    [-0.006000, 0.251334, -0.631296, 0.003690, 0.364435, -0.300889, -0.130045, -0.256731, 0.646332, -0.005417, -0.372793, 0.303027, 0.128498],
    [-0.006500, 0.268096, -0.673193, 0.003022, 0.390571, -0.322984, -0.138144, -0.273734, 0.688828, -0.005148, -0.398923, 0.324826, 0.136340],
    [-0.007000, 0.284446, -0.714093, 0.002349, 0.416082, -0.344614, -0.145905, -0.290339, 0.730251, -0.004885, -0.424339, 0.346095, 0.143836],
    [-0.007500, 0.300442, -0.754059, 0.001661, 0.441010, -0.365807, -0.153352, -0.306559, 0.770725, -0.004642, -0.449133, 0.366907, 0.151028],
    [-0.008000, 0.316092, -0.793217, 0.000975, 0.465437, -0.386631, -0.160523, -0.322453, 0.810319, -0.004405, -0.473342, 0.387287, 0.157935],
    [-0.008500, 0.331436, -0.831572, 0.000274, 0.489374, -0.407089, -0.167427, -0.338010, 0.849114, -0.004188, -0.497039, 0.407291, 0.164590],
    [-0.009000, 0.346478, -0.869249, -0.000421, 0.512897, -0.427245, -0.174099, -0.353294, 0.887166, -0.003979, -0.520246, 0.426936, 0.171006],
    [-0.009500, 0.361260, -0.906241, -0.001130, 0.536006, -0.447092, -0.180543, -0.368281, 0.924544, -0.003789, -0.543027, 0.446271, 0.177210],
    [-0.010000, 0.375773, -0.942650, -0.001831, 0.558766, -0.466686, -0.186786, -0.383031, 0.961285, -0.003606, -0.565395, 0.465303, 0.183211],
    [-0.010500, 0.390066, -0.978479, -0.002542, 0.581175, -0.486021, -0.192834, -0.397519, 0.997458, -0.003445, -0.587408, 0.484081, 0.189032],
    [-0.011000, 0.404108, -1.013808, -0.003243, 0.603298, -0.505152, -0.198710, -0.411805, 1.033070, -0.003287, -0.609059, 0.502593, 0.194674],
    [-0.011500, 0.417949, -1.048622, -0.003955, 0.625121, -0.524063, -0.204414, -0.425854, 1.068183, -0.003150, -0.630403, 0.520888, 0.200160],
    [-0.012000, 0.431575, -1.083008, -0.004657, 0.646699, -0.542801, -0.209968, -0.439720, 1.102820, -0.003021, -0.651444, 0.538963, 0.205492],
    [-0.012500, 0.445021, -1.116945, -0.005367, 0.668020, -0.561352, -0.215369, -0.453373, 1.137031, -0.002912, -0.672231, 0.556861, 0.210690],
    [-0.013000, 0.458255, -1.150519, -0.006062, 0.689148, -0.579773, -0.220643, -0.466872, 1.170808, -0.002805, -0.692741, 0.574558, 0.215749],
    [-0.013500, 0.471341, -1.183679, -0.006774, 0.710041, -0.598023, -0.225777, -0.480158, 1.204227, -0.002726, -0.713049, 0.592120, 0.220695],
    [-0.014000, 0.484216, -1.216543, -0.007462, 0.730788, -0.616181, -0.230802, -0.493322, 1.237256, -0.002642, -0.733109, 0.609502, 0.225514],
    [-0.014500, 0.496954, -1.249016, -0.008171, 0.751328, -0.634186, -0.235700, -0.506272, 1.269958, -0.002588, -0.752996, 0.626772, 0.230233],
    [-0.014999, 0.509503, -1.281234, -0.008857, 0.771742, -0.652117, -0.240500, -0.519108, 1.302327, -0.002531, -0.772677, 0.643896, 0.234842],
    [-0.015499, 0.521919, -1.313129, -0.009556, 0.791995, -0.669935, -0.245190, -0.531758, 1.334413, -0.002501, -0.792210, 0.660925, 0.239361],
    [-0.015999, 0.534150, -1.344806, -0.010233, 0.812160, -0.687707, -0.249795, -0.544307, 1.366205, -0.002470, -0.811563, 0.677829, 0.243781],
    [-0.016499, 0.546261, -1.376181, -0.010928, 0.832183, -0.705380, -0.254299, -0.556666, 1.397753, -0.002468, -0.830803, 0.694668, 0.248124],
    [-0.016999, 0.558189, -1.407370, -0.011599, 0.852143, -0.723030, -0.258728, -0.568930, 1.429031, -0.002462, -0.849888, 0.711401, 0.252376],
    [-0.017499, 0.570003, -1.438307, -0.012283, 0.871998, -0.740611, -0.263067, -0.581018, 1.460102, -0.002486, -0.868882, 0.728087, 0.256561],
    [-0.017999, 0.581638, -1.469073, -0.012948, 0.891811, -0.758184, -0.267340, -0.593009, 1.490940, -0.002509, -0.887755, 0.744693, 0.260667],
    [-0.018499, 0.593163, -1.499611, -0.013625, 0.911537, -0.775703, -0.271530, -0.604824, 1.521604, -0.002563, -0.906571, 0.761282, 0.264714],
    [-0.018999, 0.604494, -1.530035, -0.014273, 0.931271, -0.793259, -0.275668, -0.616565, 1.552047, -0.002608, -0.925271, 0.777793, 0.268685],
    [-0.019499, 0.615726, -1.560252, -0.014941, 0.950946, -0.810783, -0.279730, -0.628119, 1.582355, -0.002692, -0.943944, 0.794312, 0.272609],
    [-0.019999, 0.626763, -1.590378, -0.015579, 0.970653, -0.828363, -0.283747, -0.639598, 1.612473, -0.002769, -0.962533, 0.810781, 0.276465],
    [-0.020499, 0.637699, -1.620324, -0.016236, 0.990328, -0.845935, -0.287697, -0.650888, 1.642485, -0.002886, -0.981125, 0.827282, 0.280282],
    [-0.020999, 0.648422, -1.650219, -0.016856, 1.010079, -0.863600, -0.291612, -0.662113, 1.672323, -0.002991, -0.999649, 0.843746, 0.284035],
    [-0.021499, 0.659046, -1.679957, -0.017499, 1.029827, -0.881282, -0.295466, -0.673138, 1.702091, -0.003143, -1.018209, 0.860269, 0.287758],
    [-0.021999, 0.669438, -1.709665, -0.018102, 1.049685, -0.899086, -0.299293, -0.684097, 1.731701, -0.003282, -1.036724, 0.876776, 0.291423],
    [-0.022499, 0.679722, -1.739244, -0.018727, 1.069579, -0.916939, -0.303067, -0.694846, 1.761261, -0.003472, -1.055300, 0.893363, 0.295063],
    [-0.022918, 0.687846, -1.764646, -0.019682, 1.088293, -0.933733, -0.306504, -0.703549, 1.786527, -0.003254, -1.072585, 0.908795, 0.298343]]).T

class GripperState(enum.IntEnum):
    PAUSED = 0
    OPEN = 1
    CLOSE = 2


_UNKNOWN_GRIPPER_STATE = -1
_VALID_GRIPPER_STATES = {state.value for state in GripperState}
_VALID_STATUS_RATES = {1, 5, 10, 20, 50}


def gripper_data_info(self: robomaster.gripper.GripperSubject) -> int:
    """Return the raw status value published by the RoboMaster SDK."""
    return self._status


robomaster.gripper.GripperSubject.data_info = gripper_data_info


class Gripper(Module):
    """ROS 2 gripper module with a persistent SDK status subscription.

    Important design choices:
      * Keep one low-rate status subscription for the lifetime of the module.
      * Serialize all synchronous RoboMaster SDK calls.
      * Reserve at most one action goal at a time.
      * Use monotonic wall time for deadlines.
      * Release goal state from a ``finally`` block so one exception cannot
        permanently disable the action server.
    """

    _gripper_state: int

    def __init__(self, robot: robomaster.robot.Robot,
                 node: 'RoboMasterROS') -> None:
        self.gripper = robot.gripper
        self.node = node
        self.logger = node.get_logger()
        self.clock = node.get_clock()

        open_on_start: bool = node.declare_parameter(
            'gripper.open', False).value
        self._status_rate = int(node.declare_parameter(
            'gripper.status_rate', 5).value)
        self._action_timeout = float(node.declare_parameter(
            'gripper.action_timeout', 5.0).value)
        self._initial_state_timeout = float(node.declare_parameter(
            'gripper.initial_state_timeout', 2.0).value)
        self._status_stale_timeout = float(node.declare_parameter(
            'gripper.status_stale_timeout', 2.5).value)
        self._command_retries = max(0, int(node.declare_parameter(
            'gripper.command_retries', 2).value))
        self._retry_backoff = max(0.0, float(node.declare_parameter(
            'gripper.retry_backoff', 0.08).value))
        self._command_jitter = max(0.0, float(node.declare_parameter(
            'gripper.command_jitter', 0.05).value))

        if self._status_rate not in _VALID_STATUS_RATES:
            self.logger.warning(
                f'Unsupported gripper.status_rate={self._status_rate}; '
                'using 5 Hz')
            self._status_rate = 5

        self.msg = sensor_msgs.msg.JointState()
        self.msg.name = (
            [node.tf_frame('gripper_m_joint')] +
            [node.tf_frame(f'{side}_gripper_joint_{i}')
             for side in ('left', 'right') for i in (1, 2, 4, 5, 6, 7)])

        n = _joint_data.shape[-1]
        self._xs = np.linspace(0, 1, n)

        qos = rclpy.qos.QoSProfile(
            depth=1,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.gripper_pub = node.create_publisher(
            robomaster_msgs.msg.GripperState, 'gripper', qos)

        # RoboMaster SDK methods such as open(), close(), pause(),
        # sub_status(), and unsub_status() all perform synchronous protocol
        # transactions. Do not allow them to overlap.
        self._sdk_lock = threading.Lock()

        self._state_condition = threading.Condition()
        self._gripper_state = _UNKNOWN_GRIPPER_STATE
        self._last_status_monotonic = 0.0
        self._status_subscription_started_monotonic = 0.0
        self._status_subscribed = False

        self._goal_mutex = threading.Lock()
        self._goal_reserved = False
        self._active_goal_handle: Optional[Any] = None
        self._cancel_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._goal_done_event = threading.Event()
        self._goal_done_event.set()

        # Store the callback group as an instance member. ROS 2 callback
        # groups must remain alive for as long as their entities use them.
        self._gripper_callback_group = (
            rclpy.callback_groups.ReentrantCallbackGroup())

        # Subscribe before any open/close command. This removes the race in
        # which the gripper reaches its target before status monitoring starts.
        self._start_status_subscription()
        self.query_gripper_state()

        if open_on_start:
            if not self._send_sdk_command(GripperState.OPEN.value, 0.5):
                self.logger.warning('Initial gripper open command failed')

        self._gripper_action_server = rclpy.action.ActionServer(
            node,
            robomaster_msgs.action.GripperControl,
            'gripper',
            self.execute_gripper_callback,
            goal_callback=self.new_gripper_goal_callback,
            cancel_callback=self.cancel_gripper_callback,
            callback_group=self._gripper_callback_group)

    # ------------------------------------------------------------------
    # Lifecycle and state publication
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._shutdown_event.set()
        self.abort()

        # Best effort: do not leave a motor actively driving during shutdown.
        if getattr(self.node, 'connected', False):
            self._send_sdk_command(
                GripperState.PAUSED.value, 0.0, retries=0, jitter=False)

        self._stop_status_subscription()
        self._gripper_action_server.destroy()

    def abort(self) -> None:
        with self._goal_mutex:
            active = self._goal_reserved
            if active:
                self._cancel_event.set()

        with self._state_condition:
            self._state_condition.notify_all()

        if active:
            # Bounded wait only; never block node shutdown indefinitely.
            self._goal_done_event.wait(timeout=2.0)

    def joint_state(self, value: float) -> sensor_msgs.msg.JointState:
        self.msg.position = [
            np.interp(value, self._xs, data) for data in _joint_data]
        return self.msg

    @property
    def gripper_state(self) -> int:
        with self._state_condition:
            return self._gripper_state

    @gripper_state.setter
    def gripper_state(self, value: int) -> None:
        self._update_gripper_state(value)

    def _update_gripper_state(self, value: int) -> None:
        try:
            value = int(value)
        except (TypeError, ValueError):
            self.logger.warning(f'Ignoring invalid gripper state: {value!r}')
            return

        if value not in _VALID_GRIPPER_STATES:
            self.logger.warning(f'Ignoring unknown gripper state: {value}')
            return

        with self._state_condition:
            changed = value != self._gripper_state
            self._gripper_state = value
            self._last_status_monotonic = time.monotonic()
            self._state_condition.notify_all()

        if not changed:
            return

        # The SDK invokes this method from its receive thread. Never let a ROS
        # publication exception escape and kill that receive callback path.
        try:
            msg = robomaster_msgs.msg.GripperState(state=value)
            msg.header.stamp = self.clock.now().to_msg()
            self.gripper_pub.publish(msg)

            if value == robomaster_msgs.msg.GripperState.OPEN:
                self.node.joint_state_pub.publish(self.joint_state(0.0))
            elif value == robomaster_msgs.msg.GripperState.CLOSE:
                self.node.joint_state_pub.publish(self.joint_state(1.0))
        except Exception as exc:
            self.logger.error(
                f'Failed publishing gripper state {value}: {exc}')

    def _status_callback(self, msg: int) -> None:
        try:
            self._update_gripper_state(msg)
        except Exception as exc:
            # This callback runs in the SDK receive thread. It must never throw.
            self.logger.error(f'Unhandled gripper status callback error: {exc}')

    # ------------------------------------------------------------------
    # Persistent RoboMaster status subscription
    # ------------------------------------------------------------------

    def _start_status_subscription(self) -> bool:
        if self._shutdown_event.is_set():
            return False

        for attempt in range(self._command_retries + 1):
            try:
                with self._sdk_lock:
                    ok = bool(self.gripper.sub_status(
                        freq=self._status_rate,
                        callback=self._status_callback))
            except Exception as exc:
                ok = False
                self.logger.warning(
                    f'Gripper status subscription attempt {attempt + 1} '
                    f'raised: {exc}')

            if ok:
                self._status_subscribed = True
                self._status_subscription_started_monotonic = time.monotonic()
                return True

            self.logger.warning(
                f'Gripper status subscription attempt {attempt + 1} failed')
            self._retry_sleep(attempt)

        self._status_subscribed = False
        return False

    def _stop_status_subscription(self) -> bool:
        if not self._status_subscribed:
            return True

        try:
            with self._sdk_lock:
                ok = bool(self.gripper.unsub_status())
        except Exception as exc:
            self.logger.warning(
                f'Failed to unsubscribe from gripper status: {exc}')
            ok = False

        self._status_subscribed = False
        self._status_subscription_started_monotonic = 0.0
        return ok

    def _status_is_stale(self) -> bool:
        now = time.monotonic()
        with self._state_condition:
            last = self._last_status_monotonic
            started = self._status_subscription_started_monotonic

        # Allow a newly-created subscription time to deliver its first sample.
        if last <= 0.0:
            return started <= 0.0 or (
                now - started > self._status_stale_timeout)
        return now - last > self._status_stale_timeout

    def _ensure_status_subscription(self) -> bool:
        if self._status_subscribed and not self._status_is_stale():
            return True

        if self._status_subscribed:
            self.logger.warning(
                'Gripper status stream is stale; restarting subscription')
            self._stop_status_subscription()

        return self._start_status_subscription()

    def query_gripper_state(self) -> None:
        if not self._ensure_status_subscription():
            self.logger.warning(
                'Initial gripper status subscription is unavailable')
            return

        deadline = time.monotonic() + self._initial_state_timeout
        with self._state_condition:
            while self._gripper_state == _UNKNOWN_GRIPPER_STATE:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self.logger.warning(
                        'Timed out querying initial gripper state; '
                        'keeping state as unknown')
                    return
                self._state_condition.wait(timeout=min(0.1, remaining))

    # ------------------------------------------------------------------
    # SDK command handling
    # ------------------------------------------------------------------

    def _retry_sleep(self, attempt: int) -> None:
        delay = self._retry_backoff * (2 ** attempt)
        delay += random.uniform(0.0, self._retry_backoff)
        if delay > 0.0:
            time.sleep(delay)

    def _send_sdk_command(self, target_state: int, power: float,
                          retries: Optional[int] = None,
                          jitter: bool = True) -> bool:
        if target_state not in _VALID_GRIPPER_STATES:
            self.logger.error(
                f'Refusing invalid gripper target state {target_state}')
            return False

        if retries is None:
            retries = self._command_retries
        retries = max(0, int(retries))

        if jitter and self._command_jitter > 0.0:
            # Desynchronize a burst when many PCs send a command together.
            time.sleep(random.uniform(0.0, self._command_jitter))

        power = min(1.0, max(0.0, float(power)))
        power_percent = max(1, min(100, int(round(100.0 * power))))

        for attempt in range(retries + 1):
            try:
                with self._sdk_lock:
                    if target_state == GripperState.OPEN.value:
                        ok = bool(self.gripper.open(power=power_percent))
                    elif target_state == GripperState.CLOSE.value:
                        ok = bool(self.gripper.close(power=power_percent))
                    else:
                        ok = bool(self.gripper.pause())
            except Exception as exc:
                ok = False
                self.logger.warning(
                    f'Gripper command attempt {attempt + 1} raised: {exc}')

            if ok:
                return True

            self.logger.warning(
                f'Gripper command attempt {attempt + 1} failed '
                f'(target={target_state}, power={power_percent})')
            if attempt < retries:
                self._retry_sleep(attempt)

        return False

    # ------------------------------------------------------------------
    # ROS 2 action server
    # ------------------------------------------------------------------

    def _finish_result(self, start_monotonic: float
                       ) -> robomaster_msgs.action.GripperControl.Result:
        elapsed_ns = max(
            0, int((time.monotonic() - start_monotonic) * 1_000_000_000))
        duration = rclpy.duration.Duration(nanoseconds=elapsed_ns)
        return robomaster_msgs.action.GripperControl.Result(
            duration=duration.to_msg())

    def _release_goal(self) -> None:
        with self._goal_mutex:
            self._active_goal_handle = None
            self._goal_reserved = False
            self._cancel_event.clear()
            self._goal_done_event.set()

    def execute_gripper_callback(
            self, goal_handle: Any
            ) -> robomaster_msgs.action.GripperControl.Result:
        start_monotonic = time.monotonic()

        with self._goal_mutex:
            # Normally the reservation was made in goal_callback. Keep this
            # fallback so a lifecycle edge case cannot run two goals at once.
            if self._active_goal_handle is not None:
                goal_handle.abort()
                return self._finish_result(start_monotonic)
            self._active_goal_handle = goal_handle

        try:
            request = goal_handle.request
            target_state = int(request.target_state)
            power = float(request.power)

            self.logger.info(
                f'Start moving gripper: target={target_state}, power={power:.3f}')

            if target_state not in _VALID_GRIPPER_STATES:
                self.logger.error(
                    f'Invalid gripper target state {target_state}')
                goal_handle.abort()
                return self._finish_result(start_monotonic)

            if not math.isfinite(power) or power < 0.0 or power > 1.0:
                self.logger.error(
                    f'Invalid gripper power {power}; expected [0, 1]')
                goal_handle.abort()
                return self._finish_result(start_monotonic)

            if (self._cancel_event.is_set() or
                    goal_handle.is_cancel_requested):
                goal_handle.canceled()
                return self._finish_result(start_monotonic)

            status_ready = self._ensure_status_subscription()

            if (status_ready and not self._status_is_stale() and
                    self.gripper_state == target_state):
                goal_handle.succeed()
                self.logger.info('Gripper is already at the requested state')
                return self._finish_result(start_monotonic)

            if not status_ready and target_state != GripperState.PAUSED.value:
                self.logger.warning(
                    'No gripper status stream is available; command will be '
                    'sent, but the action cannot be confirmed')

            if not self._send_sdk_command(target_state, power):
                self.logger.error('RoboMaster SDK rejected/timed out the command')
                goal_handle.abort()
                return self._finish_result(start_monotonic)

            # The gripper ROS documentation defines PAUSE as an immediate
            # action. There is no physical end-position to wait for.
            if target_state == GripperState.PAUSED.value:
                self._update_gripper_state(GripperState.PAUSED.value)
                goal_handle.succeed()
                self.logger.info('Gripper paused')
                return self._finish_result(start_monotonic)

            deadline = time.monotonic() + self._action_timeout
            last_feedback_state: Optional[int] = None
            resubscribe_attempted = False

            while True:
                if (self._cancel_event.is_set() or
                        goal_handle.is_cancel_requested):
                    self._send_sdk_command(
                        GripperState.PAUSED.value,
                        0.0,
                        retries=0,
                        jitter=False)
                    goal_handle.canceled()
                    self.logger.warning('Gripper action canceled')
                    return self._finish_result(start_monotonic)

                if self._shutdown_event.is_set():
                    self._send_sdk_command(
                        GripperState.PAUSED.value,
                        0.0,
                        retries=0,
                        jitter=False)
                    goal_handle.abort()
                    self.logger.warning('Gripper action aborted during shutdown')
                    return self._finish_result(start_monotonic)

                current_state = self.gripper_state

                if (current_state in _VALID_GRIPPER_STATES and
                        current_state != last_feedback_state):
                    feedback = (
                        robomaster_msgs.action.GripperControl.Feedback())
                    feedback.current_state = current_state
                    goal_handle.publish_feedback(feedback)
                    last_feedback_state = current_state

                if current_state == target_state:
                    goal_handle.succeed()
                    self.logger.info(
                        'Gripper reached target after '
                        f'{(time.monotonic() - start_monotonic) * 1000.0:.0f} ms')
                    return self._finish_result(start_monotonic)

                now = time.monotonic()
                if now >= deadline:
                    self.logger.error(
                        f'Gripper did not reach target {target_state} within '
                        f'{self._action_timeout:.2f} s; current={current_state}')
                    self._send_sdk_command(
                        GripperState.PAUSED.value,
                        0.0,
                        retries=0,
                        jitter=False)
                    goal_handle.abort()
                    return self._finish_result(start_monotonic)

                # If the persistent stream died during a network disturbance,
                # restart it once during this goal instead of waiting forever.
                if (not resubscribe_attempted and self._status_is_stale() and
                        now - start_monotonic > 0.5):
                    resubscribe_attempted = True
                    self.logger.warning(
                        'No recent gripper status; attempting one resubscribe')
                    self._ensure_status_subscription()

                with self._state_condition:
                    remaining = deadline - time.monotonic()
                    self._state_condition.wait(
                        timeout=max(0.0, min(0.1, remaining)))

        except Exception as exc:
            # The action lock/reservation is released in finally even if ROS,
            # the SDK, or a publisher raises unexpectedly.
            self.logger.error(f'Unhandled gripper action error: {exc}')
            self._send_sdk_command(
                GripperState.PAUSED.value,
                0.0,
                retries=0,
                jitter=False)
            try:
                goal_handle.abort()
            except Exception:
                pass
            return self._finish_result(start_monotonic)
        finally:
            self._release_goal()

    def new_gripper_goal_callback(
            self,
            goal_request: robomaster_msgs.action.GripperControl.Goal
            ) -> rclpy.action.server.GoalResponse:
        target_state = int(goal_request.target_state)
        power = float(goal_request.power)

        if target_state not in _VALID_GRIPPER_STATES:
            self.logger.warning(
                f'Rejecting invalid gripper target state {target_state}')
            return rclpy.action.server.GoalResponse.REJECT

        if not math.isfinite(power) or power < 0.0 or power > 1.0:
            self.logger.warning(
                f'Rejecting invalid gripper power {power}')
            return rclpy.action.server.GoalResponse.REJECT

        with self._goal_mutex:
            if self._goal_reserved:
                self.logger.warning(
                    'Rejecting gripper goal because another goal is active')
                return rclpy.action.server.GoalResponse.REJECT

            self._goal_reserved = True
            self._cancel_event.clear()
            self._goal_done_event.clear()

        return rclpy.action.server.GoalResponse.ACCEPT

    def cancel_gripper_callback(
            self, goal_handle: Any) -> rclpy.action.CancelResponse:
        self.logger.info('Canceling gripper action requested')

        with self._goal_mutex:
            if not self._goal_reserved:
                return rclpy.action.CancelResponse.REJECT
            self._cancel_event.set()

        # Do not call the synchronous SDK pause() method from this callback.
        # The execute callback observes the event and pauses the gripper while
        # preserving SDK call serialization.
        with self._state_condition:
            self._state_condition.notify_all()

        return rclpy.action.CancelResponse.ACCEPT
