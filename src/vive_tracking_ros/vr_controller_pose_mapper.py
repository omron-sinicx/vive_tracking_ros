#!/usr/bin/env python3

import collections
import math
import numpy as np
import rospy
import tf2_ros
import yaml
import sys

from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Joy

from ur_control import spalg
from ur_control.constants import JOINT_TRAJECTORY_CONTROLLER, get_arm_joint_names
from ur_control.controllers import JointTrajectoryController
from ur_pykdl import ur_kinematics

from vive_tracking_ros.msg import ControllerHapticCommand
from vive_tracking_ros import conversions, math_utils


class VRControllerPoseMapper:
    """ Converter of VR HTC controllers to the robot's end effector target poses and gripper pose

    Two control modes:
        Twist-based: follow the relative trajectory defined by the controllers reported twist
        Pose-based: follow the relative trajectory defined by the estimated pose of the controller

    In both case, the trajectory is relative to a *center pose* of the robot. The center pose
    is define as the position of the robot's end effector when the tracking is (re)started.

    """

    def __init__(self, config_params):
        self.load_params(config_params)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.last_tf_stamp = rospy.get_rostime()

        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])
        self.robot_center_position = np.zeros(3)
        self.robot_center_orientation = np.array([0, 0, 0, 1])
        self.current_controller_position = np.zeros(3)
        self.current_controller_orientation = np.array([0, 0, 0, 1])
        self.controller_center_position = np.zeros(3)
        self.controller_center_orientation = np.array([0, 0, 0, 1])
        self.target_gripper_pose = 1.0  # Fully open

        # when tracking is pause, the current pose of the robot is published as the target pose
        self.pause_tracking = True
        self.grip_button_switch = False  # information only
        self.trackpad_pressed = False  # information only
        self.last_button_state = {}

        # self.arm = arm.Arm(namespace=self.robot_ns, gripper_type=None, joint_names_prefix=f'{self.robot_ns}_')
        self.arm_controller = JointTrajectoryController(publisher_name=JOINT_TRAJECTORY_CONTROLLER,
                                                        namespace=self.robot_ns,
                                                        joint_names=get_arm_joint_names(f'{self.robot_ns}_'),
                                                        timeout=1.0)

        if rospy.has_param("robot_description"):
            self.kdl = ur_kinematics(base_link=self.robot_frame, ee_link=self.end_effector_frame)
        else:
            raise ValueError("robot_description not found in the parameter server")

        # Set the initial target pose to the current pose
        if not self.center_target_pose():
            rospy.logerr("Fail to get robot's end-effector pose.\n Is the VR controller being tracked?")
            sys.exit(0)

        vive_twist_topic = '/vive/' + self.controller_name + '/twist'
        vive_pose_topic = '/vive/' + self.controller_name + '/pose'
        vive_joy_topic = '/vive/' + self.controller_name + '/joy'
        haptic_feedback_topic = '/vive/set_feedback'

        # Publishers
        self.haptic_feedback_last_stamp = rospy.get_time()
        self.haptic_feedback_pub = rospy.Publisher(haptic_feedback_topic, ControllerHapticCommand, queue_size=1)

        # Subscribers
        if self.tracking_mode == "controller_pose":
            rospy.Subscriber(vive_pose_topic, PoseStamped, self.vive_pose_cb, queue_size=1)
        elif self.tracking_mode == "controller_twist":
            rospy.Subscriber(vive_twist_topic, TwistStamped, self.vive_twist_cb, queue_size=1)
        else:
            raise ValueError(f'Invalid tracking mode "{self.tracking_mode}". Valid modes are: [controller_pose, controller_twist]')

        self.load_static_transforms()

        # Start tracking
        rospy.Subscriber(vive_joy_topic, Joy, self.vive_joy_cb, queue_size=1)
        if self.wrench_topic:
            rospy.Subscriber(self.wrench_topic, WrenchStamped, self.wrench_cb, queue_size=1)

        self.wrench_queue = collections.deque(maxlen=50)

    def load_static_transforms(self):
        controller2robot = self.get_transformation(source=self.controller_frame, target=self.robot_frame)
        if not controller2robot:
            assert ValueError("Failed to compute transform between the robot base frame and the controller")
        self.controller_to_robot_rotation = conversions.from_quaternion(controller2robot.transform.rotation)

        if self.world_frame:
            world2robot = self.get_transformation(source=self.world_frame, target=self.robot_frame)
            if not world2robot:
                assert ValueError("Failed to compute transform between the world frame and the robot base frame")
            self.world_to_robot_rotation = conversions.from_quaternion(world2robot.transform.rotation)

        if self.sensor_frame != self.end_effector_frame:
            sensor2eef_transform = self.get_transformation(self.sensor_frame, self.end_effector_frame)
            if not controller2robot:
                assert ValueError("Failed to compute transform between the ft sensor and the robot's end effector frame")
            self.sensor_to_eef_translation = conversions.from_point(sensor2eef_transform.transform.translation)

    def load_params(self, config_params):
        if isinstance(config_params, str):
            with open(config_params, 'r') as f:
                config = yaml.safe_load(f)
        elif isinstance(config_params, dict):
            config = config_params

        # Robot params
        self.robot_ns = config['robot'].get('namespace', None)
        self.robot_frame = config['robot']['base_frame']
        self.end_effector_frame = config['robot']['end_effector_frame']
        self.sensor_frame = config['robot']['ft_sensor_frame']
        self.world_frame = config['robot'].get('world_frame', None)

        # Controller params
        self.controller_name = config['controller']['name']
        self.controller_frame = config['controller']['frame_id']
        self.tracking_mode = config['controller']['tracking_mode']
        if self.tracking_mode not in ('controller_pose', 'controller_twist'):
            raise ValueError(f'Invalid tracking mode "{self.tracking_mode}". Valid modes are: [controller_pose, controller_twist]')

        # Safety params
        # Limit the displacement to the play area
        self.play_area = np.array(config['safety']['play_area'])
        self.play_area[3:] = np.deg2rad(self.play_area[3:])

        # Scale down tracking when using tracking mode = 'controller_twist'
        self.scale_velocities = config['safety']['scale_velocities']
        self.scale_velocities = np.clip(self.scale_velocities, 0.0, 1.0)

        # Scale down translation when using tracking mode = 'controller_pose'
        self.scale_translation = config['safety']['scale_translation']
        self.scale_translation = np.clip(self.scale_translation, 0.0, 1.0)

        # Delta limits from robot's current pose to target pose
        self.max_delta_translation = config['safety']['max_delta_translation']
        self.max_delta_rotation = np.deg2rad(config['safety']['max_delta_rotation'])

        # Topics
        self.wrench_topic = config['topics'].get('wrench', None)

        rospy.loginfo(f"Tracking mode: {self.tracking_mode}")

    def reset(self):
        self.pause_tracking = True
        self.grip_button_switch = False
        self.trackpad_pressed = False
        self.target_gripper_pose = 1.0

    def get_transformation(self, source, target):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=target, source_frame=source, time=rospy.Time(0), timeout=rospy.Duration(5))

        except (tf2_ros.InvalidArgumentException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)
            return False

    def center_target_pose(self):
        """
            Reset the reference position as the current pose of the robot's end effector
        """

        robot_current_pose = self.kdl.forward(self.arm_controller.get_joint_positions())

        self.target_position = robot_current_pose[:3]
        self.target_orientation = robot_current_pose[3:]

        self.robot_center_position = np.copy(self.target_position)
        self.robot_center_orientation = np.copy(self.target_orientation)

        self.controller_center_position = np.copy(self.current_controller_position)
        self.controller_center_orientation = np.copy(self.current_controller_orientation)

        return True

    def vive_joy_cb(self, data: Joy):
        app_menu_button = data.buttons[0]
        trackpad_button = data.buttons[1]

        trigger_button = data.buttons[2]
        grip_button = data.buttons[3]

        if trackpad_button and not self.last_button_state.get('trackpad_button', False):
            self.trackpad_pressed = not self.trackpad_pressed
            self.last_button_state['trackpad_button'] = trackpad_button
        else:
            self.last_button_state['trackpad_button'] = trackpad_button

        if app_menu_button and not self.last_button_state.get('app_menu_button', False):
            # re-center the target pose
            self.center_target_pose()
            self.pause_tracking = not self.pause_tracking  # Pause/Resume tracking

            self.last_button_state['app_menu_button'] = app_menu_button
        else:
            self.last_button_state['app_menu_button'] = app_menu_button

        if grip_button and not self.last_button_state.get('grip_button', False):
            self.grip_button_switch = not self.grip_button_switch
            self.last_button_state['grip_button'] = grip_button
        else:
            self.last_button_state['grip_button'] = grip_button

        if trigger_button:  # just track target pose for gripper
            self.target_gripper_pose = max(0.0, 1.0 - trigger_button/100.0)  # In percentage

    def vive_pose_cb(self, data: PoseStamped):
        """
            Compute the target pose based on the absolute pose of the vive controller
        """
        self.current_controller_position = conversions.from_point(data.pose.position)
        self.current_controller_orientation = conversions.from_quaternion(data.pose.orientation)

        if self.pause_tracking:
            return self.publish_robot_current_pose()

        # Compute relative translation/rotation from controller center position
        delta_translation = self.current_controller_position - self.controller_center_position
        delta_rotation = math_utils.orientation_error_as_euler(self.current_controller_orientation, self.controller_center_orientation)*2
        if self.world_frame:  # Rotate to a common frame of reference before applying delta
            delta_translation = math_utils.quaternion_rotate_vector(self.world_to_robot_rotation, delta_translation)
            delta_rotation = math_utils.quaternion_rotate_vector(self.world_to_robot_rotation, delta_rotation)

        # rospy.loginfo_throttle(1, f"diff {np.round(delta_translation, 2)}")
        # rospy.loginfo_throttle(1, f"diff {np.round(np.rad2deg(delta_rotation), 2)}")

        # Limit translation/rotation to the defined play_area
        delta_translation = np.clip(delta_translation, -self.play_area[:3], self.play_area[:3])
        delta_rotation = np.clip(delta_rotation, -self.play_area[3:], self.play_area[3:])

        # Scale down controller translation
        if np.any(self.scale_translation != 1.0):
            delta_translation *= self.scale_translation[:3]
            delta_rotation *= self.scale_translation[3:]

        delta_translation, delta_rotation = self.enforce_max_delta(delta_translation, delta_rotation)

        self.target_position = self.robot_center_position + delta_translation
        # self.target_orientation = math_utils.rotate_quaternion_by_delta(delta_rotation, self.robot_center_orientation)
        self.target_orientation = math_utils.rotate_quaternion_by_rpy(*delta_rotation, self.robot_center_orientation)

        self.broadcast_pose_to_tf(self.target_position, self.target_orientation)

    def vive_twist_cb(self, data: TwistStamped):
        """ Numerically integrate twist message into a pose

        Use global self.frame_id as reference for the navigation commands.
        """
        if self.pause_tracking:
            return self.publish_robot_current_pose()

        if not hasattr(self, 'last'):
            self.last = rospy.get_time()
            return

        now = rospy.get_time()
        dt = now - self.last
        self.last = now

        if math.isclose(dt, 0.0):  # wait for a significant time difference
            return

        linear_vel = conversions.from_vector3(data.twist.linear) * self.scale_velocities[:3]
        angular_vel = conversions.from_vector3(data.twist.angular) * self.scale_velocities[3:]

        # transform to robot base frame
        linear_vel = math_utils.quaternion_rotate_vector(self.controller_to_robot_rotation, linear_vel)
        angular_vel = math_utils.quaternion_rotate_vector(self.controller_to_robot_rotation, angular_vel)

        # Position update
        next_pose = self.target_position + (linear_vel * dt)

        # Orientation update
        next_orientation = math_utils.integrate_unit_quaternion_DMM(self.target_orientation, angular_vel, dt)

        delta_translation = next_pose - self.robot_center_position
        delta_rotation = math_utils.orientation_error_as_euler(next_orientation, self.robot_center_orientation)*2
        # rospy.loginfo_throttle(1, f"diff {np.round(angular_vel, 4)}  {np.round(np.rad2deg(delta_rotation), 2)}")

        if np.any(np.abs(delta_translation) > self.play_area[:3]) or np.any(np.abs(delta_rotation) > self.play_area[3:]):
            delta_translation = np.clip(delta_translation, -self.play_area[:3], self.play_area[:3])
            delta_rotation = np.clip(delta_rotation, -self.play_area[3:], self.play_area[3:])

            self.target_position = self.robot_center_position + delta_translation
            self.target_orientation = math_utils.rotate_quaternion_by_delta(delta_rotation, self.robot_center_orientation)
        else:
            self.target_position = next_pose
            self.target_orientation = next_orientation

        self.broadcast_pose_to_tf(self.target_position, self.target_orientation)

    def publish_robot_current_pose(self):
        self.target_position = self.robot_center_position
        self.target_orientation = self.robot_center_orientation
        self.broadcast_pose_to_tf(self.target_position, self.target_orientation)

    def wrench_cb(self, data: WrenchStamped):
        # Only uses the sum of forces, so the orientation is irrelevant
        self.wrench_queue.append(conversions.from_wrench(data.wrench))

        if self.pause_tracking:
            return

        wrench = np.average(self.wrench_queue, axis=0)

        if self.sensor_frame != self.end_effector_frame:
            forces = wrench[:3] + spalg.sensor_torque_to_tcp_force(sensor_torques=wrench[3:], tcp_position=self.sensor_to_eef_translation)
        else:
            forces = wrench[:3]

        total_force = np.sum(np.abs(forces))

        force_sensitivity = [3.0, 50.0]  # Min and Max force to map to vibration intensity

        if total_force > force_sensitivity[0] \
                and rospy.get_time() - self.haptic_feedback_last_stamp > 0.1:  # Avoid sending too many haptic commands

            haptic_msg = ControllerHapticCommand()
            haptic_msg.controller_name = self.controller_name
            haptic_msg.duration_microsecs = np.interp(total_force, force_sensitivity, [0.0, 3999.0])

            # rospy.loginfo(f"{round(haptic_msg.duration_microsecs, 2)} {np.round(total_force, 1)}")

            try:
                self.haptic_feedback_pub.publish(haptic_msg)
                self.haptic_feedback_last_stamp = rospy.get_time()
            except rospy.ROSException:
                pass

    def broadcast_pose_to_tf(self, position, orientation, name="vr_target_pose"):
        if self.last_tf_stamp == rospy.Time.now():
            rospy.logdebug("Ignoring request to publish TF, not enough time has passed.")
            return

        child_frame = name if not self.robot_ns else f"{self.robot_ns}_{name}"

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.robot_frame
        t.child_frame_id = child_frame
        t.transform.translation = conversions.to_point(position)
        t.transform.rotation = conversions.to_quaternion(orientation)

        try:
            self.tf_broadcaster.sendTransform(t)
        except rospy.ROSException:
            pass

        self.last_tf_stamp = t.header.stamp

    def enforce_max_delta(self, delta_translation, delta_rotation):
        """
            Make sure that the absolute error between the current pose of the robot and the target pose
            given by the controller is less or equal to `max_delta_translation` and `max_delta_rotation`.

            returns bounded deltas
        """

        robot_current_pose = self.kdl.forward(self.arm_controller.get_joint_positions())

        # distance from the starting (center) pose to the robot's current pose
        current_translation = robot_current_pose[:3] - self.robot_center_position
        current_rotation = math_utils.orientation_error_as_euler(robot_current_pose[3:], self.robot_center_orientation)*2

        # pose from the starting (center) pose to the controller's current pose
        temp_target_position = self.robot_center_position + delta_translation
        temp_target_orientation = math_utils.rotate_quaternion_by_rpy(*delta_rotation, self.robot_center_orientation)

        # distance from the robot's current pose to the controller's current pose
        error_translation = temp_target_position - robot_current_pose[:3]
        error_rotation = math_utils.orientation_error_as_euler(temp_target_orientation, robot_current_pose[3:])*2

        if np.any(np.abs(error_translation) > self.max_delta_translation) \
                or np.any(np.abs(error_rotation) > self.max_delta_rotation):
            self.broadcast_pose_to_tf(temp_target_position, temp_target_orientation, name="unconstrained_target_pose")

            # apply limits
            error_translation = np.clip(error_translation, -self.max_delta_translation, self.max_delta_translation)
            error_rotation = np.clip(error_rotation, -self.max_delta_rotation, self.max_delta_rotation)

            delta_translation = current_translation + error_translation
            delta_rotation = (current_rotation + error_rotation)

        return delta_translation, delta_rotation
