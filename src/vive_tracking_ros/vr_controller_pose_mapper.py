#!/usr/bin/env python3

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

    def __init__(self, config_filepath):
        self.load_params(config_filepath)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.last_tf_stamp = rospy.get_rostime()

        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])
        self.robot_center_position = np.zeros(3)
        self.robot_center_orientation = np.array([0, 0, 0, 1])
        self.controller_center_position = np.zeros(3)
        self.controller_center_orientation = np.array([0, 0, 0, 1])
        self.target_gripper_pose = 0.0  # Fully open

        self.enable_controller_inputs = True
        # when tracking is pause, the current pose of the robot is published as the target pose
        self.pause_tracking = False

        # Set the initial target pose to the current pose
        if not self.center_target_pose():
            rospy.logerr("Fail to get robot's end-effector pose")
            sys.exit(0)

        vive_twist_topic = '/vive/' + self.controller_name + '/twist'
        vive_pose_topic = '/vive/' + self.controller_name + '/pose'
        vive_joy_topic = '/vive/' + self.controller_name + '/joy'
        haptic_feedback_topic = '/vive/set_feedback'

        # Publishers
        self.haptic_feedback_last_stamp = rospy.get_time()
        self.haptic_feedback_pub = rospy.Publisher(haptic_feedback_topic, ControllerHapticCommand, queue_size=3)

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

    def load_params(self, config_filepath):
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)

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
        self.play_area = config['safety']['play_area']
        self.play_area[3:] = np.deg2rad(self.play_area[3:])

        # Scale down tracking when using tracking mode = 'controller_twist'
        self.scale_velocities = config['safety']['scale_velocities']
        self.scale_velocities = np.clip(self.scale_velocities, 0.0, 1.0)

        # Topics
        self.wrench_topic = config['topics'].get('wrench', None)

        rospy.loginfo(f"Tracking mode: {self.tracking_mode}")

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
        robot_current_pose = self.get_transformation(source=self.end_effector_frame, target=self.robot_frame)
        controller_current_pose = self.get_transformation(source=self.controller_name, target="vive_world")

        if not robot_current_pose or not controller_current_pose:
            rospy.logwarn("Failed to get transformation from robot current pose")
            return False

        self.target_position = conversions.from_point(robot_current_pose.transform.translation)
        self.target_orientation = conversions.from_quaternion(robot_current_pose.transform.rotation)

        self.robot_center_position = np.copy(self.target_position)
        self.robot_center_orientation = np.copy(self.target_orientation)

        self.controller_center_position = conversions.from_point(controller_current_pose.transform.translation)
        self.controller_center_orientation = conversions.from_quaternion(controller_current_pose.transform.rotation)

        return True

    def vive_joy_cb(self, data: Joy):
        if not self.enable_controller_inputs:
            return

        app_menu_button = data.buttons[0]
        trigger_button = data.buttons[2]

        if app_menu_button:
            # re-center the target pose
            if not self.center_target_pose():
                sys.exit(0)
            self.pause_tracking = not self.pause_tracking  # Pause/Resume tracking

            rospy.sleep(0.5)

        if trigger_button:  # just track target pose for gripper
            self.target_gripper_pose = max(0.0, 1.-trigger_button/100.0)  # In percentage

    def vive_pose_cb(self, data: PoseStamped):
        """
            Compute the target pose based on the absolute pose of the vive controller
        """
        if self.pause_tracking:
            return self.publish_robot_current_pose()

        controller_position = conversions.from_point(data.pose.position)
        controller_orientation = conversions.from_quaternion(data.pose.orientation)

        delta_translation = controller_position - self.controller_center_position
        delta_rotation = math_utils.quaternions_orientation_error(controller_orientation, self.controller_center_orientation)*2
        if self.world_frame:  # Rotate to a common frame of reference before applying delta
            delta_translation = math_utils.quaternion_rotate_vector(self.world_to_robot_rotation, delta_translation)
            delta_rotation = math_utils.quaternion_rotate_vector(self.world_to_robot_rotation, delta_rotation)

        # rospy.loginfo_throttle(1, f"diff {np.round(delta_translation, 2)}")
        # rospy.loginfo_throttle(1, f"diff {np.round(np.rad2deg(delta_rotation), 2)}")

        if np.any(np.abs(delta_translation) > self.play_area[:3]) or np.any(np.abs(delta_rotation) > self.play_area[3:]):

            for i in range(3):
                if np.abs(delta_translation)[i] > self.play_area[i]:
                    delta_translation[i] = math.copysign(self.play_area[i], delta_translation[i])

                if np.abs(delta_rotation)[i] > self.play_area[i+3]:
                    delta_rotation[i] = math.copysign(self.play_area[i+3], delta_rotation[i])

        self.target_position = self.robot_center_position + delta_translation
        self.target_orientation = math_utils.rotate_quaternion_by_rpy(*delta_rotation, self.robot_center_orientation)

        self.broadcast_pose_to_tf()

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
        delta_rotation = math_utils.quaternions_orientation_error(next_orientation, self.robot_center_orientation)
        # rospy.loginfo_throttle(1, f"diff {np.round(angular_vel, 4)}  {np.round(np.rad2deg(delta_rotation), 2)}")

        if np.any(np.abs(delta_translation) > self.play_area[:3]) or np.any(np.abs(delta_rotation) > self.play_area[3:]):
            for i in range(3):
                if np.abs(delta_translation)[i] > self.play_area[i]:
                    delta_translation[i] = math.copysign(self.play_area[i], delta_translation[i])

                if np.abs(delta_rotation)[i] > self.play_area[i+3]:
                    delta_rotation[i] = math.copysign(self.play_area[i+3], delta_rotation[i])

            self.target_position = self.robot_center_position + delta_translation
            self.target_orientation = math_utils.rotate_quaternion_by_rpy(*delta_rotation, self.robot_center_orientation)
        else:
            self.target_position = next_pose
            self.target_orientation = next_orientation

        self.broadcast_pose_to_tf()

    def publish_robot_current_pose(self):
        self.target_position = np.copy(self.robot_center_position)
        self.target_orientation = np.copy(self.robot_center_orientation)
        self.broadcast_pose_to_tf()

    def wrench_cb(self, data: WrenchStamped):
        # Only uses the sum of forces, so the orientation is irrelevant
        wrench = conversions.from_wrench(data.wrench)

        if self.sensor_frame != self.end_effector_frame:
            forces = wrench[:3] + spalg.sensor_torque_to_tcp_force(sensor_torques=wrench[3:], tcp_position=self.sensor_to_eef_translation)
        else:
            forces = wrench[:3]

        total_force = np.sum(np.abs(forces))

        force_sensitivity = [3.0, 50.0]  # Min and Max force to map to vibration intensity

        if total_force > force_sensitivity[0] \
                and rospy.get_time() - self.haptic_feedback_last_stamp > 0.075:  # Avoid sending too many haptic commands

            haptic_msg = ControllerHapticCommand()
            haptic_msg.controller_name = self.controller_name
            haptic_msg.duration_microsecs = np.interp(total_force, force_sensitivity, [0.0, 3999.0])

            # rospy.loginfo(f"{round(haptic_msg.duration_microsecs, 2)} {np.round(total_force, 1)}")

            # self.haptic_feedback_pub.publish(haptic_msg)
            self.haptic_feedback_last_stamp = rospy.get_time()

    def broadcast_pose_to_tf(self):
        if self.last_tf_stamp == rospy.Time.now():
            rospy.logdebug("Ignoring request to publish TF, not enough time has passed.")
            return

        child_frame = "vr_target_pose" if not self.robot_ns else self.robot_ns + "_vr_target_pose"

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.robot_frame
        t.child_frame_id = child_frame
        t.transform.translation = conversions.to_point(self.target_position)
        t.transform.rotation = conversions.to_quaternion(self.target_orientation)

        try:
            self.tf_broadcaster.sendTransform(t)
        except rospy.ROSException:
            pass

        self.last_tf_stamp = t.header.stamp
