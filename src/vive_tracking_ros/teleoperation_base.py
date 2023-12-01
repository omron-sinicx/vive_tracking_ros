#!/usr/bin/env python3

import math
import numpy as np
import rospy
import threading
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Joy
import tf2_ros
import sys
from ur_control import spalg

from vive_tracking_ros.msg import ViveControllerFeedback
from vive_tracking_ros import conversions, math_utils


class TeleoperationBase:
    """ Convert Twist messages to PoseStamped

    Use this node to integrate twist messages into a moving target pose in
    Cartesian space.  An initial TF lookup assures that the target pose always
    starts at the robot's end-effector.
    """

    def __init__(self):
        self.load_params()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.last_tf_stamp = rospy.get_rostime()

        self.vive_to_robot_rotation = conversions.from_quaternion(self.get_transformation(source=self.vive_base_frame, target=self.robot_frame).transform.rotation)
        if self.world_frame:
            self.world_to_robot_rotation = conversions.from_quaternion(self.get_transformation(source=self.world_frame, target=self.robot_frame).transform.rotation)

        self.target_position = np.zeros(3)
        self.target_orientation = np.array([0, 0, 0, 1])
        self.robot_center_position = np.zeros(3)
        self.robot_center_orientation = np.array([0, 0, 0, 1])
        self.controller_center_position = np.zeros(3)
        self.controller_center_orientation = np.array([0, 0, 0, 1])

        self.enable_teleoperation_condition = threading.Condition()
        self.enable_teleoperation = False

        self.enable_controller_inputs = True

        # Set the intial target pose to the current pose
        if not self.center_target_pose():
            rospy.logerr("Fail to get robot's end-effector pose")
            sys.exit(0)

        pose_topic = rospy.get_param('~pose_topic', default="my_pose")
        wrench_topic = rospy.get_param('~wrench_topic', default="/wrench")
        haptic_feedback_topic = rospy.get_param('~haptic_feedback_topic', default="/vive/set_feedback")
        vive_twist_topic = '/vive/' + self.controller_name + '/twist'
        vive_pose_topic = '/vive/' + self.controller_name + '/pose'
        vive_joy_topic = '/vive/' + self.controller_name + '/joy'

        # Publishers
        self.target_pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=3)
        self.haptic_feedback_last_stamp = rospy.get_time()
        self.haptic_feedback_pub = rospy.Publisher(haptic_feedback_topic, ViveControllerFeedback, queue_size=3)

        # Subscribers
        if self.tracking_mode == "controller_pose":
            rospy.Subscriber(vive_pose_topic, PoseStamped, self.vive_pose_cb, queue_size=1)
        elif self.tracking_mode == "controller_twist":
            rospy.Subscriber(vive_twist_topic, TwistStamped, self.vive_twist_cb, queue_size=1)
        else:
            raise ValueError(f'Invalid tracking mode "{self.tracking_mode}". Valid modes are: [controller_pose, controller_twist]')

        if self.sensor_frame != self.end_effector_frame:
            sensor2eef_transform = self.get_transformation(self.sensor_frame, self.end_effector_frame)
            self.sensor2eef_traslation = conversions.from_point(sensor2eef_transform.transform.translation)

        rospy.Subscriber(vive_joy_topic, Joy, self.vive_joy_cb, queue_size=1)
        rospy.Subscriber(wrench_topic, WrenchStamped, self.wrench_cb, queue_size=1)

    def load_params(self):
        self.robot_ns = rospy.get_param('~robot_namespace', default="")
        self.robot_frame = rospy.get_param('~robot_base_link', default="base_link")
        self.end_effector_frame = rospy.get_param('~robot_end_effector_link', default="tool0")
        self.sensor_frame = rospy.get_param('~sensor_link', default=self.end_effector_frame)
        self.world_frame = rospy.get_param('~world_frame', default=None)

        self.vive_base_frame = rospy.get_param('~vive_frame_id', default="world")
        self.controller_name = rospy.get_param('~controller_name', default="right_controller")

        # Limit the displacement to the play area
        self.play_area = rospy.get_param('~play_area', [0.05, 0.05, 0.05, 15, 15, 15])
        self.play_area[3:] = np.deg2rad(self.play_area[3:])

        # Limit contact interaction
        self.max_force_torque = rospy.get_param('~max_contact_force_torque', default=[50., 50., 50., 5., 5., 5.])

        self.scale_velocities = rospy.get_param('~scale_velocities', [1., 1., 1., 1., 1., 1.])
        self.scale_velocities = np.clip(self.scale_velocities, 0.0, 1.0)

        self.tracking_mode = rospy.get_param('~tracking_mode', "controller_pose")

        self.visualization_only = rospy.get_param('~visualization_only', False)

        if self.tracking_mode not in ('controller_pose', 'controller_twist'):
            raise ValueError(f'Invalid tracking mode "{self.tracking_mode}". Valid modes are: [controller_pose, controller_twist]')

        rospy.loginfo(f"Teleoperation mode: {self.tracking_mode}")
        rospy.loginfo(f"Teleoperation visualization only: {self.visualization_only}")

    def get_transformation(self, source, target):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=target, source_frame=source, time=rospy.Time(0), timeout=rospy.Duration(5))

        except (tf2_ros.InvalidArgumentException, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)
            return False

    def center_target_pose(self):
        robot_current_pose = self.get_transformation(source=self.end_effector_frame, target=self.robot_frame)
        controller_current_pose = self.get_transformation(source=self.controller_name, target="vive_world")

        if not robot_current_pose or not controller_current_pose:
            rospy.logwarn("Failed to get transformation")
            return False

        self.target_position = conversions.from_point(robot_current_pose.transform.translation)
        self.target_orientation = conversions.from_quaternion(robot_current_pose.transform.rotation)

        self.robot_center_position = self.target_position
        self.robot_center_orientation = self.target_orientation

        self.controller_center_position = conversions.from_point(controller_current_pose.transform.translation)
        self.controller_center_orientation = conversions.from_quaternion(controller_current_pose.transform.rotation)

        self.last = rospy.get_time()

        return True

    def vive_pose_cb(self, data: PoseStamped):

        if not self.enable_teleoperation:
            return

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

        if not self.enable_teleoperation:
            return

        now = rospy.get_time()
        dt = now - self.last
        self.last = now

        dt = dt

        linear_vel = conversions.from_vector3(data.twist.linear) * self.scale_velocities[:3]
        angular_vel = conversions.from_vector3(data.twist.angular) * self.scale_velocities[3:]

        # transform to robot base frame
        linear_vel = math_utils.quaternion_rotate_vector(self.vive_to_robot_rotation, linear_vel)
        angular_vel = math_utils.quaternion_rotate_vector(self.vive_to_robot_rotation, angular_vel)

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

    def vive_joy_cb(self, data: Joy):
        if not self.enable_controller_inputs:
            return

        app_menu_button = data.buttons[0]

        if app_menu_button:
            # re-center the target pose
            if not self.center_target_pose():
                sys.exit(0)

            # Enable/Disable tracking
            self.set_teleoperation_status(enable=(not self.enable_teleoperation))

            rospy.sleep(0.5)

    def set_teleoperation_status(self, enable=True):
        with self.enable_teleoperation_condition:
            self.enable_teleoperation = enable
            if self.enable_teleoperation:
                rospy.loginfo("=== Tracking Enabled  ===")
            else:
                rospy.loginfo("=== Tracking Disabled ===")
            self.enable_teleoperation_condition.notify_all()

    def wrench_cb(self, data: WrenchStamped):
        if not self.enable_teleoperation:
            return

        # Only uses the sum of forces, so the orientation is irrelevant
        wrench = conversions.from_wrench(data.wrench)

        if self.sensor_frame != self.end_effector_frame:
            forces = wrench[:3] + spalg.sensor_torque_to_tcp_force(sensor_torques=wrench[3:], tcp_position=self.sensor2eef_traslation)
        else:
            forces = wrench[:3]

        total_force = np.sum(np.abs(forces))

        force_sensitivity = [3.0, 50.0]  # Min and Max force to map to vibration intensity

        if total_force > force_sensitivity[0] \
                and rospy.get_time() - self.haptic_feedback_last_stamp > 0.075:  # Avoid sending too many haptic commands

            haptic_msg = ViveControllerFeedback()
            haptic_msg.controller_name = self.controller_name
            haptic_msg.duration_microsecs = np.interp(total_force, force_sensitivity, [0.0, 3999.0])

            # rospy.loginfo(f"{round(haptic_msg.duration_microsecs, 2)} {np.round(total_force, 1)}")

            self.haptic_feedback_pub.publish(haptic_msg)
            self.haptic_feedback_last_stamp = rospy.get_time()

        if np.any(np.abs(wrench) > self.max_force_torque):
            self.enable_teleoperation = False
            rospy.logwarn("Tracking stopped, excessive contact force detected: %s" % (np.round(wrench, 1)))

    def publish_target_pose(self, target_position, target_orientation):
        if not rospy.is_shutdown():
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.robot_frame
            msg.pose.position = conversions.to_point(target_position)
            msg.pose.orientation = conversions.to_quaternion(target_orientation)

            # rospy.loginfo_throttle(1, msg)

            try:
                self.target_pose_pub.publish(msg)
            except rospy.ROSException:
                # Swallow 'publish() to closed topic' error.
                # This rarely happens on killing this node.
                pass

    def broadcast_pose_to_tf(self):
        if self.last_tf_stamp == rospy.Time.now():
            rospy.logdebug("Ignoring request to publish TF, not enough time has passed.")
            return

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.robot_frame
        t.child_frame_id = self.robot_ns + "_vr_target_pose"
        t.transform.translation = conversions.to_point(self.target_position)
        t.transform.rotation = conversions.to_quaternion(self.target_orientation)

        self.tf_broadcaster.sendTransform(t)

        self.last_tf_stamp = t.header.stamp
