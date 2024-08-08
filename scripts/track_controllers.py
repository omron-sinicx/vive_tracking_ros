#!/bin/env python
import openvr
from math import tau
import numpy as np
import rospy
import tf2_ros
import threading

import geometry_msgs.msg
import sensor_msgs.msg
from vive_tracking_ros.filters import PoseLowPassFilter, QuaternionLowPassFilter
import vive_tracking_ros.msg
from vive_tracking_ros import math_utils, conversions

from tf import transformations as tr

from vive_tracking_ros.triad_openvr import triad_openvr


class ViveTrackingROS():
    """
        Use this node to publish the pose and twist tracked from VR controllers using openVR.
        The controller inputs are also published as sensor_msgs.msg.Joy messages
    """

    def __init__(self) -> None:
        rospy.init_node("vive_tracking_ros")

        config_file = rospy.get_param("~config_file", None)

        self.vr = triad_openvr(configfile_path=config_file)

        self.topic_map = {}
        self.haptic_feedback_sub = rospy.Subscriber("/vive/set_feedback", vive_tracking_ros.msg.ControllerHapticCommand, self.haptic_feedback)

        publishing_rate = int(rospy.get_param("~publishing_rate", 100))
        self.pub_rate = rospy.Rate(publishing_rate)

        device_events_rate = int(rospy.get_param("~device_events_rate", 1000))
        self.dev_pub_rate = rospy.Rate(device_events_rate)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.last_tf_stamp_dict = {}

        self.vr.print_discovered_objects()

        self.is_quat_flipped = {}

        # keep previous orientation to check quaternion sign
        publishing_rate = rospy.get_param("~filter_pose", True)
        self.pose_filter = {}
        self.quaternion_filter_fraction = rospy.get_param("~quaternion_filter_alpha", 0.1)
        self.position_filter_window_size = int(rospy.get_param("~position_filter_window_size", 10))

    def run(self):

        controller_input_thread = threading.Thread(target=self.wait_for_controller_events)
        controller_input_thread.start()

        while not rospy.is_shutdown():

            detected_controllers = self.vr.object_names["Controller"]

            for device_name in detected_controllers:
                self.publish_twist(device_name)

                pose = self.compute_device_pose(device_name)
                if pose is not None:
                    self.publish_controller_pose(device_name, pose)
                    self.broadcast_pose_to_tf(device_name, pose)

            # publishing rate
            self.pub_rate.sleep()

        controller_input_thread.join()

    def wait_for_controller_events(self):
        """
            Listen for controller input events.
            Only publish the controller state when an event is received. 
            If the trigger or touch_pad is being pressed, publish the controller state continuously.
        """
        controller_map = {}

        event = openvr.VREvent_t()
        while not rospy.is_shutdown():

            while self.vr.vrsystem.pollNextEvent(event):
                if event.eventType in [openvr.VREvent_ButtonPress, openvr.VREvent_ButtonUnpress]:
                    # print("button pressed", event.trackedDeviceIndex, event.data.controller.button)
                    self.publish_controller_input(self.vr.device_index_map[event.trackedDeviceIndex])
                    if event.data.controller.button == 33 or event.data.controller.button == 32:
                        controller_map[event.trackedDeviceIndex] = {event.data.controller.button: True}

                if event.eventType == openvr.VREvent_ButtonUnpress:
                    # print("button unpressed", event.trackedDeviceIndex, event.data.controller.button)
                    controller_map[event.trackedDeviceIndex] = {event.data.controller.button: False}

                # Include events related to new devices been activated or deactivated
                # to automatically add/remove them to the pool of devices
                if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                    self.vr.add_tracked_device(event.trackedDeviceIndex)
                elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                    # If we were already tracking this device, quit tracking it.
                    if event.trackedDeviceIndex in self.vr.device_index_map:
                        if self.vr.vr.getTrackedDeviceClass(event.trackedDeviceIndex) == openvr.TrackedDeviceClass_TrackingReference:
                            # If the lighthouses are disconnected, check quat flipped again
                            self.is_quat_flipped = {}
                        self.vr.remove_tracked_device(event.trackedDeviceIndex)

            self.dev_pub_rate.sleep()

            # Keep publishing the controller status if the button pressed is the Trigger or the touchpad
            for controller_id, buttons in controller_map.items():
                for button_status in buttons.values():
                    if button_status:
                        self.publish_controller_input(self.vr.device_index_map[controller_id])

    def publish_controller_input(self, device_name):
        device = self.vr.get_device(device_name)
        if not device:
            return

        button_state_topic = self.topic_map.get(device_name, rospy.Publisher("/vive/" + device_name + "/joy", sensor_msgs.msg.Joy, queue_size=10))

        controller_inputs = device.get_controller_inputs()

        inputs_msg = sensor_msgs.msg.Joy()

        trigger_pose = np.interp(controller_inputs['trigger'], [0.25, 0.85], [0.0, 1.0])

        inputs_msg.buttons = [
            int(controller_inputs['menu_button']),
            int(controller_inputs['trackpad_pressed']),
            int(trigger_pose*100),
            int(controller_inputs['grip_button']),
        ]

        button_state_topic.publish(inputs_msg)

    def publish_twist(self, device_name):
        device = self.vr.get_device(device_name)
        if not device:
            return

        # Controller velocity
        # x and z - diagonals, y - up/down
        linear_velocity = device.get_velocity()
        angular_velocity = device.get_angular_velocity()

        if linear_velocity is None or angular_velocity is None:
            return

        # Rotate twist to align with ROS world (x forward/backward, y right/left, z up/down)
        rotation = tr.quaternion_from_euler(0.0, tau/8, 0.0)
        rotation = math_utils.rotate_quaternion_by_rpy(-tau/4, tau/2, 0.0, rotation)

        linear_velocity = math_utils.quaternion_rotate_vector(rotation, linear_velocity[:])
        angular_velocity = math_utils.quaternion_rotate_vector(rotation, angular_velocity[:])

        twist_topic = self.topic_map.get(device_name, rospy.Publisher("/vive/" + device_name + "/twist", geometry_msgs.msg.TwistStamped, queue_size=10))

        twist_msg = geometry_msgs.msg.TwistStamped()
        twist_msg.header.frame_id = device_name
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.twist.linear = geometry_msgs.msg.Vector3(*linear_velocity)
        twist_msg.twist.angular = geometry_msgs.msg.Vector3(*angular_velocity)

        twist_topic.publish(twist_msg)

    def compute_device_pose(self, device_name):
        device = self.vr.get_device(device_name)
        if not device:
            return

        # Get controller pose from openvr
        pose = device.get_pose_quaternion()

        if pose is None:
            return None

        # Apply filter to raw device pose
        if self.pose_filter.get(device_name, None) is None:
            self.pose_filter[device_name] = PoseLowPassFilter(self.quaternion_filter_fraction, self.position_filter_window_size)
        pose = self.pose_filter[device_name].update(pose)

        # rospy.loginfo_throttle(1, f"device {device_name} q:{np.round(pose[3:],4)}")

        if self.is_quat_flipped.get(device_name, None) is None:
            error_unflipped = np.linalg.norm(math_utils.orientation_error_as_rotation_vector(pose[3:], [0.125, 0.724, -0.0191, 0.6781]))
            error_flipped = np.linalg.norm(math_utils.orientation_error_as_rotation_vector(pose[3:], [0.0101, -0.8328, 0.138, 0.5359]))

            self.is_quat_flipped[device_name] = error_unflipped > error_flipped
            rospy.loginfo(f"Quat error unflipped: {error_unflipped}")
            rospy.loginfo(f"Quat error flipped: {error_flipped}")
            rospy.loginfo(f"Is quat flipped?: {self.is_quat_flipped}")

        # Rotate twist to align with ROS world (x forward/backward, y right/left, z up/down)
        # TODO(cambel): how to have a unified orientation? maybe check basestation orientation?
        if not self.is_quat_flipped.get(device_name):
            # 1nd orientation
            rotation = tr.quaternion_from_euler(0.0, -tau/2, 0.0)
            rotation = math_utils.rotate_quaternion_by_rpy(tau/4, 0.0, 0, rotation)
            rotation = math_utils.rotate_quaternion_by_rpy(0.0, 0.0, tau/2, rotation)
        else:
            # 2nd orientation
            rotation = tr.quaternion_from_euler(0.0, 0.0, tau/2)
            rotation = math_utils.rotate_quaternion_by_rpy(-tau/4, 0.0, 0, rotation)
            rotation = math_utils.rotate_quaternion_by_rpy(0.0, 0.0, tau/16, rotation)

        pose[:3] = math_utils.quaternion_rotate_vector(rotation, pose[:3])
        pose[3:] = math_utils.normalize_quaternion(math_utils.quaternion_multiply(rotation, pose[3:]))

        return pose

    def publish_controller_pose(self, device_name, pose):  # relative to the base stations

        pose_topic = self.topic_map.get(device_name, rospy.Publisher("/vive/" + device_name + "/pose", geometry_msgs.msg.PoseStamped, queue_size=10))

        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.frame_id = "vive_world"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position = conversions.to_point(pose[:3])
        pose_msg.pose.orientation = conversions.to_quaternion(pose[3:])

        pose_topic.publish(pose_msg)

    def broadcast_pose_to_tf(self, device_name, pose):
        if self.last_tf_stamp_dict.get(device_name, None) \
                and self.last_tf_stamp_dict[device_name] == rospy.Time.now():
            rospy.logerr("Ignoring request to publish TF, not enough time has passed.")
            return

        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "vive_world"
        t.child_frame_id = device_name
        t.transform.translation = geometry_msgs.msg.Vector3(*pose[:3])
        t.transform.rotation = conversions.to_quaternion(pose[3:])

        self.tf_broadcaster.sendTransform(t)
        self.last_tf_stamp_dict[device_name] = rospy.get_rostime().to_sec()

    def haptic_feedback(self, msg: vive_tracking_ros.msg.ControllerHapticCommand):
        device = self.vr.get_device(msg.controller_name)
        if not device:
            return

        # Intensity assumed to be between 0 and 1 inclusive
        duration_micros = int(np.clip(msg.duration_microsecs, 0.0, 3999.0))
        device.trigger_haptic_pulse(duration_micros=duration_micros)


if __name__ == '__main__':
    vive_tracking_ros = ViveTrackingROS()
    vive_tracking_ros.run()
