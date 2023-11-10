#!/bin/env python
from math import floor, tau
import rospy
import numpy as np

import geometry_msgs.msg
import sensor_msgs.msg
import vive_tracking_ros.msg
from vive_tracking_ros import math_utils

from tf_conversions import transformations as tr

from vive_tracking_ros.triad_openvr import triad_openvr


class ViveTrackingROS():
    def __init__(self) -> None:
        rospy.init_node("vive_tracking_ros")

        config_file = rospy.get_param("~config_file", None)

        self.vr = triad_openvr(configfile_path=config_file)

        self.topic_map = {}
        self.haptic_feedback_sub = rospy.Subscriber("/vive/set_feedback", vive_tracking_ros.msg.ViveControllerFeedback, self.haptic_feedback)

        publishing_rate = int(rospy.get_param("~publishing_rate", 100))
        self.pub_rate = rospy.Rate(publishing_rate)

        self.vr.print_discovered_objects()

    def run(self):

        update_devices_interval = 5  # seconds
        update_devices_start_time = rospy.get_time()

        while not rospy.is_shutdown():

            # Check for new controllers, remove disconnected ones
            if (rospy.get_time() - update_devices_start_time) > update_devices_interval:
                update_devices_start_time = rospy.get_time()
                self.vr.poll_vr_events()

            detected_controllers = self.vr.object_names["Controller"]

            for device_name in detected_controllers:

                self.publish_twist(device_name)
                self.publish_controller_input(device_name)

            # publishing rate
            self.pub_rate.sleep()

    def publish_controller_input(self, device_name):
        button_state_topic = self.topic_map.get(device_name, rospy.Publisher("/vive/" + device_name + "/joy", sensor_msgs.msg.Joy, queue_size=10))

        controller_inputs = self.vr.devices[device_name].get_controller_inputs()

        inputs_msg = sensor_msgs.msg.Joy()

        inputs_msg.buttons = [
            int(controller_inputs['menu_button']),
            int(controller_inputs['trackpad_pressed']),
            int(controller_inputs['trigger']*100),
            int(controller_inputs['grip_button']),
        ]

        button_state_topic.publish(inputs_msg)

    def publish_twist(self, device_name):
        # Controller velocity
        # x and z - diagonals, y - up/down
        linear_velocity = self.vr.devices[device_name].get_velocity()
        angular_velocity = self.vr.devices[device_name].get_angular_velocity()

        if linear_velocity is None or angular_velocity is None:
            return

        # Rotate twist to align with ROS world (x forward/backward, y right/left, z up/down)
        rotation = tr.quaternion_from_euler(0.0, tau/8, 0.0)
        rotation = math_utils.rotate_quaternion_by_rpy(-tau/4, tau/2, 0.0, rotation)

        linear_velocity = math_utils.quaternion_rotate_vector(rotation, linear_velocity[:])
        angular_velocity = math_utils.quaternion_rotate_vector(rotation, angular_velocity[:])

        twist_topic = self.topic_map.get(device_name, rospy.Publisher("/vive/" + device_name + "/twist", geometry_msgs.msg.Twist, queue_size=10))

        twist_msg = geometry_msgs.msg.Twist()
        twist_msg.linear = geometry_msgs.msg.Vector3(*linear_velocity)
        twist_msg.angular = geometry_msgs.msg.Vector3(*angular_velocity)

        twist_topic.publish(twist_msg)

    def haptic_feedback(self, msg: vive_tracking_ros.msg.ViveControllerFeedback):
        device = self.vr.devices.get(msg.controller_name, None)
        if device:
            # Intensity assumed to be between 0 and 1 inclusive
            duration_micros = int(np.clip(msg.duration_microsecs, 0.0, 3999.0))
            device.trigger_haptic_pulse(duration_micros=duration_micros)


if __name__ == '__main__':
    vive_tracking_ros = ViveTrackingROS()
    vive_tracking_ros.run()
