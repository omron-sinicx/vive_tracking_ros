#!/bin/env python
from math import floor
import rospy
import numpy as np

import geometry_msgs.msg
import sensor_msgs.msg

from tf_conversions import transformations as tr

from vive_tracking_ros.triad_openvr import triad_openvr


class ViveTrackingROS():
    def __init__(self) -> None:
        rospy.init_node("vive_tracking_ros")

        config_file = rospy.get_param("~config_file", None)
        
        self.vr = triad_openvr(configfile_path=config_file)

        self.topic_map = {}
        self.haptic_feedback_sub = rospy.Subscriber("/vive/feedback", sensor_msgs.msg.JoyFeedback, self.haptic_feedback)

        publishing_rate = int(rospy.get_param("~publishing_rate", 100))
        self.pub_rate = rospy.Rate(publishing_rate)

        self.vr.print_discovered_objects()

    def run(self):

        update_devices_interval = 5  # seconds
        update_devices_start_time = rospy.get_time()

        while not rospy.is_shutdown():
            # Check for new controllers, remove disconnected ones every 30 seconds

            if rospy.get_time() - update_devices_start_time > update_devices_interval:
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
            floor(controller_inputs['trigger']),
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
        rotation = tr.quaternion_from_euler(0.0, np.deg2rad(45), np.deg2rad(-90))
        rotation = rotate_quaternion_by_rpy(0.0, np.deg2rad(-90), 0.0, rotation)
        
        linear_velocity = quaternion_rotate_vector(rotation, linear_velocity[:])
        angular_velocity = quaternion_rotate_vector(rotation, angular_velocity[:])
        
        twist_topic = self.topic_map.get(device_name, rospy.Publisher("/vive/" + device_name + "/twist", geometry_msgs.msg.Twist, queue_size=10))

        twist_msg = geometry_msgs.msg.Twist()
        twist_msg.linear = geometry_msgs.msg.Vector3(*linear_velocity)
        twist_msg.angular = geometry_msgs.msg.Vector3(*angular_velocity)

        twist_topic.publish(twist_msg)

    def haptic_feedback(self, msg: sensor_msgs.msg.JoyFeedback):
        device = self.vr.devices.get(msg.id, None)
        if device:
            device.trigger_haptic_pulse(duration_micros=msg.intensity)


def quaternion_rotate_vector(quaternion, vector):
    """
        Return vector rotated by a given unit quaternion
        v' = q * v * q.conjugate() 
    """
    q_vector = np.append(vector, 0)
    return tr.quaternion_multiply(tr.quaternion_multiply(quaternion, q_vector), tr.quaternion_conjugate(quaternion))[:3]

def rotate_quaternion_by_rpy(roll, pitch, yaw, q_in, rotated_frame=False):
    """
    if rotated_frame == True, Apply RPY rotation in the reference frame of the quaternion.

    Otherwise, Apply RPY rotation in the rotated frame (the one to which the quaternion has rotated the reference frame).
    """
    q_rot = tr.quaternion_from_euler(roll, pitch, yaw)

    if rotated_frame:
        q_rotated = tr.quaternion_multiply(q_in, q_rot)
    else:
        q_rotated = tr.quaternion_multiply(q_rot, q_in)

    return q_rotated

if __name__ == '__main__':
    vive_tracking_ros = ViveTrackingROS()
    vive_tracking_ros.run()
