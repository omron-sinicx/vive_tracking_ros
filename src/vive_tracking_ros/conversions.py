import numpy as np

from geometry_msgs.msg import (Point, Quaternion)


def to_quaternion(array):
    """
  Converts a numpy array into a C{geometry_msgs/Quaternion} ROS message.
  @type  array: np.array
  @param array: The position as numpy array
  @rtype: geometry_msgs/Quaternion
  @return: The resulting ROS message
  """
    return Quaternion(*array)


def to_point(array):
    """
  Converts a numpy array into a C{geometry_msgs/Point} ROS message.
  @type  array: np.array
  @param array: The position as numpy array
  @rtype: geometry_msgs/Point
  @return: The resulting ROS message
  """
    return Point(*array)


def from_quaternion(msg):
    """
  Converts a C{geometry_msgs/Quaternion} ROS message into a numpy array.
  @type  msg: geometry_msgs/Quaternion
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
    return np.array([msg.x, msg.y, msg.z, msg.w], dtype=float)


def from_vector3(msg):
    """
  Converts a C{geometry_msgs/Vector3} ROS message into a numpy array.
  @type  msg: geometry_msgs/Vector3
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
    return np.array([msg.x, msg.y, msg.z], dtype=float)

# ROS types <--> Numpy types


def from_point(msg):
    """
  Converts a C{geometry_msgs/Point} ROS message into a numpy array.
  @type  msg: geometry_msgs/Point
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
    return from_vector3(msg)


def from_wrench(msg):
    """
  Converts a C{geometry_msgs/Wrench} ROS message into a numpy array.
  @type  msg: geometry_msgs/Wrench
  @param msg: The ROS message to be converted
  @rtype: np.array
  @return: The resulting numpy array
  """
    array = np.zeros(6)
    array[:3] = from_vector3(msg.force)
    array[3:] = from_vector3(msg.torque)
    return array
