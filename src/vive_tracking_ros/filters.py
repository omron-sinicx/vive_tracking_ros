import collections
import numpy as np
from tf import transformations as tr


class QuaternionLowPassFilter:
    def __init__(self, fraction):
        self.fraction = fraction
        self.last_quaternion = None

    def update(self, new_quaternion):
        if self.last_quaternion is None:
            self.last_quaternion = new_quaternion
            return new_quaternion

        # Spherical linear interpolation (SLERP)
        filtered_quaternion = tr.quaternion_slerp(self.last_quaternion, new_quaternion, self.fraction)

        self.last_quaternion = filtered_quaternion
        return filtered_quaternion


class LowPassFilter:
    def __init__(self, window_size):
        self.buffer = collections.deque(maxlen=window_size)

    def update(self, data):
        self.buffer.append(data)
        return np.mean(self.buffer, axis=0)


class PoseLowPassFilter:
    def __init__(self, alpha, window_size) -> None:
        self.pos_filter = LowPassFilter(window_size)
        self.quat_filter = QuaternionLowPassFilter(alpha)

    def update(self, pose):
        filtered_pose = np.zeros_like(pose)
        filtered_pose[:3] = self.pos_filter.update(pose[:3])
        filtered_pose[3:] = self.quat_filter.update(pose[3:])
        return pose
