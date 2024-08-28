import math
import numpy as np
import quaternion
from tf_conversions import transformations as tr


def quaternion_from_matrix(matrix):
    q = quaternion.from_rotation_matrix(matrix[:3, :3])
    return from_np_quaternion(q)


def skew(v):
    """
    Returns the 3x3 skew matrix.
    The skew matrix is a square matrix M{A} whose transpose is also its
    negative; that is, it satisfies the condition M{-A = A^T}.
    @type v: array
    @param v: The input array
    @rtype: array, shape (3,3)
    @return: The resulting skew matrix
    """
    skv = np.roll(np.roll(np.diag(np.asarray(v).flatten()), 1, 1), -1, 0)
    return (skv - skv.T)


# def quaternions_orientation_error(quat_target, quat_source):
#     """
#     Calculates the orientation error between two quaternions
#     quat_target is the desired orientation
#     quat_source is the current orientation
#     both with respect to the same fixed frame

#     return vector part
#     """
#     qs = np.array(quat_source)
#     qt = np.array(quat_target)

#     ne = qs[3]*qt[3] + np.dot(qs[:3].T, qt[:3])
#     ee = qs[3]*np.array(qt[:3]) - qt[3]*np.array(qs[:3]) + np.dot(skew(qs[:3]), qt[:3])
#     ee *= np.sign(ne)  # disambiguate the sign of the quaternion
#     return ee
def orientation_error_as_rotation_vector(quat_target, quat_source):
    qt = to_np_quaternion(quat_target)
    qs = to_np_quaternion(quat_source)
    return quaternion.as_rotation_vector(qt*qs.conjugate())


def quaternions_orientation_error(quat_target, quat_source):
    qt = to_np_quaternion(quat_target)
    qs = to_np_quaternion(quat_source)
    return from_np_quaternion(qt*qs.conjugate())


def orientation_error_as_euler(quat_target, quat_source):
    """
    Calculates the orientation error between two quaternions
    quat_target is the desired orientation
    quat_source is the current orientation
    both with respect to the same fixed frame

    return vector part
    """
    qs = np.array(quat_source)
    qt = np.array(quat_target)

    ne = qs[3]*qt[3] + np.dot(qs[:3].T, qt[:3])
    ee = qs[3]*np.array(qt[:3]) - qt[3]*np.array(qs[:3]) + np.dot(skew(qs[:3]), qt[:3])
    ee *= np.sign(ne)  # disambiguate the sign of the quaternion
    return ee


def quaternion_multiply(quaternion1, quaternion0):
    q1 = to_np_quaternion(quaternion1)
    q0 = to_np_quaternion(quaternion0)
    return from_np_quaternion(q1*q0)


def integrate_unit_quaternion_DMM(q, w, dt):
    """ Integrate a unit quaternion using the Direct Multiplication Method"""
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return q
    q_tmp = np.concatenate([np.sin(w_norm*dt/2)*w/w_norm, [np.cos(w_norm*dt/2.)]])
    return quaternion_multiply(q_tmp, q)


def integrate_unit_quaternion_euler(q, w, dt):
    """ Integrate a unit quaterniong using Euler Method"""
    qw = np.append(w, 0)
    return (q + quaternion_multiply(qw, q)*0.5*dt)


def normalize_quaternion(quat):
    return from_np_quaternion(to_np_quaternion(quat).normalized())


def quaternion_rotate_vector(quat, vector):
    """
        Return vector rotated by a given unit quaternion
    """
    q = to_np_quaternion(quat)
    return quaternion.rotate_vectors(q, vector)


def rotate_quaternion_by_delta(axis_angle, q_in, rotated_frame=False):
    np_q_rot = quaternion.from_rotation_vector(axis_angle)
    np_q_in = to_np_quaternion(q_in)

    if rotated_frame:
        q_rotated = np_q_in*np_q_rot
    else:
        q_rotated = np_q_rot*np_q_in

    return from_np_quaternion(q_rotated)


def rotate_quaternion_by_rpy(roll, pitch, yaw, q_in, rotated_frame=False):
    """
    if rotated_frame == True, Apply RPY rotation in the reference frame of the quaternion.

    Otherwise, Apply RPY rotation in the rotated frame (the one to which the quaternion has rotated the reference frame).
    """
    np_q_rot = to_np_quaternion(tr.quaternion_from_euler(roll, pitch, yaw))
    np_q_in = to_np_quaternion(q_in)

    if rotated_frame:
        q_rotated = np_q_in*np_q_rot
    else:
        q_rotated = np_q_rot*np_q_in

    return from_np_quaternion(q_rotated)


def ortho6_from_quaternion(quat):
    R = quaternion.as_rotation_matrix(to_np_quaternion(quat))
    return R[:3, :2].T.flatten()


def axis_angle_from_quaternion(quat):
    return quaternion.as_rotation_vector(to_np_quaternion(quat))


def axis2quat(axis):
    np_q = quaternion.from_rotation_vector(axis)
    return np.array([np_q.x, np_q.y, np_q.z, np_q.w])


def to_np_quaternion(q):
    """ Create a numpy quaternion W, X, Y, Z """
    return np.quaternion(q[3], q[0], q[1], q[2])


def from_np_quaternion(np_q):
    return np.array([np_q.x, np_q.y, np_q.z, np_q.w])
