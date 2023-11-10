import math
import numpy as np
from tf_conversions import transformations as tr
import rospy


def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


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


def quaternions_orientation_error(quat_target, quat_source):
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


def rotation_between_quaternions(q1, q2):
    R1 = tr.quaternion_matrix(q1)
    R2 = tr.quaternion_matrix(q2)

    R_rel = np.dot(R2, np.linalg.inv(R1))
    euler_angles = tr.euler_from_matrix(R_rel)

    return np.array(euler_angles)


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    """
    return np.array((-quaternion[0], -quaternion[1],
                     -quaternion[2], quaternion[3]), dtype=np.float64)


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> numpy.allclose(q, [-44, -14, 48, 28])
    True

    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)


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


def normalize_quaternion(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


def quaternion_rotate_vector(quaternion, vector):
    """
        Return vector rotated by a given unit quaternion
    """
    q_vector = np.append(vector, 0)
    return quaternion_multiply(quaternion_multiply(quaternion, q_vector), quaternion_conjugate(quaternion))[:3]


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


def is_unit_quaternion(q, tolerance=1e-6):
    # Calculate the magnitude of the quaternion
    magnitude = np.linalg.norm(q)

    # Check if the magnitude is close to 1 within the specified tolerance
    return np.isclose(magnitude, 1.0, rtol=tolerance)
