import numpy as np
from math import radians, degrees, sin, cos, atan, sqrt
from scipy.spatial.transform import Rotation

# References:
# 1. Gosselin, C. M. and Gagne, M. (1995).
#    A Closed-Form Solution for the Direct Kinematics of a Special Class of
#    Spherical Three-Degree-of-Freedom Parallel Manipulators.
#    In: Computational Kinematics 95.


# Standard unit vectors of the reference frame
# in reference coordinate system
r1_R = np.array([[1], [0], [0]], dtype=np.float64)
r2_R = np.array([[0], [1], [0]], dtype=np.float64)
r3_R = np.array([[0], [0], [1]], dtype=np.float64)

# Unit vectors along the axes of the actuators
# in reference coordinate system
u1_R = r1_R
u2_R = r3_R
u3_R = r2_R

# Considering (vi)s as unit vectors along the axes attached to the platform,
# vi_0 shows the vector vi at reference configuration (initial value of vi)
# in reference coordinate system
v1_0_R = -u3_R
v2_0_R = -u1_R
v3_0_R = -u2_R

# Standard unit vectors of the reference frame
# in world coordinate system
r1_W = np.array([[ sqrt(2/3)], [         0], [-sqrt(1/3)]], dtype=np.float64)
r2_W = np.array([[-sqrt(1/6)], [-sqrt(1/2)], [-sqrt(1/3)]], dtype=np.float64)
r3_W = np.array([[-sqrt(1/6)], [ sqrt(1/2)], [-sqrt(1/3)]], dtype=np.float64)

# Coordinate transformation matrix
# from reference coordinate system to world coordinate system
T_R2W = np.concatenate((r1_W, r2_W, r3_W), axis=1, dtype=np.float64)

# Coordinate transformation matrix
# from world coordinate system to reference coordinate system
T_W2R = T_R2W.T

# Rotation matrix around the x-axis of the coordinate system
def Rx(xrad):

    return np.array(

        [[  1,          0,           0],
         [  0,  cos(xrad),  -sin(xrad)],
         [  0,  sin(xrad),   cos(xrad)]],

        dtype=np.float64
    )

# Rotation matrix around the y-axis of the coordinate system
def Ry(yrad):

    return np.array(

        [[  cos(yrad),  0,  sin(yrad)],
         [          0,  1,          0],
         [ -sin(yrad),  0,  cos(yrad)]],

        dtype=np.float64
    )

# Rotation matrix around the z-axis of the coordinate system
def Rz(zrad):

    return np.array(

        [[  cos(zrad), -sin(zrad),  0],
         [  sin(zrad),  cos(zrad),  0],
         [          0,          0,  1]],

        dtype=np.float64
    )


def solve_inverse_kinematics(zdeg, ydeg, xdeg):

    zrad, yrad, xrad = radians(zdeg), radians(ydeg), radians(xdeg)

    # Rotation matrix in world coordinate system
    rot_W = np.matmul(Rx(xrad), np.matmul(Ry(yrad), Rz(zrad)))

    # Rotation matrix in reference coordinate system
    rot_R = np.matmul(T_W2R, np.matmul(rot_W, T_R2W))

    # Value of vi after applying the rotation
    v1 = np.matmul(rot_R, v1_0_R)
    v2 = np.matmul(rot_R, v2_0_R)
    v3 = np.matmul(rot_R, v3_0_R)

    # Rotation of each actuator around its axis
    theta1 = atan(v1[2]/v1[1])
    theta2 = atan(v2[1]/v2[0])
    theta3 = atan(v3[0]/v3[2])

    return np.array([degrees(theta1), degrees(theta2), degrees(theta3)])


def solve_forward_kinematics(theta1, theta2, theta3):

    theta1, theta2, theta3 = radians(theta1), radians(theta2), radians(theta3)

    # Calculate euler angles associated respectively with
    # each of the successive rotations
    phi1 = -theta2

    C1 = cos(phi1)*sin(phi1)*sin(theta1)*cos(theta3) + cos(theta1)*sin(theta3)
    C2 = cos(phi1)*cos(theta1)*cos(theta3) - sin(phi1)*sin(theta1)*sin(theta3)
    phi2 = atan(-C1/C2)

    A1 = cos(phi1)*sin(theta1)
    B1 = cos(phi2)*cos(theta1) - sin(phi1)*sin(phi2)*sin(theta1)
    phi3 = atan(-A1/B1)

    s1, s2, s3 = sin(phi1), sin(phi2), sin(phi3)
    c1, c2, c3 = cos(phi1), cos(phi2), cos(phi3)

    # Rotation matrix in reference coordinate system
    rot_R = np.array(

        [[ c1*c2,  c1*s2*s3+s1*c3, -c1*s2*c3+s1*s3],
         [-s1*c2, -s1*s2*s3+c1*c3,  s1*s2*c3+c1*s3],
         [    s2,          -c2*s3,           c2*c3]],

        dtype=np.float64
    )

    # Rotation matrix in world coordinate system
    rot_W = np.matmul(T_R2W, np.matmul(rot_R, T_W2R))
    rotation =  Rotation.from_matrix(rot_W)

    # Rotation angles around axes of the world coordinate system
    angles = rotation.as_euler("zyx", degrees=True)
    return angles
