import numpy as np
from math import sin, cos, pi, atan, degrees, radians
from copy import deepcopy

u1 = np.array([[1], [0], [0]])
u2 = np.array([[0], [1], [0]])
u3 = np.array([[0], [0], [1]])

v1_0 = deepcopy(-u3)
v2_0 = deepcopy(-u1)
v3_0 = deepcopy(-u2)

e1 = np.array([[-1], [-1], [-1]])
e1 = e1/np.linalg.norm(e1)
e2 = np.array([[-1], [.5], [.5]])
e2 = e2/np.linalg.norm(e2)
e3 = np.cross(e1.reshape(3,), e2.reshape(3,))
e3 = e3.reshape(3,1)

def ikp(zdeg, ydeg, xdeg):

     Q1 = np.dot(e1,e1.T) + cos(radians(zdeg))*( np.eye(3) - np.dot(e1,e1.T)) + \
          sin(radians(zdeg))*np.array([[0,-e1.item(2),e1.item(1)], [e1.item(2),0,-e1.item(0)], [-e1.item(1),e1.item(0),0]])

     Q2 = np.dot(e2,e2.T) + cos(radians(ydeg))*( np.eye(3) - np.dot(e2,e2.T)) + \
          sin(radians(ydeg))*np.array([[0,-e2.item(2),e2.item(1)], [e2.item(2),0,-e2.item(0)], [-e2.item(1),e2.item(0),0]])

     Q3 = np.dot(e3,e3.T) + cos(radians(xdeg))*( np.eye(3) - np.dot(e3,e3.T)) + \
          sin(radians(xdeg))*np.array([[0,-e3.item(2),e3.item(1)], [e3.item(2),0,-e3.item(0)], [-e3.item(1),e3.item(0),0]])

     v1_f = np.dot(Q2,np.dot(Q3,np.dot(Q1, v1_0)))
     v2_f = np.dot(Q2,np.dot(Q3,np.dot(Q1, v2_0)))
     v3_f = np.dot(Q2,np.dot(Q3,np.dot(Q1, v3_0)))

     zdeg1 = atan(v1_f[1]/v1_f[2])
     zdeg2 = atan(v2_f[2]/v2_f[0])
     zdeg3 = atan(v3_f[0]/v3_f[1])

     rotation = np.array([degrees(zdeg1), degrees(zdeg2), degrees(zdeg3)]) * 4096/360

     return rotation.astype(int)