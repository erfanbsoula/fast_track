# run this script from the base directory
import sys
sys.path.append(".")

import time
import AgileEye

start_position = AgileEye.HOME_POSITION + AgileEye.solve_inverse_kinematics(0, 0, 10)
end_position = AgileEye.HOME_POSITION + AgileEye.solve_inverse_kinematics(0, 0, -10)

DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600

robot = AgileEye.initialize(start_position)

for i in range(4):

    robot.setGoalPosition(start_position)
    time.sleep(3)
    robot.setGoalPosition(end_position)
    time.sleep(3)