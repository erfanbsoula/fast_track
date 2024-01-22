import time
import dynamixel_sdk as dxl
from Motor import Dynamixel_MX_106, Dynamixel_MX_64, MotorGroup
from AgileEye import ikp
import numpy as np

# default setting
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600

# HOME_POSITIONS = np.array([2583, 2065, 2064])
HOME_POSITIONS = np.array([2583, 2065, 2064])

order = [0, 1, 2]

portHandler = dxl.PortHandler(DEVICENAME)

if portHandler.openPort():
    print("succeeded to open the port")
else:
    print("failed to open the port")
    quit()

if portHandler.setBaudRate(BAUDRATE):
    print("succeeded to change the baudrate")
else:
    print("failed to change the baudrate")
    quit()

motors = []
home_pos = []

for idx in order:
    
    if idx == 0:
        motor = Dynamixel_MX_64(portHandler, 1, HOME_POSITIONS[0])
    
    elif idx == 1:
        motor = Dynamixel_MX_106(portHandler, 2, HOME_POSITIONS[1])

    else:
        motor = Dynamixel_MX_106(portHandler, 3, HOME_POSITIONS[2])
    
    motors.append(motor)
    home_pos.append(HOME_POSITIONS[idx])


motors = MotorGroup(motors)
home_pos = np.array(home_pos)

# time.sleep(2)
# motors.setGoalPositions(list(HOME_POSITIONS+ikp(0, -30, 0)))
# time.sleep(2)

# while True:
#     for angle in range(-30, 31, 2):
#         motors.setGoalPositions(list(HOME_POSITIONS+ikp(0, angle, 0)))
#         time.sleep(0.005)

#     time.sleep(0.1)
    
#     for angle in range(30, -31, -2):
#         motors.setGoalPositions(list(HOME_POSITIONS+ikp(0, angle, 0)))
#         time.sleep(0.005)

# motors.setGoalPositions(list(HOME_POSITIONS+ikp(0, 0, 0)))

while True:
    
    time.sleep(3)
    motors.setGoalPositions(list(home_pos + ikp(0, 30, 0)))
    time.sleep(3)
    motors.setGoalPositions(list(home_pos + ikp(0,-30, 0)))

# motors.disableTorque()

# Close port
portHandler.closePort()