from abc import ABC, abstractmethod
from typing import List, Type
import dynamixel_sdk as dxl

SPEED = 15

class DynamixelMotor(ABC):

    def __init__(self, portHandler, dxlId, homePosition):
        self.PORT_HANDLER = portHandler
        self.DXL_ID = dxlId
        self.HOME_POSITION = homePosition
        self.packetHandler = dxl.PacketHandler(self.PROTOCOL_VERSION)

        if self.enableTorque():
            print(f"Dynamixel {dxlId} has been successfully connected")

        if self.setProfileVelocity(self.PROFILE_VELOCITY):
            print(f"Dynamixel {dxlId} profile velocity is set")

        if self.setGoalPosition(self.HOME_POSITION):
            print(f"Dynamixel {dxlId} moveing to home position ...")

    @property
    @abstractmethod
    def PROTOCOL_VERSION(self):
        pass

    @property
    @abstractmethod
    def PROFILE_VELOCITY(self):
        pass
    
    def checkResults(self, dxl_comm_result, dxl_error):

        if dxl_comm_result != dxl.COMM_SUCCESS:
            print(self.packetHandler.getTxRxResult(dxl_comm_result))
            return False

        elif dxl_error != 0:
            print(self.packetHandler.getRxPacketError(dxl_error))
            return False

        return True

    def write1Byte(self, address, value):
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.PORT_HANDLER, self.DXL_ID, address, value)

        return self.checkResults(dxl_comm_result, dxl_error)

    def write2Byte(self, address, value):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.PORT_HANDLER, self.DXL_ID, address, value)

        return self.checkResults(dxl_comm_result, dxl_error)

    def write4Byte(self, address, value):
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.PORT_HANDLER, self.DXL_ID, address, value)

        return self.checkResults(dxl_comm_result, dxl_error)

    def read2Byte(self, address):
        result, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(
            self.PORT_HANDLER, self.DXL_ID, address)

        if self.checkResults(dxl_comm_result, dxl_error):
            return result

        else: return None

    def read4Byte(self, address):
        result, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.PORT_HANDLER, self.DXL_ID, address)

        if self.checkResults(dxl_comm_result, dxl_error):
            return result

        else: return None
    
    @abstractmethod
    def enableTorque(self):
        pass

    @abstractmethod
    def disableTorque(self):
        pass

    @abstractmethod
    def setGoalPosition(self, angle):
        pass

    @abstractmethod
    def getPresentPosition(self):
        pass

    @abstractmethod
    def setProfileVelocity(self, velocity):
        pass


class Dynamixel_MX_106(DynamixelMotor):
    
    PROTOCOL_VERSION = 1.0

    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PRO_VELOCITY = 112

    VELOCITY_UNIT_SCALE = 0.229 # rpm
    PROFILE_VELOCITY = int(SPEED * 60 / 360 / VELOCITY_UNIT_SCALE)

    def __init__(self, portHandler, dxlId, homePosition):
                super().__init__(portHandler, dxlId, homePosition)

    def enableTorque(self):
        return self.write1Byte(self.ADDR_TORQUE_ENABLE, 1)

    def disableTorque(self):
        return self.write1Byte(self.ADDR_TORQUE_ENABLE, 0)

    def setGoalPosition(self, angle):
        return self.write4Byte(self.ADDR_GOAL_POSITION, angle)

    def getPresentPosition(self):
        return self.read4Byte(self.ADDR_GOAL_POSITION)

    def setProfileVelocity(self, velocity):
        return self.write4Byte(self.ADDR_PRO_VELOCITY, velocity)


class Dynamixel_MX_64(DynamixelMotor):
    
    PROTOCOL_VERSION = 1.0

    ADDR_TORQUE_ENABLE = 24
    ADDR_GOAL_POSITION = 30
    ADDR_PRESENT_POSITION = 36
    ADDR_PRO_VELOCITY = 32

    VELOCITY_UNIT_SCALE = 0.114 # rpm
    PROFILE_VELOCITY = int(SPEED * 60 / 360 / VELOCITY_UNIT_SCALE)

    def __init__(self, portHandler, dxlId, homePosition):
        super().__init__(portHandler, dxlId, homePosition)

    def enableTorque(self):
        return self.write1Byte(self.ADDR_TORQUE_ENABLE, 1)

    def disableTorque(self):
        return self.write1Byte(self.ADDR_TORQUE_ENABLE, 0)

    def setGoalPosition(self, angle):
        return self.write2Byte(self.ADDR_GOAL_POSITION, angle)

    def getPresentPosition(self):
        return self.read2Byte(self.ADDR_GOAL_POSITION)

    def setProfileVelocity(self, velocity):
        return self.write2Byte(self.ADDR_PRO_VELOCITY, velocity)


class MotorGroup():

    def __init__(self, motors: List[Type[DynamixelMotor]] = []):
        self.motors = motors

    def add(self, motor: Type[DynamixelMotor]):
        self.motors.append(motor)

    def disableTorque(self):
        for motor in self.motors:
            motor.disableTorque()

    def setGoalPositions(self, angles: List[int]):

        if len(angles) != len(self.motors):
            return False

        result = True
        for i, motor in enumerate(self.motors):
            result = result and motor.setGoalPosition(angles[i])

        return result

    def getPresentPositions(self):

        results = []
        for motor in self.motors:
            results.append(motor.getPresentPosition())

        return results