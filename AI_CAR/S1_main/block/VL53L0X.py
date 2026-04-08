# S1_main 내장용 - VL53L0X 거리센서 (라즈베리 파이 전용)
# vl53l0x_python.so를 이 폴더에 두세요
import os
from ctypes import *

VL53L0X_GOOD_ACCURACY_MODE = 0
VL53L0X_BETTER_ACCURACY_MODE = 1
VL53L0X_BEST_ACCURACY_MODE = 2

tof_lib = None
i2cbus = None

try:
    import smbus
    i2cbus = smbus.SMBus(1)

    def i2c_read(address, reg, data_p, length):
        ret_val = 0
        try:
            result = i2cbus.read_i2c_block_data(address, reg, length)
            for i in range(length):
                data_p[i] = result[i]
        except IOError:
            ret_val = -1
        return ret_val

    def i2c_write(address, reg, data_p, length):
        ret_val = 0
        try:
            data = [data_p[i] for i in range(length)]
            i2cbus.write_i2c_block_data(address, reg, data)
        except IOError:
            ret_val = -1
        return ret_val

    _dir = os.path.dirname(os.path.abspath(__file__))
    _so = os.path.join(_dir, 'vl53l0x_python.so')
    if os.path.isfile(_so):
        tof_lib = CDLL(_so)
    elif os.path.isfile("/home/pi/autonomousCar/block/vl53l0x_python.so"):
        tof_lib = CDLL("/home/pi/autonomousCar/block/vl53l0x_python.so")

    if tof_lib is not None:
        READFUNC = CFUNCTYPE(c_int, c_ubyte, c_ubyte, POINTER(c_ubyte), c_ubyte)
        WRITEFUNC = CFUNCTYPE(c_int, c_ubyte, c_ubyte, POINTER(c_ubyte), c_ubyte)
        tof_lib.VL53L0X_set_i2c(READFUNC(i2c_read), WRITEFUNC(i2c_write))
except Exception:
    pass  # Windows 또는 Pi에 센서 없을 때

class VL53L0X(object):
    object_number = 0

    def __init__(self, address=0x29, TCA9548A_Num=255, TCA9548A_Addr=0, **kwargs):
        self.device_address = address
        self.TCA9548A_Device = TCA9548A_Num
        self.TCA9548A_Address = TCA9548A_Addr
        self.my_object_number = VL53L0X.object_number
        VL53L0X.object_number += 1

    def start_ranging(self, mode=VL53L0X_GOOD_ACCURACY_MODE):
        if tof_lib is None:
            raise RuntimeError('VL53L0X .so not loaded')
        tof_lib.startRanging(self.my_object_number, mode, self.device_address,
                             self.TCA9548A_Device, self.TCA9548A_Address)

    def stop_ranging(self):
        if tof_lib is not None:
            tof_lib.stopRanging(self.my_object_number)

    def get_distance(self):
        return tof_lib.getDistance(self.my_object_number) if tof_lib else 0
