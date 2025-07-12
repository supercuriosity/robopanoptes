import atexit
import logging
import time
from typing import Optional, Sequence, Union, Tuple

import numpy as np
import serial

# 假设 Feetech 协议常量（你需要根据实际文档调整）
ADDR_TORQUE_ENABLE = 0x40       # 假定地址：0x40，用于使能/禁止力矩
ADDR_GOAL_POSITION = 0x74       # 假定地址：0x74，用于设置目标位置
LEN_GOAL_POSITION = 2           # 假定数据长度2个字节
# 对于低速舵机，单位可能不同，这里假设输入单位为度

# 假设一个简单包格式:
# [Header(0xFF,0xFF), ID, LENGTH, COMMAND, DATA..., CHECKSUM]
# 下面的命令号定义为：
CMD_SET_GOAL_POSITION = 0x03
CMD_SET_TORQUE_ENABLE = 0x04

class FeetechDriver:
    """
    Feetech 舵机通信驱动示例（Feetech 协议需参考具体文档进行修改）。
    提供 set_torque_enabled, sync_write, write_desired_pos, write_byte 等接口，
    接口设计与 DynamixelDriver 保持一致，以方便上层调用（例如 snake_agent）。
    """
    def __init__(self,
                 motor_ids: Sequence[int],
                 port: str = "/dev/ttyUSB0",
                 baudrate: int = 115200,
                 timeout: float = 0.1):
        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self._sync_writers = {}  # 简单模拟的 group sync 写入缓存

    def connect(self):
        self.serial = serial.Serial(port=self.port_name,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout)
        logging.info("Connected to Feetech servos on port %s", self.port_name)

    @property
    def is_connected(self) -> bool:
        return self.serial is not None and self.serial.is_open

    def disconnect(self):
        if self.is_connected:
            # 关闭前可以禁用所有舵机
            self.set_torque_enabled(self.motor_ids, False)
            self.serial.close()
            self.serial = None
            logging.info("Disconnected from Feetech servos on port %s", self.port_name)

    def check_connected(self):
        if not self.is_connected:
            raise OSError("Not connected. Please call connect() first.")

    def send_packet(self, packet: bytes):
        self.check_connected()
        self.serial.write(packet)
        # 根据需要可添加读取响应的逻辑
        time.sleep(0.005)

    def build_packet(self, motor_id: int, command: int, data: bytes) -> bytes:
        """
        构建数据包，格式: Header(0xFF, 0xFF) + motor_id (1B) + LENGTH (1B) + command (1B) + data + checksum (1B)
        checksum 为除 Header 外所有字节的和的低 8 位取反。
        """
        header = bytes([0xFF, 0xFF])
        id_byte = bytes([motor_id])
        length = 1 + len(data) + 1  # command + data + checksum
        length_byte = bytes([length])
        command_byte = bytes([command])
        packet_without_checksum = id_byte + length_byte + command_byte + data
        checksum = (~(sum(packet_without_checksum)) & 0xFF)
        checksum_byte = bytes([checksum])
        packet = header + packet_without_checksum + checksum_byte
        return packet

    def write_byte(self, motor_ids: Sequence[int], value: int, address: int) -> Sequence[int]:
        """
        对于 Feetech 舵机，假定写入一个字节的命令使用自定义 command CMD_SET_TORQUE_ENABLE(或其他)
        此处 address 参数可以用作标识不同命令。
        返回写入失败的 motor id 列表。
        """
        self.check_connected()
        errored_ids = []
        # 这里简单将 address 当作区分命令的依据
        for motor_id in motor_ids:
            # 根据 address 选择命令：
            if address == ADDR_TORQUE_ENABLE:
                command = CMD_SET_TORQUE_ENABLE
                # data 包含一个字节的目标值
                data = bytes([value])
            else:
                # 其他地址暂不支持
                logging.error("Unsupported address %s in write_byte", address)
                errored_ids.append(motor_id)
                continue
            packet = self.build_packet(motor_id, command, data)
            try:
                self.send_packet(packet)
            except Exception as e:
                logging.error("Error writing byte to motor id %s: %s", motor_id, e)
                errored_ids.append(motor_id)
        return errored_ids

    def set_torque_enabled(self,
                           motor_ids: Sequence[int],
                           enabled: bool,
                           retries: int = -1,
                           retry_interval: float = 0.25):
        """
        设置舵机的开/关力矩。
        """
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = self.write_byte(remaining_ids, int(enabled), ADDR_TORQUE_ENABLE)
            if remaining_ids:
                logging.error("Could not set torque %s for IDs: %s",
                              'enabled' if enabled else 'disabled',
                              str(remaining_ids))
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def sync_write(self, motor_ids: Sequence[int],
                   values: Sequence[Union[int, float]],
                   address: int,
                   size: int):
        """
        对一组舵机同时写入数据。
        由于 Feetech 驱动可能不支持组写，本实现遍历 motor_ids 逐个发送数据包。
        """
        self.check_connected()
        assert len(motor_ids) == len(values), "motor_ids and values length mismatch"
        for motor_id, value in zip(motor_ids, values):
            # 假设 value 已经是目标值，若需要转换请在此进行（如角度转换）
            # 将 value 转换到整数形式，并生成数据包
            if size == 2:
                # 2字节数据，按 little-endian 编码
                data = int(value).to_bytes(size, byteorder='little', signed=False)
            else:
                data = int(value).to_bytes(size, byteorder='little', signed=False)
            # 使用 CMD_SET_GOAL_POSITION 作为设置目标位置命令
            packet = self.build_packet(motor_id, CMD_SET_GOAL_POSITION, data)
            self.send_packet(packet)

    def write_desired_pos(self, motor_ids: Sequence[int],
                          positions: np.ndarray):
        """
        写入舵机目标位置。
        参数 positions 单位为度（你也可以改为其他单位），需要转换成舵机内部单位（此处假定不做转换）。
        """
        self.check_connected()
        assert len(motor_ids) == len(positions), "Mismatch in write_desired_pos"
        # 如果需要转换，例如将度转换为实际寄存器值，可在此处理
        self.sync_write(motor_ids, positions, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def check_connected_and_raise(self):
        self.check_connected()

    def __enter__(self):
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

def feetech_cleanup_handler():
    # 如果有多个 driver 实例，自动调用 disconnect() 以安全关闭串口
    logging.info("Cleanup Feetech driver resources.")

atexit.register(feetech_cleanup_handler)

# For command-line testing
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--motors', required=True,
        help='Comma-separated list of motor IDs, e.g., 1,2,3')
    parser.add_argument(
        '-p', '--port', default='/dev/ttyUSB0', help='Serial port device')
    parser.add_argument(
        '-b', '--baud', type=int, default=115200, help='Baudrate')
    args = parser.parse_args()

    motor_ids = [int(x) for x in args.motors.split(",")]
    driver = FeetechDriver(motor_ids, port=args.port, baudrate=args.baud)
    driver.connect()
    try:
        logging.info("Setting torque enabled for motors %s", motor_ids)
        driver.set_torque_enabled(motor_ids, True)
        # 示例：将所有舵机设置到 90 度
        target_positions = np.full((len(motor_ids),), 90.0)
        driver.write_desired_pos(motor_ids, target_positions)
        logging.info("Sent target positions: %s", target_positions.tolist())
    finally:
        driver.disconnect()