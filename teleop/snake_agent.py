import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from dynamixel_driver import DynamixelDriver

class DynamixelRobot():

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4000000,
        start_joints: Optional[np.ndarray] = None,
        enable_torque: bool = False,
    ):
        self.kP = 600   #600
        self.kI = 0   #0
        self.kD = 200   #200
        self.curr_lim = 550
        self._joint_ids = joint_ids

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

        self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
        self._driver.connect()
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * 5, 11, 1)
        self._driver.set_torque_enabled(joint_ids, enable_torque)
        # self._driver.set_torque_enabled(joint_ids, False)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kP, 84, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kI, 82, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kD, 80, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.curr_lim, 102, 2)
        self._torque_on = True
        self._last_pos = None
        self._alpha = 0.99

        if start_joints is not None:
            # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
            new_joint_offsets = []
            current_joints = self.get_joint_pos()
            assert current_joints.shape == start_joints.shape
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints, start_joints, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    np.pi
                    * 2
                    * np.round((-s_joint + c_joint) / (2 * np.pi))
                    * self._joint_signs[idx]
                    + joint_offset
                )
            self._joint_offsets = np.array(new_joint_offsets)

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_pos(self) -> np.ndarray:
        pos = (self._driver.read_pos() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos
    
    def get_joint_vel(self) -> np.ndarray:
        return self._driver.read_vel() * self._joint_signs

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.write_desired_pos(self._joint_ids, joint_state + self._joint_offsets)

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_enabled(self._joint_ids, mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_pos": self.get_joint_pos(), "joint_vel": self.get_joint_vel()}
    

@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None, enable_torque: bool = False
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            joint_signs=list(self.joint_signs),
            port=port,
            start_joints=start_joints,
            enable_torque=enable_torque
        )
    
class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()

class SnakeAgent():
    def __init__(
            self,
            port: str,
            dynamixel_config: Optional[DynamixelRobotConfig] = None,
            enable_torque: bool = False,
            start_joints: Optional[np.ndarray] = None,
            control_rate_hz: float = 100.0,
            camera_dict: Optional[Dict[str, Any]] = None,
            camera_server: Optional[Any] = None,
    ) -> None:
        self._robot = dynamixel_config.make_robot(
            port=port, start_joints=start_joints, enable_torque=enable_torque
        )
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._camera_server = camera_server

    def get_obs(self) -> Dict[str, Any]:
        observations = {}
        for name, camera in self._camera_dict.items(): 
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            if camera.enable_depth:
                observations[f"{name}_depth"] = depth
        if self._camera_server is not None:
            imgs, top_img, side_img = self._camera_server.get_data()
            observations["usb_cam"] = imgs
            if top_img is not None:
                observations["top_cam"] = top_img
            if side_img is not None:
                observations["side_cam"] = side_img

        robot_obs = self._robot.get_observations()
        observations["joint_pos"] = robot_obs["joint_pos"]
        observations["joint_vel"] = robot_obs["joint_vel"]
        return observations

    def get_act(self) -> np.ndarray:
        return self._robot.get_joint_pos()
    
    def set_act(self, joints: np.ndarray) -> Dict[str, Any]:
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

