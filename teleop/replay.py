import os
import datetime
import time
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Dict
import numpy as np
import tyro
import zarr

from snake_agent import DynamixelRobotConfig
from snake_agent import SnakeAgent

@dataclass  
class Args:
    client_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EIOT-if00-port0"
    folder: str = "snake_data_box_0927/2024-09-29_22-29-14.zarr"
    num_joints: int = 9

def main(args):
    joint_ids = list(range(args.num_joints))
    config = DynamixelRobotConfig(joint_ids=joint_ids, joint_offsets=np.zeros(args.num_joints), joint_signs=np.ones(args.num_joints))
    client = SnakeAgent(port=args.client_port, dynamixel_config=config, enable_torque=True)

    # load all actions sorted by name (timestamp)
    actions = os.listdir(args.folder)
    actions.sort()

    # initialize: set client joint positions to initial joint positions
    reset_joints = pickle.load(open(os.path.join(args.folder, actions[0]), "rb"))
    print(reset_joints)
    curr_joints = client.get_act()
    assert reset_joints.shape == curr_joints.shape
    max_delta = (np.abs(reset_joints - curr_joints)).max()
    steps = min(int(max_delta / 0.01), 1000)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        client.set_act(jnt)
        time.sleep(0.001)

    for action_name in actions[1:]:
        action = pickle.load(open(os.path.join(args.folder, action_name), "rb"))
        print(action)
        client.set_act(action)

if __name__ == '__main__':
    main(tyro.cli(Args))