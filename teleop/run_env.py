import os
import datetime
import time
from dataclasses import dataclass
import numpy as np
import tyro
from multiprocessing.managers import SharedMemoryManager
from snake_agent import DynamixelRobotConfig, SnakeAgent
from camera import USBCamera, camServer
import zarr

@dataclass  
class Args:
    client_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HD8F2-if00-port0"
    leader_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EJ6N-if00-port0"
    folder: str = "data"
    num_joints: int = 9
    control_rate: int = 30
    camera_fps: int = 30

def main(args):
    joint_ids = list(range(args.num_joints))
    leader_config = DynamixelRobotConfig(joint_ids=joint_ids, joint_offsets=np.zeros(args.num_joints), joint_signs=np.ones(args.num_joints))
    client_config = DynamixelRobotConfig(joint_ids=joint_ids, joint_offsets=np.zeros(args.num_joints), joint_signs=np.ones(args.num_joints))
    device_ids = [
    # top-down, side (external cameras)
    '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1:1.0-video-index0',
    # top, front, back, left, right
    '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:10:1.0-video-index0', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:11:1.0-video-index0', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0', '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0', 
    # front
    '/dev/v4l/by-path/pci-0000:2b:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:2e:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:2f:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:30:00.0-usb-0:2:1.0-video-index0', 
    # back
    '/dev/v4l/by-path/pci-0000:33:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:36:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:37:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:38:00.0-usb-0:2:1.0-video-index0',
    # left
    '/dev/v4l/by-path/pci-0000:1b:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:1e:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:1f:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:20:00.0-usb-0:2:1.0-video-index0', 
    # right
    '/dev/v4l/by-path/pci-0000:23:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:26:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:27:00.0-usb-0:2:1.0-video-index0', '/dev/v4l/by-path/pci-0000:28:00.0-usb-0:2:1.0-video-index0', 
    ]
    usbcams = [USBCamera(device_id=idx, fps=args.camera_fps) for idx in device_ids]
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    camera_server = camServer(shm_manager, usbcams, None, None)
    time.sleep(2)
    camera_clients = {}
    leader = SnakeAgent(port=args.leader_port, dynamixel_config=leader_config, enable_torque=False, control_rate_hz=args.control_rate)
    client = SnakeAgent(port=args.client_port, dynamixel_config=client_config, enable_torque=True, control_rate_hz=args.control_rate, camera_dict=camera_clients, camera_server=camera_server)

    # initialize: set client joint positions to leader joint positions
    curr_joints = client.get_act()
    reset_joints = leader.get_act()
    assert reset_joints.shape == curr_joints.shape
    max_delta = (np.abs(reset_joints - curr_joints)).max()
    steps = min(int(max_delta / 0.01), 1000)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        client.set_act(jnt)
        time.sleep(0.001)
    print("Initialized joint positions", client.get_act(), leader.get_act())

    obs = client.get_obs()
    save_data = {
        "rgb": [],
        "timestamp": [],
        "joint_pos": [],
        "joint_vel": [],
        "control": [],
    }
    try:
        while len(save_data["rgb"]) < 720:
            action = leader.get_act() 
            now = time.time()
            save_data["rgb"].append(obs["usb_cam"])
            save_data["timestamp"].append(now)
            save_data["joint_pos"].append(obs["joint_pos"])
            save_data["joint_vel"].append(obs["joint_vel"])
            save_data["control"].append(action)
            obs = client.set_act(action)
        print('timeout!')
        pass
    except KeyboardInterrupt:
        pass
    if camera_server is not None:
        camera_server.end()
    del camera_server
    del leader
    del client
    zarr_output_path = os.path.join(args.folder, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zarr")
    root = zarr.group(store=zarr.DirectoryStore(zarr_output_path), overwrite=True)
    root.create_dataset(
        "rgb", 
        data=np.array(save_data["rgb"]),
        shape=(len(save_data["rgb"]), len(device_ids), 480, 640, 3),
        compressor=zarr.Blosc(cname="zstd", clevel=5),
    )
    root.create_dataset(
        "timestamp", 
        data=np.array(save_data["timestamp"]),
        shape=(len(save_data["timestamp"]),),
        compressor=zarr.Blosc(cname="zstd", clevel=5),
    )
    root.create_dataset(
        "joint_pos", 
        data=np.array(save_data["joint_pos"]),
        shape=(len(save_data["joint_pos"]), args.num_joints),
        compressor=zarr.Blosc(cname="zstd", clevel=5),
    )
    root.create_dataset(
        "joint_vel", 
        data=np.array(save_data["joint_vel"]),
        shape=(len(save_data["joint_vel"]), args.num_joints),
        compressor=zarr.Blosc(cname="zstd", clevel=5),
    )   
    root.create_dataset(
        "control", 
        data=np.array(save_data["control"]),
        shape=(len(save_data["control"]), args.num_joints),
        compressor=zarr.Blosc(cname="zstd", clevel=5),
    )

if __name__ == "__main__":
    main(tyro.cli(Args))