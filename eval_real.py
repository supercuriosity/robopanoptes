import os
import datetime
import time
import hydra
import dill
import numpy as np
from typing import Optional, Protocol, Tuple, List
import zarr
import torch
from multiprocessing.managers import SharedMemoryManager
import mujoco
from diffusion_policy.common.pytorch_util import dict_apply

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from teleop.snake_agent import DynamixelRobotConfig
from teleop.snake_agent import SnakeAgent
from teleop.camera import USBCamera, camServer
from process_data import get_image_transform

def flip_image(image):
    return np.flip(np.flip(image, axis=0), axis=1)

def get_policy_obs(obs, resize_tf=None):
    policy_obs = {
        'joint_pos': obs['joint_pos'].astype(np.float32),
    }      
    for i in [7, 9, 12, 14, 16, 18, 19, 21]:
        obs['usb_cam'][i] = np.flip(np.flip(obs['usb_cam'][i], axis=0), axis=1)
    for i, frame in enumerate(obs['usb_cam'][2:]):
        rgb = resize_tf(frame)
        rgb = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
        policy_obs[f'camera{i}_rgb'] = rgb
    return policy_obs

def main(
        input, 
        port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HD8F2-if00-port0",
        num_joints: int = 9,
        control_rate: int = 30,
        camera_fps: int = 30,
        initialize: bool = False,
        record: bool = False,
        save_dir: str = "snake_videos",
        random_disabling: bool = False,    # randomly disable cameras
        random_disabling_prob: float = 0.1,    # probability of frame being disabled some cameras
        random_latency: bool = False,      # randomly add latency to cameras
        random_latency_prob: float = 0.2,  # probability of adding latency to some cameras
        random_latency_range: Tuple[float, float] = (0, 1.0),   # range of latency to add to cameras
        store_attention: bool = False,  # store attention maps
         ):
    mjmodel = mujoco.MjModel.from_xml_path('./robot/scene_up.xml')
    mjdata = mujoco.MjData(mjmodel)
    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name: ", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # creating model
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16

    img_obs_horizon = cfg.task.img_obs_horizon
    low_dim_obs_horizon = cfg.task.low_dim_obs_horizon  # same as img_obs_horizon
    
    device = torch.device('cuda')
    policy.eval().to(device)

    print("policy inference")
    policy.reset()
    if store_attention:
        dec_attn_weights = []
        hooks = [
            policy.model.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            )
        ]
    in_res = (640, 480)
    crop_ratio_w = 1.0
    out_res = (224, 224)

    resize_tf = get_image_transform(in_res, out_res, crop_ratio_h=1.0, crop_ratio_w=crop_ratio_w)

    # setup robot
    joint_ids = list(range(num_joints))
    robot_config = DynamixelRobotConfig(joint_ids=joint_ids, joint_offsets=np.zeros(num_joints), joint_signs=np.ones(num_joints))

    device_ids = [
    # top-down, side
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
    usbcams = [USBCamera(device_id=idx, fps=camera_fps) for idx in device_ids]
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    camera_server = camServer(shm_manager, usbcams, random_disabling=random_disabling, random_disabling_prob=random_disabling_prob, random_latency=random_latency, random_latency_prob=random_latency_prob, random_latency_range=random_latency_range)
    time.sleep(2)
    camera_clients = {}

    robot = SnakeAgent(port=port, dynamixel_config=robot_config, enable_torque=True, control_rate_hz=control_rate, camera_dict=camera_clients, camera_server=camera_server)

    if initialize:
        # initialize: set client joint positions to leader joint positions
        curr_joints = robot.get_act()
        reset_joints = np.ones_like(curr_joints) * np.pi
        assert reset_joints.shape == curr_joints.shape
        max_delta = (np.abs(reset_joints - curr_joints)).max()
        steps = min(int(max_delta / 0.01), 1000)
        for jnt in np.linspace(curr_joints, reset_joints, steps):
            robot.set_act(jnt)
            time.sleep(0.001)
        print("Initialized joint positions", robot.get_act())

    obs = robot.get_obs()
    os.makedirs(save_dir, exist_ok=True)
    save_data = {
        "rgb": [],
        "joint_pos": [],
        "joint_vel": [],
        "control": [],
    }
    with torch.no_grad():
        policy.reset()
        obs_dict_np = get_policy_obs(obs, resize_tf=resize_tf)
        mjdata.qpos[:] = obs['joint_pos'] - np.pi
        mjdata.qvel[:] = obs['joint_vel']
        mujoco.mj_step(mjmodel, mjdata)
        obs_dict_np['camera_ori'] = mjdata.site_xmat.copy().reshape((-1, 3, 3))[..., :2].reshape((-1, 6))[[16, 17, 19, 18, 20, 15, 11, 7, 3, 14, 10, 6, 2, 12, 8, 4, 0, 13, 9, 5, 1], :]
        obs_dict_np['camera_pos'] = mjdata.site_xpos.copy().reshape((-1, 3))[[16, 17, 19, 18, 20, 15, 11, 7, 3, 14, 10, 6, 2, 12, 8, 4, 0, 13, 9, 5, 1], :]
        last_obs_dict_np = {k: np.repeat(v[None], img_obs_horizon, axis=0) for k, v in obs_dict_np.items()}
        obs_dict = dict_apply(last_obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        # print("obs_dict shape: ", {k: v.shape for k, v in obs_dict.items()})
        result = policy.predict_action(obs_dict)
        action = result['action_pred'][0].detach().to('cpu').numpy()
        assert action.shape[-1] == num_joints
        del result
        if store_attention:
            dec_attn_weights.pop()

    print("ready")
    try:
        while True:
            obs = robot.get_obs()  
            if record:
                save_data["rgb"].append(obs["usb_cam"])
                save_data["joint_pos"].append(obs["joint_pos"])
                save_data["joint_vel"].append(obs["joint_vel"])

            with torch.no_grad():
                obs_dict_np = get_policy_obs(obs, resize_tf=resize_tf)
                mjdata.qpos[:] = obs['joint_pos'] - np.pi
                mjdata.qvel[:] = obs['joint_vel']
                mujoco.mj_step(mjmodel, mjdata)
                obs_dict_np['camera_ori'] = mjdata.site_xmat.copy().reshape((-1, 3, 3))[..., :2].reshape((-1, 6))[[16, 17, 19, 18, 20, 15, 11, 7, 3, 14, 10, 6, 2, 12, 8, 4, 0, 13, 9, 5, 1], :]
                obs_dict_np['camera_pos'] = mjdata.site_xpos.copy().reshape((-1, 3))[[16, 17, 19, 18, 20, 15, 11, 7, 3, 14, 10, 6, 2, 12, 8, 4, 0, 13, 9, 5, 1], :]
                last_obs_dict_np = {k: np.concatenate([last_obs_dict_np[k][1:], obs_dict_np[k][None]], axis=0) for k in obs_dict_np.keys()}
                obs_dict = dict_apply(last_obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                actions = policy.predict_action(obs_dict)['action_pred'][0].detach().cpu().numpy()

                for action in actions:
                    robot.set_act(action)
                if record:
                    save_data["control"].append(actions)
    except KeyboardInterrupt:
        print("Interrupted")
        pass

    if record:
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        zarr_output_path = os.path.join(save_dir, f"{now}.zarr")
        root = zarr.group(store=zarr.DirectoryStore(zarr_output_path), overwrite=True)
        root.create_dataset(
            "rgb", 
            data=np.array(save_data["rgb"]),
            shape=(len(save_data["rgb"]), len(device_ids), 480, 640, 3),
            compressor=zarr.Blosc(cname="zstd", clevel=5),
        )
        root.create_dataset(
            "joint_pos", 
            data=np.array(save_data["joint_pos"]),
            shape=(len(save_data["joint_pos"]), num_joints),
            compressor=zarr.Blosc(cname="zstd", clevel=5),
        )
        root.create_dataset(
            "joint_vel", 
            data=np.array(save_data["joint_vel"]),
            shape=(len(save_data["joint_vel"]), num_joints),
            compressor=zarr.Blosc(cname="zstd", clevel=5),
        )   
        save_data["control"] = np.stack(save_data["control"])
        root.create_dataset(
            "control", 
            data=save_data["control"],
            shape=save_data["control"].shape,
            compressor=zarr.Blosc(cname="zstd", clevel=5),
        )
        if store_attention:
            for hook in hooks:
                hook.remove()
            dec_attn_weights = torch.concat(dec_attn_weights, 0)
            print("dec_attn_weights shape: ", dec_attn_weights.shape)
            root.create_dataset(
                "attn_weights",
                data=dec_attn_weights.cpu().numpy(),
                shape=dec_attn_weights.shape,
                compressor=zarr.Blosc(cname="zstd", clevel=5),
            )
        
if __name__ == '__main__':
    main('sweep.ckpt', initialize=False, record=True, save_dir="snake_videos")