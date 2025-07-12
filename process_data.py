import os
from tqdm import tqdm
import cv2
import zarr
import numpy as np
import mujoco
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def flip(imgs):
    return np.flip(np.flip(imgs, axis=1), axis=2)

def get_image_transform(in_res, out_res, crop_ratio_h: float=1.0, crop_ratio_w: float=1.0, bgr_to_rgb: bool=False, w_slice_start=None, h_slice_start=None):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio_h)
    cw = round(ih * crop_ratio_w / oh * ow)
    interp_method = cv2.INTER_AREA

    if w_slice_start is None:
        w_slice_start = (iw - cw) // 2
    else:
        w_slice_start = round(iw * w_slice_start)
    w_slice = slice(w_slice_start, w_slice_start + cw)
    if h_slice_start is None:
        h_slice_start = (ih - ch) // 2
    else:
        h_slice_start = round(ih * h_slice_start)
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform

def generate_replay_buffer_from_zarr(zarr_dir, output_path, compression_level=99, in_res=(640, 480), out_res=(224, 224), render=False):
    '''
    file structure:
    snake_data/
    ├── 2025-07-10_13-33-48.zarr
    │   ├── control
    │   ├── joint_pos
    │   ├── joint_vel
    │   ├── rgb
    │   ├── timestamp
    ├── ...
    '''
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore()
    )
    resize_tf = get_image_transform(in_res, out_res, crop_ratio_h=1.0, crop_ratio_w=1.0)
    
    episode_files = os.listdir(zarr_dir)
    episode_files = [file for file in episode_files if file.endswith(".zarr")]
    episode_files.sort()
    num_frames = sum(len(zarr.open_group(os.path.join(zarr_dir, episode_file))['rgb']) for episode_file in episode_files)

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    num_cam = zarr.open_group(os.path.join(zarr_dir, episode_files[0]))['rgb'].shape[1]

    num_cam_to_use = num_cam - 2    # excluding the top-down and side-view cameras which are only used for comparisons

    for i in range(num_cam_to_use):
        out_replay_buffer.data.require_dataset(
            name=f'camera{i}_rgb',
            shape=(num_frames,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    mjmodel = mujoco.MjModel.from_xml_path('./robot/scene_up.xml')
    mjdata = mujoco.MjData(mjmodel)
    if render:
        renderer = mujoco.Renderer(mjmodel, 256, 256)
        camera = mujoco.MjvCamera()

    def process_episode(replay_buffer, episode_path):
        data = zarr.open_group(episode_path)
        if data['rgb'].shape[1] != num_cam:
            print(f"Skipping episode {episode_path} due to mismatch in number of cameras")
            return
        episode_data = {
            'joint_pos': data['joint_pos'],
            'joint_vel': data['joint_vel'],
            'action': data['control']
        }
        cam_pos = []
        cam_ori = []
        if render:
            imgs = []
        for i in range(data['joint_pos'].shape[0]):
            mjdata.qpos[:] = data['joint_pos'][i] - np.pi
            mjdata.qvel[:] = data['joint_vel'][i]
            mujoco.mj_step(mjmodel, mjdata)
            orientations = mjdata.site_xmat.copy().reshape((-1, 3, 3))[..., :2] # only keep the first two columns of the rotation matrix
            orientations = orientations.reshape((-1, 6))
            cam_pos.append(mjdata.site_xpos.copy().reshape((-1, 3)))
            cam_ori.append(orientations)
            # cam_pos.append(np.concatenate([positions, orientations], axis=1))
            if render:
                renderer.update_scene(mjdata, camera)
                img = renderer.render()
                imgs.append(np.array(img).astype(np.uint8))

        episode_data['camera_pos'] = np.stack(cam_pos, axis=0)[:, [16, 17, 19, 18, 20, 15, 11, 7, 3, 14, 10, 6, 2, 12, 8, 4, 0, 13, 9, 5, 1], :]
        episode_data['camera_ori'] = np.stack(cam_ori, axis=0)[:, [16, 17, 19, 18, 20, 15, 11, 7, 3, 14, 10, 6, 2, 12, 8, 4, 0, 13, 9, 5, 1], :]

        if render:
            out = cv2.VideoWriter('camera.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 256))
            for img in imgs:
                out.write(img)
            out.release()

        for i in [7, 9, 12, 14, 16, 18, 19, 21]:
            data['rgb'][:, i, ...] = np.flip(np.flip(data['rgb'][:, i, ...], axis=1), axis=2)
        for i in range(num_cam - 2):
            episode_data[f'camera{i}_rgb'] = np.stack([resize_tf(frame) for frame in data['rgb'][:,i+2]], axis=0)

        replay_buffer.add_episode(episode_data)

    with tqdm(total=len(episode_files)) as pbar:
        for episode_file in episode_files:
            process_episode(out_replay_buffer, os.path.join(zarr_dir, episode_file))
            pbar.update()

    # Save to disk
    print(f"Saving ReplayBuffer to {output_path}")
    out_replay_buffer.save_to_path(output_path)
    print("Done!")
    
if __name__ == "__main__":
    generate_replay_buffer_from_zarr('saved_data_dir', 'dataset.zarr.zip', in_res=(640, 480), out_res=(224, 224))

