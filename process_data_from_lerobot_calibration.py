import os, glob
from tqdm import tqdm
import cv2
import torch
import zarr
import numpy as np
import pandas as pd
import mujoco
import importlib
from pathlib import Path
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

QR_DICT = {
    "oversensor calibration qrcode 1": {"pos": [0.0, 0.0, 0.0], "orient": [0.0, 0.0, 0.0]},
    "oversensor calibration qrcode 2": {"pos": [1.0, 0.0, 0.0], "orient": [0.0, 0.0, np.pi/2]},
    "oversensor calibration qrcode 3": {"pos": [1.0, 0.0, 0.0], "orient": [0.0, 0.0, np.pi/2]},
    "oversensor calibration qrcode 4": {"pos": [1.0, 0.0, 0.0], "orient": [0.0, 0.0, np.pi/2]},
    "oversensor calibration qrcode 5": {"pos": [1.0, 0.0, 0.0], "orient": [0.0, 0.0, np.pi/2]},
}

def calculate_camera_position(qr_data, qr_pos, qr_rot, QR_DICT):
    camera_positions = []
    camera_orientations = []
    
    for data, tvec, rvec in zip(qr_data, qr_pos, qr_rot):
        if data is None or tvec is None or rvec is None:
            camera_positions.append(None)
            camera_orientations.append(None)
            continue
        
        # 获取二维码实际位置和方向
        qr_info = QR_DICT.get(data)
        if qr_info is None:
            camera_positions.append(None)
            camera_orientations.append(None)
            continue
        
        qr_pos_actual = np.array(qr_info["pos"])
        qr_orient_actual = np.array(qr_info["orient"])
        
        # 将二维码和摄像头的旋转向量转换为旋转矩阵
        rotation_matrix_qr, _ = cv2.Rodrigues(qr_orient_actual)
        rotation_matrix_cam, _ = cv2.Rodrigues(np.array(rvec))
        
        # 组合旋转矩阵
        combined_rotation_matrix = np.dot(rotation_matrix_qr, rotation_matrix_cam)
        
        # 转换回旋转向量
        combined_rvec, _ = cv2.Rodrigues(combined_rotation_matrix)
        
        # 计算摄像头位置
        camera_pos = qr_pos_actual + np.array(tvec)
        
        camera_positions.append(camera_pos)
        camera_orientations.append(combined_rvec.flatten())
    
    return camera_positions, camera_orientations

def calculate_base_position(camera_positions, camera_orientations, episode_camera_pos, episode_camera_ori):
    base_positions = []
    
    for cam_pos, cam_ori, ep_cam_pos, ep_cam_ori in zip(camera_positions, camera_orientations, episode_camera_pos, episode_camera_ori):
        if cam_pos is None or cam_ori is None:
            base_positions.append(None)
            continue
        
        # 将 rvec 转换为旋转矩阵
        print("cam_ori:",cam_ori)
        print("cam_ori.type:",type(cam_ori))
        rotation_matrix, _ = cv2.Rodrigues(cam_ori)
        
        # 组合旋转矩阵
        combined_rotation_matrix = np.dot(rotation_matrix, ep_cam_ori)
        
        # 提取前两列
        combined_orientation = combined_rotation_matrix[:, :2]
        
        # 计算 base 的实际位置
        base_pos = ep_cam_pos + cam_pos  # 示例计算方式
        
        base_positions.append({"pos": base_pos, "orient": combined_orientation})
    
    return base_positions

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

def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    # initialize video decoder
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]

    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    # convert to float32 in [0,1] range (channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames

def generate_replay_buffer_from_zarr(lerobot_dir, output_path, compression_level=99, in_res=(640, 480), out_res=(224, 224), render=False):
    '''
    file structure:
    project_name/
    ├── data/chunk-000
    │   ├── episode_000000.parquet
    │   ├── episode_000001.parquet
    │   ├── episode_000002.parquet
    │   ├── ...
    ├── video/chunk-000
    │   ├── camera1
    │   │   ├── episode_000000.mp4
    │   │   ├── episode_000001.mp4
    │   │   ├── episode_000002.mp4
    │   ├── camera2
    │   │   ├── episode_000000.mp4
    │   │   ├── episode_000001.mp4
    │   │   ├── episode_000002.mp4
    │   ├── ...
    ├── ...
    '''
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore()
    )
    resize_tf = get_image_transform(in_res, out_res, crop_ratio_h=1.0, crop_ratio_w=1.0)
    
    data_dir=os.path.join(lerobot_dir, "data")
    video_dir=os.path.join(lerobot_dir, "videos")
    # parquet_files = [os.path.basename(path) for path in glob.glob(data_dir+'/chunk-*/*.parquet')]
    parquet_files = [path for path in glob.glob(data_dir+'/chunk-*/*.parquet')]
    parquet_files.sort()

    tolerance_s=1e-4
    camera_name_to_id={  # 使用相机名称到ID的映射
        "module1_f":0,
        "module1_b":1,
        "module8_f":2,
        "module8_b":3,
        "head":4,
        "bottom":5,
    }
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    num_frames=0
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        num_frames+=len(df)
    # Initialize datasets
    for i in camera_name_to_id.values():
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
    # 二维码检测相关代码开始
    def read_camera_parameters(filepath = 'camera_parameters/intrinsic.dat'):
        try:
            inf = open(filepath, 'r')
        except FileNotFoundError:
            cmtx = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)  # 默认内部参数
            dist = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return cmtx, dist
        
        cmtx = []
        dist = []
        # ignore first line
        line = inf.readline()
        for _ in range(3):
            line = inf.readline().split()
            line = [float(en) for en in line]
            cmtx.append(line)
        # ignore line that says "distortion"
        line = inf.readline()
        line = inf.readline().split()
        line = [float(en) for en in line]
        dist.append(line)
        # cmtx = camera matrix, dist = distortion parameters
        return np.array(cmtx), np.array(dist)

    def get_qr_coords(cmtx, dist, points):
        # Selected coordinate points for each corner of QR code.
        qr_edges = np.array([[0,0,0],
                             [0,1,0],
                             [1,1,0],
                             [1,0,0]], dtype = 'float32').reshape((4,1,3))
        # determine the orientation of QR code coordinate system with respect to camera coorindate system.
        ret, rvec, tvec = cv2.solvePnP(qr_edges, points, cmtx, dist)
        # Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
        unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
        if ret:
            points, jac = cv2.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
            return points, rvec.flatten().tolist(), tvec.flatten().tolist()
        else: 
            return [], [], []
    cmtx, dist = read_camera_parameters()
    # 二维码检测相关代码结束

    for parquet_path in tqdm(parquet_files, desc="Processing episodes"):
        # Load parquet data
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        episode_length = len(df)
        timestamps = df['timestamp'].to_numpy()
        
        # Get video paths
        chunk = os.path.basename(os.path.dirname(parquet_path))
        episode_id = os.path.basename(parquet_path).split('.')[0]
        video_chunk_dir = os.path.join(video_dir, chunk)
        
        # Process each camera
        camera_frames = dict()
        camera_pos_list = dict()
        camera_rot_list = dict()
        for cam_dir in sorted(glob.glob(os.path.join(video_chunk_dir, 'observation.images.*'))):
            cam_name = os.path.basename(cam_dir).replace('observation.images.', '')
            if cam_name not in camera_name_to_id:
                continue
            cam_id = camera_name_to_id[cam_name]
            video_path = os.path.join(cam_dir, f'{episode_id}.mp4')
            
            # Decode video frames with torchcodec
            try:
                frames_tensor = decode_video_frames_torchcodec(
                    video_path=Path(video_path),
                    timestamps=timestamps.tolist(),
                    tolerance_s=tolerance_s,
                    device="cpu"
                )
            except Exception as e:
                print(f"Failed to decode {video_path}: {str(e)}")
                continue
            # Process frames and collect QR code data
            frames = []
            qr_data = []  # 新增二维码数据列表
            qr_pos = []  # 新增二维码位置列表
            qr_rot = []  # 新增二维码旋转列表
            for frame in frames_tensor:
                # Convert tensor to numpy array (HWC format)
                frame_np = frame.permute(1, 2, 0).numpy() * 255
                frame_np = frame_np.astype(np.uint8)
                # Apply transformations
                processed_frame = resize_tf(frame_np)
                frames.append(processed_frame)

                # QR code detection added here
                try:
                    # QR code detection added here
                    qr = cv2.QRCodeDetector()
                    data, points, _ = qr.detectAndDecode(frame_np)
                except cv2.error as e:
                    print(f"QR code detection failed: {str(e)}")
                    data, points = None, None
                data, points, _ = qr.detectAndDecode(frame_np)
                if points is not None and data:
                    _, rvec, tvec = get_qr_coords(cmtx, dist, points.reshape(4,2))
                    # 将数据添加到列表中
                    qr_data.append(data)
                    qr_pos.append(tvec)
                    qr_rot.append(rvec)
                    print("!INFO:",cam_id,data,tvec,rvec)
                else:
                    # 如果未检测到二维码，则添加None或空值
                    qr_data.append(None)
                    qr_pos.append(None)
                    qr_rot.append(None)
                    # print("!INFO:",cam_id,"NO QRcode!")
            assert len(frames) == episode_length, f"Camera {cam_id} frames {len(frames)} != {episode_length}"
            camera_frames[cam_id] = np.array(frames)
            camera_positions, camera_orientations = calculate_camera_position(qr_data, qr_pos, qr_rot, QR_DICT)
            camera_pos_list[cam_id] = camera_positions
            camera_rot_list[cam_id] = camera_orientations

        # Build episode data
        episode_data = {
            'action': np.stack(df['action'].values).astype(np.float32),
            'joint_pos': np.stack(df['observation.state'].values).astype(np.float32)/180*np.pi,
            'joint_vel': np.ones_like(np.stack(df['observation.state'].values), dtype=np.float32),
            'timestamp': timestamps,
        }
        if render:
            cam_pos = []
            cam_ori = []
            if render:
                imgs = []
            for i in range(episode_data['joint_pos'].shape[0]):
                # joint_pos has converted from -90~90 to -pi/2~pi/2
                mjdata.qpos[:] = episode_data['joint_pos'][i]   # no bias - np.pi/180*100
                # mjdata.qvel[:] = episode_data['joint_vel'][i]
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

            episode_data['camera_pos'] = np.stack(cam_pos, axis=0)[:, [1,2,14,15], :]
            episode_data['camera_ori'] = np.stack(cam_ori, axis=0)[:, [1,2,14,15], :]
            # 计算 base 的实际位置
            base_positions = calculate_base_position(camera_pos_list, camera_rot_list, episode_data['camera_pos'], episode_data['camera_ori'])

            if render:
                out = cv2.VideoWriter('camera.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 256))
                for img in imgs:
                    out.write(img)
                out.release()

        for cam_id in camera_frames:
            episode_data[f'camera{cam_id}_rgb'] = camera_frames[cam_id].astype(np.uint8)
        
        # for key in episode_data:
        #     print(key)
        #     print(episode_data[key].dtype)
        
        out_replay_buffer.add_episode(episode_data)
    
    # Final save
    out_replay_buffer.save_to_path(output_path)
    print("Done!")

if __name__ == "__main__":
    generate_replay_buffer_from_zarr('/home/yuchenyang/.cache/huggingface/lerobot/yucy207/robopanoptes_test12', '/home/yuchenyang/workspace/RoboPanoptes/dataset.zarr.zip', in_res=(640, 480), out_res=(224, 224), render=False)