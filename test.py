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
from process_data_from_lerobot import decode_video_frames_torchcodec
vp="/home/yuchenyang/.cache/huggingface/lerobot/yucy207/robopanoptes_test1/videos/chunk-000/observation.images.head/episode_000000.mp4"
pp="/home/yuchenyang/.cache/huggingface/lerobot/yucy207/robopanoptes_test1/data/chunk-000/episode_000000.parquet"
df = pd.read_parquet(pp, engine='pyarrow')
frames_tensor = decode_video_frames_torchcodec(video_path=Path(vp),timestamps=df["timestamp"].tolist(),tolerance_s=1e-4,device='cpu')
