import os
import glob
import time
from pathlib import Path
from typing import Optional, Protocol, Tuple, List
import cv2
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager
from usb_util import reset_all_elgato_devices, get_sorted_v4l_paths

import numpy as np


class CameraDriver(Protocol):
    """Camera protocol.

    A protocol for a camera driver. This is used to abstract the camera from the rest of the code.
    """

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """


class DummyCamera(CameraDriver):
    """A dummy camera for testing."""

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """
        if img_size is None:
            return (
                np.random.randint(255, size=(480, 640, 3), dtype=np.uint8),
                np.random.randint(255, size=(480, 640, 1), dtype=np.uint16),
            )
        else:
            return (
                np.random.randint(
                    255, size=(img_size[0], img_size[1], 3), dtype=np.uint8
                ),
                np.random.randint(
                    255, size=(img_size[0], img_size[1], 1), dtype=np.uint16
                ),
            )


class SavedCamera(CameraDriver):
    def __init__(self, path: str = "example"):
        self.path = str(Path(__file__).parent / path)
        from PIL import Image

        self._color_img = Image.open(f"{self.path}/image.png")
        self._depth_img = Image.open(f"{self.path}/depth.png")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if img_size is not None:
            color_img = self._color_img.resize(img_size)
            depth_img = self._depth_img.resize(img_size)
        else:
            color_img = self._color_img
            depth_img = self._depth_img

        return np.array(color_img), np.array(depth_img)[:, :, 0:1]
    
def get_device_ids(usb = False) -> List[str]:
    device_ids = []
    if usb:
        video_ids = sorted(glob.glob('/dev/v4l/by-path/pci*'))
        device_ids = [vid for vid in video_ids if int(vid[-1]) == 0]
    else:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
            device_ids.append(dev.get_info(rs.camera_info.serial_number))
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(self, device_id: Optional[str] = None, flip: bool = False, enable_depth: bool = True):
        import pyrealsense2 as rs

        self._device_id = device_id
        self.enable_depth = enable_depth

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        if enable_depth:
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self._pipeline.start(config)
        self._flip = flip

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        if img_size is None:
            image = color_image[:, :, ::-1]
            if self.enable_depth:
                depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            if self.enable_depth:
                depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            if self.enable_depth:
                depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
                return image, depth
        else:
            if self.enable_depth:
                depth = depth[:, :, None]
                return image, depth
        return image, None
    
class USBCamera(CameraDriver):
    def __init__(self, device_id, flip: bool = False, fps: int = 30, resolution: Tuple[int, int] = (480, 640)):
        self._device_id = device_id
        self._cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._flip = flip
        self.fps = fps
        self.resolution = resolution

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        if img_size is not None:
            frame = cv2.resize(frame, img_size)
        if self._flip:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame, None
    
    def close(self):
        self._cap.release()

def _debug_read(camera, save_datastream=False, enable_depth=True, window_name="image"):
    cv2.namedWindow(window_name)
    if enable_depth:
        cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.03)
        image, depth = camera.read()
        if enable_depth:
            depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        cv2.imshow(window_name, image[:, :, ::-1])
        if enable_depth:
            cv2.imshow("depth", depth)
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            if enable_depth:
                cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            if enable_depth:
                cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break
    cv2.destroyWindow(window_name)


class camServer:
    def __init__(self, 
                 shm_manager: SharedMemoryManager, 
                 cameras: List[USBCamera],
                 top_camera: Optional[USBCamera] = None,
                 side_camera: Optional[USBCamera] = None,
                 random_disabling: bool = False,    # randomly disable cameras
                 random_disabling_prob: float = 0.2,    # probability of frame being disabled some cameras
                #  random_disabling_range: Tuple[float, float] = (0, 0.25),   # range of number of cameras to disable
                 random_latency: bool = False,      # randomly add latency to cameras
                 random_latency_prob: float = 0.2,  # probability of adding latency to some cameras
                 random_latency_range: Tuple[float, float] = (0, 1.0)   # range of latency to add to cameras
                 ):
        self.cameras = cameras
        self.top_camera = top_camera
        self.side_camera = side_camera
        self.shm = shm_manager.SharedMemory(size=640*480*3*len(cameras))
        # shm_size = sum([cam.resolution[0]*cam.resolution[1]*3 for cam in cameras])
        # self.shm = shm_manager.SharedMemory(size=shm_size)
        self.shm_array = np.ndarray((len(cameras), 480, 640, 3), dtype=np.uint8, buffer=self.shm.buf)
        self.shm_array[:] = 0
        self.shm_top_array = None
        self.shm_side_array = None
        if top_camera is not None:
            self.shm_top = shm_manager.SharedMemory(size=top_camera.resolution[0]*top_camera.resolution[1]*3)
            self.shm_top_array = np.ndarray((top_camera.resolution[0], top_camera.resolution[1], 3), dtype=np.uint8, buffer=self.shm_top.buf)
            self.shm_top_array[:] = 0

        if side_camera is not None:
            self.shm_side = shm_manager.SharedMemory(size=side_camera.resolution[0]*side_camera.resolution[1]*3)
            self.shm_side_array = np.ndarray((side_camera.resolution[0], side_camera.resolution[1], 3), dtype=np.uint8, buffer=self.shm_side.buf)
            self.shm_side_array[:] = 0

        self.random_disabling = random_disabling
        self.random_disabling_prob = random_disabling_prob
        # self.random_disabling_range = random_disabling_range
        self.random_latency = random_latency
        self.random_latency_prob = random_latency_prob
        self.random_latency_range = random_latency_range

        processes = [Process(target=self.loop, args=(i,)) for i in range(len(cameras))]
        if top_camera is not None:
            processes.append(Process(target=self.loop_top))
        if side_camera is not None:
            processes.append(Process(target=self.loop_side))
        for process in processes:
            process.start()
        self.processes = processes 

    def loop(self, camera_idx: int):
        while True:
            if self.random_disabling and np.random.rand() < self.random_disabling_prob:
                self.shm_array[camera_idx] = 0
                time.sleep(1/self.cameras[camera_idx].fps)
            else:
                img, _ = self.cameras[camera_idx].read()
                self.shm_array[camera_idx] = img
                time.sleep(1/self.cameras[camera_idx].fps)
            if self.random_latency and np.random.rand() < self.random_latency_prob:
                latency = np.random.uniform(*self.random_latency_range)
                time.sleep(latency)

    def loop_top(self): 
        while True:
            img, _ = self.top_camera.read()
            self.shm_top_array = img
            time.sleep(1/self.top_camera.fps)
    
    def loop_side(self):
        while True:
            img, _ = self.side_camera.read()
            self.shm_side_array = img
            time.sleep(1/self.side_camera.fps)

    def get_data(self):
        top = self.shm_top_array.copy() if self.shm_top_array is not None else None
        side = self.shm_side_array.copy() if self.shm_side_array is not None else None
        return self.shm_array.copy(), top, side
    
    def join(self):
        for process in self.processes:
            process.join()

    def end(self):
        for process in self.processes:
            process.terminate()
        for camera in self.cameras:
            camera.close()

if __name__ == "__main__":
    # reset_all_elgato_devices()
    # device_ids = get_device_ids(usb=True)
    # device_ids = get_sorted_v4l_paths(by_id=False)
    # print(f"Found {len(device_ids)} devices")
    # print(device_ids)
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

    usbcams = [USBCamera(device_id=idx) for idx in device_ids]
    # top_cam = USBCamera(device_id='/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2:1.0-video-index0', fps=30)
    # side_cam = USBCamera(device_id='/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1:1.0-video-index0', fps=30)
    top_cam = None  
    side_cam = None

    shm_manager = SharedMemoryManager()
    shm_manager.start()
    server = camServer(shm_manager, usbcams, top_cam, side_cam)
    time.sleep(2)
    print("starting loop")

    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(f"side_{save_dir}", exist_ok=True)
    # os.makedirs(f"top_{save_dir}", exist_ok=True)

    for i in range(1000):
        print("getting data")
        data, _, _ = server.get_data()
        np.save(f"{save_dir}/data_{i}.npy", data)
        # np.save(f"top_{save_dir}/data_{i}.npy", top_data)
        # np.save(f"side_{save_dir}/data_{i}.npy", side_data)
        time.sleep(0.1)
    server.join()