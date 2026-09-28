import cv2
import numpy as np
import torch
import random
from typing import List, Tuple

# Constants
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

class VideoTransform:
    """
    Video transformations for training, validation, testing, and inference modes.
    Applies resizing, cropping, flipping, color jitter, and normalization.
    """
    def __init__(self, mode: str = 'train', crop_size: int = 112, 
                 resize_h: int = 128, resize_w: int = 171, 
                 inference_h: int = 112, inference_w: int = 144):
        assert mode in ['train', 'val', 'test', 'inference'], f"Unknown mode: {mode}"
        self.mode = mode
        self.crop_size = crop_size
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.inference_h = inference_h
        self.inference_w = inference_w

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        """
        Apply transforms to a video clip.
        Args:
            clip (np.ndarray): Video clip of shape (T, H, W, C) with uint8 values [0, 255], RGB.
        Returns:
            torch.Tensor: Normalized video clip of shape (C, T, H, W), float32.
        """
        T, H, W, C = clip.shape
        
        # Decide resize dimensions based on mode
        if self.mode == 'inference':
            target_h, target_w = self.inference_h, self.inference_w
        else:
            target_h, target_w = self.resize_h, self.resize_w

        # Resize all frames (same size for all frames)
        resized_clip = np.empty((T, target_h, target_w, C), dtype=np.uint8)
        for i in range(T):
            resized_clip[i] = cv2.resize(clip[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
        clip = resized_clip

        # Spatial augmentations/cropping
        if self.mode == 'train':
            # Random crop
            if target_h >= self.crop_size and target_w >= self.crop_size:
                y = random.randint(0, target_h - self.crop_size)
                x = random.randint(0, target_w - self.crop_size)
                clip = clip[:, y:y+self.crop_size, x:x+self.crop_size, :]
            
            # Random horizontal flip
            if random.random() > 0.5:
                clip = np.flip(clip, axis=2)
                
            # Random color jitter
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            
            # Apply color jitter in float space to prevent uint8 overflow issues
            clip = clip.astype(np.float32)
            clip = clip * contrast + (brightness - 1.0) * 255.0
            clip = np.clip(clip, 0, 255).astype(np.uint8)
            
        elif self.mode in ['val', 'test']:
            # Center crop
            if target_h >= self.crop_size and target_w >= self.crop_size:
                y = (target_h - self.crop_size) // 2
                x = (target_w - self.crop_size) // 2
                clip = clip[:, y:y+self.crop_size, x:x+self.crop_size, :]
                
        # For 'inference', no crop is applied.

        # Convert to float32 [0, 1]
        clip = clip.astype(np.float32) / 255.0

        # Normalize with Kinetics mean and std
        mean = np.array(KINETICS_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array(KINETICS_STD, dtype=np.float32).reshape(1, 1, 1, 3)
        clip = (clip - mean) / std

        # Transpose to (C, T, H, W)
        clip = np.transpose(clip, (3, 0, 1, 2))
        
        return torch.from_numpy(clip)

def decode_video_clips(video_path: str, target_fps: int = 10, clip_length: int = 16, clip_stride: int = 8) -> List[Tuple[np.ndarray, float, float]]:
    """
    Decodes a video using OpenCV at the target FPS and extracts sliding window clips.
    Args:
        video_path (str): Path to the video file.
        target_fps (int): FPS to downsample the video to.
        clip_length (int): Number of frames per clip.
        clip_stride (int): Frame stride for the sliding window.
    Returns:
        List of tuples containing (clip_frames (T, H, W, C), start_time_sec, end_time_sec)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0 # fallback if metadata is missing

    # Calculate frame stride to achieve the target_fps from the original_fps
    frame_stride = max(1, round(original_fps / target_fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_stride == 0:
            # OpenCV loads in BGR; convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        frame_count += 1
        
    cap.release()
    
    num_frames = len(frames)
    if num_frames < clip_length:
        return []
        
    clips = []
    # Sliding window extraction
    for start_idx in range(0, num_frames - clip_length + 1, clip_stride):
        end_idx = start_idx + clip_length
        clip_frames = np.stack(frames[start_idx:end_idx], axis=0)
        
        # Calculate timestamps based on target_fps
        start_time_sec = start_idx / target_fps
        end_time_sec = end_idx / target_fps
        
        clips.append((clip_frames, start_time_sec, end_time_sec))
        
    return clips

def temporal_segment_sampling(total_frames: int, num_segments: int = 32, clip_length: int = 16, jitter: bool = True) -> List[int]:
    """
    Divides the video into num_segments equal segments for Multiple Instance Learning (Sultani-style bag sampling).
    From each segment, picks one starting frame index for a clip.
    Args:
        total_frames (int): Total number of frames in the video.
        num_segments (int): Number of segments (MIL bags).
        clip_length (int): Length of each clip in frames.
        jitter (bool): If True, randomly sample within the segment. If False, pick center.
    Returns:
        List[int]: List of starting frame indices for each segment.
    """
    # Number of valid starting indices for a clip
    valid_length = total_frames - clip_length + 1
    
    if valid_length <= 0:
        return []
        
    segment_size = valid_length / num_segments
    start_indices = []
    
    for i in range(num_segments):
        start_seg = i * segment_size
        end_seg = (i + 1) * segment_size
        
        if jitter:
            idx = random.uniform(start_seg, end_seg)
        else:
            idx = (start_seg + end_seg) / 2
            
        idx = int(idx)
        # Ensure the index is within valid bounds
        idx = max(0, min(idx, valid_length - 1))
        start_indices.append(idx)
        
    return start_indices
