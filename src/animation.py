"""
Animation generation and frame management module.
"""

import numpy as np
from PIL import Image
import imageio
from typing import Union, List, Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod


class FrameGenerator(ABC):
    """Base class for frame generation strategies."""
    
    @abstractmethod
    def generate_frames(self, base_image: np.ndarray, num_frames: int, 
                       warp_func: Callable) -> List[np.ndarray]:
        """Generate animation frames."""
        pass


class StandardFrameGenerator(FrameGenerator):
    """Standard frame generation - applies increasing distortion."""
    
    def generate_frames(self, base_image: np.ndarray, num_frames: int,
                       warp_func: Callable) -> List[np.ndarray]:
        frames = []
        
        for i in range(num_frames):
            if i == 0:
                # First frame is the original image
                frames.append(base_image.copy())
            else:
                # Apply distortion to the original image
                warped_frame = warp_func(base_image, frame=i, total_frames=num_frames)
                frames.append(warped_frame)
        
        return frames


class CumulativeFrameGenerator(FrameGenerator):
    """Cumulative frame generation - each frame builds on the previous."""
    
    def generate_frames(self, base_image: np.ndarray, num_frames: int,
                       warp_func: Callable) -> List[np.ndarray]:
        frames = []
        last_frame = base_image.copy()
        
        for i in range(num_frames):
            if i == 0:
                # First frame is the original image
                frames.append(base_image.copy())
            else:
                # Apply distortion to the last frame
                warped_frame = warp_func(last_frame, frame=i, total_frames=num_frames)
                last_frame = warped_frame.copy()
                frames.append(warped_frame)
        
        return frames


class PingPongFrameGenerator(FrameGenerator):
    """Generate frames that play forward then backward."""
    
    def generate_frames(self, base_image: np.ndarray, num_frames: int,
                       warp_func: Callable) -> List[np.ndarray]:
        # Generate forward frames
        forward_frames = []
        half_frames = num_frames // 2
        
        for i in range(half_frames):
            if i == 0:
                forward_frames.append(base_image.copy())
            else:
                warped_frame = warp_func(base_image, frame=i, total_frames=half_frames)
                forward_frames.append(warped_frame)
        
        # Create full sequence: forward + backward
        frames = forward_frames + forward_frames[-2::-1]
        
        # Ensure we have exactly num_frames
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        return frames[:num_frames]


class AnimationEngine:
    """Main animation engine for creating pixel stretching animations."""
    
    def __init__(self, frame_generator: Optional[FrameGenerator] = None):
        self.frame_generator = frame_generator or StandardFrameGenerator()
    
    def load_image(self, input_path: Union[str, Path]) -> np.ndarray:
        """Load and prepare image for animation."""
        input_path = Path(input_path)
        
        img = Image.open(input_path)
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        return np.array(img)
    
    def save_animation(self, frames: List[np.ndarray], output_path: Union[str, Path],
                      fps: int = 30, codec: str = 'libx264', quality: Optional[int] = None):
        """Save frames as video animation."""
        output_path = Path(output_path)
        
        # Determine save parameters based on format
        if output_path.suffix.lower() == '.mp4':
            save_kwargs = {
                'fps': fps,
                'codec': codec,
            }
            if quality:
                save_kwargs['quality'] = quality
        else:
            save_kwargs = {'fps': fps}
        
        imageio.mimsave(output_path, frames, **save_kwargs)
    
    def create_animation(self, input_path: Union[str, Path], output_path: Union[str, Path],
                        warp_func: Callable, frames: int = 60, fps: int = 30,
                        post_process: Optional[Callable] = None) -> None:
        """Create complete animation from input image."""
        # Load image
        base_image = self.load_image(input_path)
        
        # Generate frames
        frame_list = self.frame_generator.generate_frames(base_image, frames, warp_func)
        
        # Apply post-processing if provided
        if post_process:
            frame_list = [post_process(frame) for frame in frame_list]
        
        # Save animation
        self.save_animation(frame_list, output_path, fps)


class AnimationSequence:
    """Manage complex animation sequences with multiple effects."""
    
    def __init__(self):
        self.segments = []
    
    def add_segment(self, effect_func: Callable, frames: int, 
                   transition_frames: int = 0):
        """Add an animation segment."""
        self.segments.append({
            'effect': effect_func,
            'frames': frames,
            'transition_frames': transition_frames
        })
    
    def generate_frames(self, base_image: np.ndarray) -> List[np.ndarray]:
        """Generate all frames for the sequence."""
        all_frames = []
        
        for i, segment in enumerate(self.segments):
            segment_frames = []
            
            # Generate frames for this segment
            for frame in range(segment['frames']):
                t = frame / max(segment['frames'] - 1, 1)
                warped = segment['effect'](base_image, t)
                segment_frames.append(warped)
            
            # Add transition if not the last segment
            if i < len(self.segments) - 1 and segment['transition_frames'] > 0:
                next_effect = self.segments[i + 1]['effect']
                for t_frame in range(segment['transition_frames']):
                    t = t_frame / segment['transition_frames']
                    # Blend between current and next effect
                    current = segment['effect'](base_image, 1.0)
                    next_frame = next_effect(base_image, 0.0)
                    blended = (1 - t) * current + t * next_frame
                    segment_frames.append(blended.astype(np.uint8))
            
            all_frames.extend(segment_frames)
        
        return all_frames