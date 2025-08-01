"""
Post-processing effects for video frames.

Post-processors are applied after the main animation generation phase,
allowing for additional effects to be applied to the entire video.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Literal, Union
from scipy.ndimage import rotate
import scipy.ndimage as ndimage
from .transforms import SineWaveTransform, DisplacementCalculator, EdgeHandler


class PostProcessor(ABC):
    """Abstract base class for all post-processing effects."""
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_index: int, 
                     total_frames: int, **kwargs) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            frame_index: Current frame index (0-based)
            total_frames: Total number of frames in the video
            **kwargs: Additional parameters that may be passed from the pipeline
            
        Returns:
            Processed frame as numpy array
        """
        pass
    
    def get_progress(self, frame_index: int, total_frames: int) -> float:
        """
        Calculate normalized progress (0.0 to 1.0) for the current frame.
        
        Args:
            frame_index: Current frame index
            total_frames: Total number of frames
            
        Returns:
            Progress value between 0.0 and 1.0
        """
        return frame_index / max(total_frames - 1, 1) if total_frames > 1 else 0.0


class PostProcessorChain(PostProcessor):
    """Chain multiple post-processors together."""
    
    def __init__(self, processors: List[PostProcessor]):
        """
        Initialize the chain with a list of processors.
        
        Args:
            processors: List of PostProcessor instances to apply in sequence
        """
        self.processors = processors
    
    def process_frame(self, frame: np.ndarray, frame_index: int,
                     total_frames: int, **kwargs) -> np.ndarray:
        """Apply all processors in sequence."""
        result = frame
        for processor in self.processors:
            result = processor.process_frame(result, frame_index, total_frames, **kwargs)
        return result


class IdentityPostProcessor(PostProcessor):
    """Pass-through processor that returns frames unchanged."""
    
    def process_frame(self, frame: np.ndarray, frame_index: int,
                     total_frames: int, **kwargs) -> np.ndarray:
        """Return frame unchanged."""
        return frame


class UpscalePostProcessor(PostProcessor):
    """Upscale frames by an integer factor."""
    
    def __init__(self, scale_factor: int = 2, 
                 method: Literal['nearest', 'bilinear', 'cubic'] = 'nearest'):
        """
        Initialize upscaler.
        
        Args:
            scale_factor: Integer scale factor (2x, 3x, 4x, etc.)
            method: Interpolation method
        """
        self.scale_factor = scale_factor
        self.method = method
        
        # Map method names to scipy order values
        self.method_map = {
            'nearest': 0,
            'bilinear': 1,
            'cubic': 3
        }
    
    def process_frame(self, frame: np.ndarray, frame_index: int,
                     total_frames: int, **kwargs) -> np.ndarray:
        """Upscale the frame."""
        if self.scale_factor <= 1:
            return frame
        
        if self.method == 'nearest':
            # Use fast nearest neighbor for better performance
            return self._upscale_nearest(frame)
        else:
            # Use scipy zoom for other methods
            order = self.method_map[self.method]
            return ndimage.zoom(frame, (self.scale_factor, self.scale_factor, 1), 
                              order=order, mode='nearest')
    
    def _upscale_nearest(self, frame: np.ndarray) -> np.ndarray:
        """Fast nearest neighbor upscaling."""
        height, width = frame.shape[:2]
        upscaled_height = height * self.scale_factor
        upscaled_width = width * self.scale_factor
        
        upscaled = np.zeros((upscaled_height, upscaled_width) + frame.shape[2:], 
                           dtype=frame.dtype)
        
        for y in range(upscaled_height):
            for x in range(upscaled_width):
                src_y = y // self.scale_factor
                src_x = x // self.scale_factor
                upscaled[y, x] = frame[src_y, src_x]
        
        return upscaled


class SineWavePostProcessor(PostProcessor):
    """Apply sine wave distortion to video frames using shared transforms."""
    
    def __init__(self, 
                 axis: Union[Literal['vertical', 'horizontal', 'diagonal'], float] = 'vertical',
                 angle: Optional[float] = None,
                 frequency: float = 3.0,
                 amplitude: float = 0.05,
                 phase: float = 0.0,
                 speed: float = 0.5,
                 displacement_mode: Literal['translate', 'scale', 'both'] = 'translate',
                 edge_behavior: Literal['wrap', 'clamp', 'fade', 'mirror'] = 'wrap',
                 amplitude_curve: Literal['constant', 'linear', 'ease_in', 'ease_out', 'ease_in_out'] = 'constant',
                 start_amplitude: Optional[float] = None,
                 end_amplitude: Optional[float] = None):
        """
        Initialize sine wave post-processor.
        
        Args:
            axis: Displacement axis ('vertical', 'horizontal', 'diagonal', or angle in degrees)
            angle: Custom angle in degrees (overrides axis if provided)
            frequency: Number of wave cycles
            amplitude: Wave amplitude (0.0 to 1.0, as fraction of dimension)
            phase: Initial phase offset in radians
            speed: Animation speed (phase change per frame)
            displacement_mode: How to displace pixels
            edge_behavior: How to handle edges
            amplitude_curve: How amplitude changes over time
            start_amplitude: Starting amplitude (defaults to amplitude)
            end_amplitude: Ending amplitude (defaults to amplitude)
        """
        # Handle axis/angle specification
        if angle is not None:
            axis_value = angle
        elif isinstance(axis, (int, float)):
            axis_value = axis
        else:
            axis_value = axis
            
        # Initialize shared transform
        self.transform = SineWaveTransform(
            frequency=frequency,
            amplitude=amplitude,
            phase=phase,
            speed=speed,
            axis=axis_value
        )
        
        self.displacement_mode = displacement_mode
        self.edge_behavior = edge_behavior
        self.amplitude_curve = amplitude_curve
        self.start_amplitude = start_amplitude if start_amplitude is not None else amplitude
        self.end_amplitude = end_amplitude if end_amplitude is not None else amplitude
        
        # Store original amplitude for curve calculations
        self.base_amplitude = amplitude
    
    def process_frame(self, frame: np.ndarray, frame_index: int,
                     total_frames: int, **kwargs) -> np.ndarray:
        """Apply sine wave distortion to the frame."""
        progress = self.get_progress(frame_index, total_frames)
        
        # Update transform amplitude based on curve
        current_amplitude = self.transform.get_amplitude_at_time(
            progress, 
            self.amplitude_curve,
            self.start_amplitude,
            self.end_amplitude
        )
        
        # Calculate displacement field
        height, width = frame.shape[:2]
        time_factor = frame_index / max(total_frames - 1, 1) if total_frames > 1 else 0.0
        dx, dy = self.transform.calculate_frame_displacement(width, height, time_factor)
        
        # Scale amplitude (transform returns normalized values)
        dx *= current_amplitude / self.base_amplitude
        dy *= current_amplitude / self.base_amplitude
        
        # Apply displacement with edge handling
        result = self._apply_displacement_with_edge_handling(frame, dx, dy)
        
        return result
    
    def _apply_displacement_with_edge_handling(self, frame: np.ndarray, 
                                               dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Apply displacement with edge handling using shared utilities."""
        height, width = frame.shape[:2]
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply displacement to get source coordinates
        src_x = x - dx
        src_y = y - dy
        
        # Apply edge behavior
        src_x, src_y, alpha_mask = EdgeHandler.apply_2d_edge_behavior(
            src_x, src_y, width, height, 
            self.edge_behavior, fade_width=0.1
        )
        
        # Apply displacement using shared calculator
        result = DisplacementCalculator.apply_displacement(
            frame, -dx, -dy,  # Negative because we calculated source coords
            mode=self.displacement_mode,
            order=1  # Bilinear interpolation
        )
        
        # Apply alpha mask if using fade
        if self.edge_behavior == 'fade' and alpha_mask is not None:
            if len(result.shape) == 3:
                # Apply to each channel
                for c in range(result.shape[2]):
                    result[:, :, c] = result[:, :, c] * alpha_mask
            else:
                result = result * alpha_mask
        
        return result