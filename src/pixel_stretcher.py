import numpy as np
from PIL import Image
import imageio
from typing import Optional, Union, Literal, List, Tuple, Callable
from pathlib import Path
from .distortion_effects import (
    BiasedStretchEffect, WaveDistortionEffect, PivotStretchEffect
)


class PixelStretcher:
    def __init__(
        self,
        max_stretch: float = 0.5,
        pivot: Literal['center', 'top', 'bottom'] = 'center',
        interpolation: Literal['nearest', 'bilinear'] = 'nearest',
        temporal_smoothing: float = 0.0,
        seed: Optional[int] = None,
        upscale: int = 1,
        cumulative: bool = False,
        wave_amplitude: float = 0.1,
        wave_frequency: float = 2.0,
        wave_phase_shift: float = 0.0,
        wave_phase_speed: float = 0.2,
        use_wave_distortion: bool = False,
        stretch_bias: float = 0.0,
        stretch_curve: Literal['linear', 'constant', 'ease_in', 'ease_out', 'ease_in_out', 'custom'] = 'linear',
        start_stretch: Optional[float] = None,
        end_stretch: Optional[float] = None,
        custom_curve_func: Optional[Callable[[float], float]] = None
    ):
        self.max_stretch = max_stretch
        self.pivot = pivot
        self.interpolation = interpolation
        self.temporal_smoothing = temporal_smoothing
        self.upscale = max(1, int(upscale))
        self.cumulative = cumulative
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_phase_shift = wave_phase_shift
        self.wave_phase_speed = wave_phase_speed
        self.use_wave_distortion = use_wave_distortion
        self.stretch_bias = np.clip(stretch_bias, -1.0, 1.0)  # Clamp between -1 and 1
        self.stretch_curve = stretch_curve
        self.start_stretch = start_stretch if start_stretch is not None else 0.0
        self.end_stretch = end_stretch if end_stretch is not None else max_stretch
        self.custom_curve_func = custom_curve_func
        self.rng = np.random.RandomState(seed)
        self._previous_factors = None
        
    def _get_pivot_point(self, height: int) -> float:
        if self.pivot == 'center':
            return height / 2
        elif self.pivot == 'top':
            return 0
        else:  # bottom
            return height
    
    def _calculate_stretch_scale(self, t: float) -> float:
        """Calculate the stretch scale based on the curve type and progress t (0 to 1)."""
        if self.stretch_curve == 'constant':
            # Always use end_stretch (which defaults to max_stretch)
            curve_value = 1.0
        elif self.stretch_curve == 'linear':
            # Linear interpolation (default behavior)
            curve_value = t
        elif self.stretch_curve == 'ease_in':
            # Start slow, accelerate (quadratic)
            curve_value = t * t
        elif self.stretch_curve == 'ease_out':
            # Start fast, decelerate
            curve_value = t * (2.0 - t)
        elif self.stretch_curve == 'ease_in_out':
            # S-curve (smooth at both ends)
            curve_value = t * t * (3.0 - 2.0 * t)
        elif self.stretch_curve == 'custom' and self.custom_curve_func:
            # Use custom function
            curve_value = self.custom_curve_func(t)
        else:
            # Fallback to linear
            curve_value = t
        
        # Interpolate between start_stretch and end_stretch
        return self.start_stretch + (self.end_stretch - self.start_stretch) * curve_value
    
    def _generate_stretch_factors(self, width: int, stretch_scale: float = 1.0) -> np.ndarray:
        # Generate base random values between 0 and 1
        base_random = self.rng.rand(width)
        
        # Apply bias to determine stretch direction
        if self.stretch_bias == 0:
            # No bias: uniform distribution between -1 and 1
            factors = (base_random * 2 - 1)
        else:
            # With bias: mostly one direction with occasional opposite
            factors = np.zeros(width)
            
            for i in range(width):
                # Determine probability of stretching in the biased direction
                bias_probability = 0.5 + abs(self.stretch_bias) * 0.4  # 50% to 90% based on bias
                
                if self.rng.rand() < bias_probability:
                    # Stretch in the biased direction
                    if self.stretch_bias > 0:
                        factors[i] = base_random[i]  # Positive (downward stretch)
                    else:
                        factors[i] = -base_random[i]  # Negative (upward stretch)
                else:
                    # Occasional stretch in opposite direction (reduced magnitude)
                    if self.stretch_bias > 0:
                        factors[i] = -base_random[i] * 0.3  # Small upward movement
                    else:
                        factors[i] = base_random[i] * 0.3  # Small downward movement
        
        # Apply stretch scale directly (it already incorporates start/end stretch)
        factors = factors * stretch_scale
        
        if self.temporal_smoothing > 0 and self._previous_factors is not None:
            factors = (self.temporal_smoothing * self._previous_factors + 
                      (1 - self.temporal_smoothing) * factors)
        
        self._previous_factors = factors.copy()
        return factors
    
    def _warp_column_nearest(
        self, 
        column: np.ndarray, 
        factor: float, 
        pivot: float
    ) -> np.ndarray:
        height = len(column)
        output_rows = np.arange(height)
        
        # For directional stretching (melting effect)
        if factor > 0:  # Downward stretch
            # Each pixel pulls from above proportionally to its position
            source_rows = output_rows - factor * height * (output_rows / height)
        else:  # Upward stretch
            # Each pixel pulls from below proportionally to its position
            source_rows = output_rows - factor * height * (1 - output_rows / height)
        
        # Clip to valid range and round to nearest integer
        source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
        
        return column[source_rows]
    
    def _warp_column_bilinear(
        self, 
        column: np.ndarray, 
        factor: float, 
        pivot: float
    ) -> np.ndarray:
        height = len(column)
        output_rows = np.arange(height)
        
        # For directional stretching (melting effect)
        if factor > 0:  # Downward stretch
            # Each pixel pulls from above proportionally to its position
            source_rows = output_rows - factor * height * (output_rows / height)
        else:  # Upward stretch
            # Each pixel pulls from below proportionally to its position
            source_rows = output_rows - factor * height * (1 - output_rows / height)
        
        # Clip to valid range
        source_rows = np.clip(source_rows, 0, height - 1)
        
        # Bilinear interpolation
        lower_idx = np.floor(source_rows).astype(int)
        upper_idx = np.minimum(lower_idx + 1, height - 1)
        weights = source_rows - lower_idx
        
        # Handle multi-channel images
        if len(column.shape) > 1:
            result = (column[lower_idx] * (1 - weights)[:, np.newaxis] + 
                     column[upper_idx] * weights[:, np.newaxis])
        else:
            result = column[lower_idx] * (1 - weights) + column[upper_idx] * weights
            
        return result.astype(column.dtype)
    
    def _warp_column_wave_nearest(
        self,
        column: np.ndarray,
        factor: float,
        column_index: int,
        total_width: int,
        time_phase: float = 0.0
    ) -> np.ndarray:
        height = len(column)
        output_rows = np.arange(height)
        
        # Calculate wave-based displacement for each row
        normalized_y = output_rows / height  # 0 to 1
        normalized_x = column_index / total_width  # 0 to 1
        
        # Create wave pattern with animated phase
        wave = np.sin(2 * np.pi * self.wave_frequency * normalized_y + 
                     self.wave_phase_shift + normalized_x * np.pi + time_phase)
        
        # Apply displacement
        displacement = wave * self.wave_amplitude * height * factor
        source_rows = output_rows + displacement
        
        # Clip to valid range and round to nearest integer
        source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
        
        return column[source_rows]
    
    def _warp_column_wave_bilinear(
        self,
        column: np.ndarray,
        factor: float,
        column_index: int,
        total_width: int,
        time_phase: float = 0.0
    ) -> np.ndarray:
        height = len(column)
        output_rows = np.arange(height)
        
        # Calculate wave-based displacement for each row
        normalized_y = output_rows / height  # 0 to 1
        normalized_x = column_index / total_width  # 0 to 1
        
        # Create wave pattern with animated phase
        wave = np.sin(2 * np.pi * self.wave_frequency * normalized_y + 
                     self.wave_phase_shift + normalized_x * np.pi + time_phase)
        
        # Apply displacement
        displacement = wave * self.wave_amplitude * height * factor
        source_rows = output_rows + displacement
        
        # Clip to valid range
        source_rows = np.clip(source_rows, 0, height - 1)
        
        # Bilinear interpolation
        lower_idx = np.floor(source_rows).astype(int)
        upper_idx = np.minimum(lower_idx + 1, height - 1)
        weights = source_rows - lower_idx
        
        # Handle multi-channel images
        if len(column.shape) > 1:
            result = (column[lower_idx] * (1 - weights)[:, np.newaxis] + 
                     column[upper_idx] * weights[:, np.newaxis])
        else:
            result = column[lower_idx] * (1 - weights) + column[upper_idx] * weights
            
        return result.astype(column.dtype)
    
    def _upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using nearest neighbor interpolation."""
        if self.upscale <= 1:
            return image
            
        height, width = image.shape[:2]
        upscaled_height = height * self.upscale
        upscaled_width = width * self.upscale
        upscaled_output = np.zeros((upscaled_height, upscaled_width) + image.shape[2:], dtype=image.dtype)
        
        # Use nearest neighbor for upscaling
        for y in range(upscaled_height):
            for x in range(upscaled_width):
                src_y = y // self.upscale
                src_x = x // self.upscale
                upscaled_output[y, x] = image[src_y, src_x]
        
        return upscaled_output
    
    def warp_frame(self, image: np.ndarray, stretch_scale: float = 1.0, apply_upscale: bool = True, time_phase: float = 0.0) -> np.ndarray:
        height, width = image.shape[:2]
        output = np.zeros_like(image)
        
        # Generate stretch factors for each column
        factors = self._generate_stretch_factors(width, stretch_scale)
        pivot = self._get_pivot_point(height)
        
        # Warp each column
        for x in range(width):
            if self.use_wave_distortion:
                # Use wave-based distortion
                if self.interpolation == 'nearest':
                    output[:, x] = self._warp_column_wave_nearest(image[:, x], factors[x], x, width, time_phase)
                else:
                    output[:, x] = self._warp_column_wave_bilinear(image[:, x], factors[x], x, width, time_phase)
            else:
                # Use original pivot-based distortion
                if self.interpolation == 'nearest':
                    output[:, x] = self._warp_column_nearest(image[:, x], factors[x], pivot)
                else:
                    output[:, x] = self._warp_column_bilinear(image[:, x], factors[x], pivot)
        
        # Apply upscaling if requested
        if apply_upscale and self.upscale > 1:
            return self._upscale_image(output)
        
        return output
    
    def create_animation(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        frames: int = 60,
        fps: int = 30,
        loop: int = 0
    ) -> None:
        # Load input image
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        img = Image.open(input_path)
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        # Generate frames
        frame_list = []
        
        # Generate frames with interpolated stretch
        if self.cumulative:
            # For cumulative mode, work at original resolution and upscale at the end
            last_frame = img_array.copy()
            
            for i in range(frames):
                # Calculate time-based phase for wave animation
                time_phase = (i / max(frames - 1, 1)) * 2 * np.pi * self.wave_phase_speed
                
                if i == 0:
                    # First frame is the original image (upscaled if needed)
                    frame_list.append(self._upscale_image(img_array))
                else:
                    # Calculate progress t from 0 to 1
                    t = i / max(frames - 1, 1)
                    # Get the current stretch value from the curve
                    current_stretch = self._calculate_stretch_scale(t)
                    # For cumulative mode, calculate the incremental stretch
                    # This is the difference from the previous frame
                    prev_t = (i - 1) / max(frames - 1, 1)
                    prev_stretch = self._calculate_stretch_scale(prev_t)
                    incremental_stretch = current_stretch - prev_stretch
                    
                    # Apply only the incremental distortion to the last frame
                    warped_frame = self.warp_frame(last_frame, incremental_stretch, apply_upscale=False, time_phase=time_phase)
                    last_frame = warped_frame.copy()
                    # Upscale the result for output
                    frame_list.append(self._upscale_image(warped_frame))
        else:
            # Non-cumulative mode: apply increasing distortion to original image
            for i in range(frames):
                # Calculate progress t from 0 to 1
                t = i / max(frames - 1, 1)
                # Use curve to calculate actual stretch scale
                stretch_scale = self._calculate_stretch_scale(t)
                # Calculate time-based phase for wave animation
                time_phase = t * 2 * np.pi * self.wave_phase_speed
                
                if i == 0 and self.start_stretch == 0:
                    # First frame is the original image only if start_stretch is 0
                    frame_list.append(self._upscale_image(img_array))
                else:
                    # Apply distortion to the original image with upscaling
                    warped_frame = self.warp_frame(img_array, stretch_scale, apply_upscale=True, time_phase=time_phase)
                    frame_list.append(warped_frame)
        
        # Save animation as video
        imageio.mimsave(
            output_path,
            frame_list,
            fps=fps
        )
    
    def reset(self) -> None:
        self._previous_factors = None