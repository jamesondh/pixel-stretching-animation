import numpy as np
from PIL import Image
import imageio
from typing import Optional, Union, Literal, List, Tuple, Callable
from pathlib import Path
from .distortion_effects import (
    BiasedStretchEffect, WaveDistortionEffect, PivotStretchEffect, DistortionEffect
)


class PixelStretcher:
    """
    Unified pixel stretcher that supports both legacy built-in distortion
    and the new effect object system.
    """
    
    def __init__(
        self,
        # Effect-based parameters (v2 mode)
        effect: Optional[DistortionEffect] = None,
        
        # Legacy parameters (v1 mode)
        max_stretch: float = 0.5,
        pivot: Literal['center', 'top', 'bottom'] = 'center',
        wave_amplitude: float = 0.1,
        wave_frequency: float = 2.0,
        wave_phase_shift: float = 0.0,
        wave_phase_speed: float = 0.2,
        use_wave_distortion: bool = False,
        stretch_bias: float = 0.0,
        
        # Common parameters
        interpolation: Literal['nearest', 'bilinear'] = 'nearest',
        temporal_smoothing: float = 0.0,
        seed: Optional[int] = None,
        upscale: int = 1,
        cumulative: bool = False,
        stretch_curve: Literal['linear', 'constant', 'ease_in', 'ease_out', 'ease_in_out', 'custom'] = 'linear',
        start_stretch: Optional[float] = None,
        end_stretch: Optional[float] = None,
        custom_curve_func: Optional[Callable[[float], float]] = None
    ):
        # Store effect if provided (v2 mode)
        self.effect = effect
        
        # Store legacy parameters
        self.max_stretch = max_stretch
        self.pivot = pivot
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_phase_shift = wave_phase_shift
        self.wave_phase_speed = wave_phase_speed
        self.use_wave_distortion = use_wave_distortion
        self.stretch_bias = np.clip(stretch_bias, -1.0, 1.0)
        
        # Common parameters
        self.interpolation = interpolation
        self.temporal_smoothing = temporal_smoothing
        self.upscale = max(1, int(upscale))
        self.cumulative = cumulative
        self.stretch_curve = stretch_curve
        self.start_stretch = start_stretch if start_stretch is not None else 0.0
        self.end_stretch = end_stretch if end_stretch is not None else (1.0 if effect else max_stretch)
        self.custom_curve_func = custom_curve_func
        self.rng = np.random.RandomState(seed)
        self._previous_factors = None
        
    def _get_pivot_point(self, height: int) -> float:
        """Get pivot point for legacy mode."""
        if self.pivot == 'center':
            return height / 2
        elif self.pivot == 'top':
            return 0
        else:  # bottom
            return height
    
    def _calculate_stretch_scale(self, t: float) -> float:
        """Calculate the stretch scale based on the curve type and progress t (0 to 1)."""
        if self.stretch_curve == 'constant':
            curve_value = 1.0
        elif self.stretch_curve == 'linear':
            curve_value = t
        elif self.stretch_curve == 'ease_in':
            curve_value = t * t
        elif self.stretch_curve == 'ease_out':
            curve_value = t * (2.0 - t)
        elif self.stretch_curve == 'ease_in_out':
            curve_value = t * t * (3.0 - 2.0 * t)
        elif self.stretch_curve == 'custom' and self.custom_curve_func:
            curve_value = self.custom_curve_func(t)
        else:
            curve_value = t
        
        return self.start_stretch + (self.end_stretch - self.start_stretch) * curve_value
    
    def _generate_stretch_factors(self, width: int, stretch_scale: float = 1.0) -> np.ndarray:
        """Generate stretch factors for legacy mode."""
        # Generate base random values between 0 and 1
        base_random = self.rng.rand(width)
        
        # Apply bias to determine stretch direction
        if self.stretch_bias == 0:
            factors = (base_random * 2 - 1)
        else:
            factors = np.zeros(width)
            
            for i in range(width):
                bias_probability = 0.5 + abs(self.stretch_bias) * 0.4
                
                if self.rng.rand() < bias_probability:
                    if self.stretch_bias > 0:
                        factors[i] = base_random[i]
                    else:
                        factors[i] = -base_random[i]
                else:
                    if self.stretch_bias > 0:
                        factors[i] = -base_random[i] * 0.3
                    else:
                        factors[i] = base_random[i] * 0.3
        
        factors = factors * stretch_scale
        
        if self.temporal_smoothing > 0 and self._previous_factors is not None:
            factors = (self.temporal_smoothing * self._previous_factors + 
                      (1 - self.temporal_smoothing) * factors)
        
        self._previous_factors = factors.copy()
        return factors
    
    def _apply_temporal_smoothing(self, factors: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to stretch factors."""
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
        """Legacy nearest neighbor warping."""
        height = len(column)
        output_rows = np.arange(height)
        
        if factor > 0:  # Downward stretch
            source_rows = output_rows - factor * height * (output_rows / height)
        else:  # Upward stretch
            source_rows = output_rows - factor * height * (1 - output_rows / height)
        
        source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
        
        return column[source_rows]
    
    def _warp_column_bilinear(
        self, 
        column: np.ndarray, 
        factor: float, 
        pivot: float
    ) -> np.ndarray:
        """Legacy bilinear interpolation warping."""
        height = len(column)
        output_rows = np.arange(height)
        
        if factor > 0:  # Downward stretch
            source_rows = output_rows - factor * height * (output_rows / height)
        else:  # Upward stretch
            source_rows = output_rows - factor * height * (1 - output_rows / height)
        
        source_rows = np.clip(source_rows, 0, height - 1)
        
        # Bilinear interpolation
        lower_idx = np.floor(source_rows).astype(int)
        upper_idx = np.minimum(lower_idx + 1, height - 1)
        weights = source_rows - lower_idx
        
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
        """Legacy wave-based nearest neighbor warping."""
        height = len(column)
        output_rows = np.arange(height)
        
        normalized_y = output_rows / height
        normalized_x = column_index / total_width
        
        wave = np.sin(2 * np.pi * self.wave_frequency * normalized_y + 
                     self.wave_phase_shift + normalized_x * np.pi + time_phase)
        
        displacement = wave * self.wave_amplitude * height * factor
        source_rows = output_rows + displacement
        
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
        """Legacy wave-based bilinear interpolation warping."""
        height = len(column)
        output_rows = np.arange(height)
        
        normalized_y = output_rows / height
        normalized_x = column_index / total_width
        
        wave = np.sin(2 * np.pi * self.wave_frequency * normalized_y + 
                     self.wave_phase_shift + normalized_x * np.pi + time_phase)
        
        displacement = wave * self.wave_amplitude * height * factor
        source_rows = output_rows + displacement
        
        source_rows = np.clip(source_rows, 0, height - 1)
        
        # Bilinear interpolation
        lower_idx = np.floor(source_rows).astype(int)
        upper_idx = np.minimum(lower_idx + 1, height - 1)
        weights = source_rows - lower_idx
        
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
        
        for y in range(upscaled_height):
            for x in range(upscaled_width):
                src_y = y // self.upscale
                src_x = x // self.upscale
                upscaled_output[y, x] = image[src_y, src_x]
        
        return upscaled_output
    
    def warp_frame(self, image: np.ndarray, stretch_scale: float = 1.0, 
                   apply_upscale: bool = True, time_phase: float = 0.0,
                   frame: int = 0, total_frames: int = 60) -> np.ndarray:
        """
        Warp a single frame using either effect object (v2) or legacy logic (v1).
        
        For v2 mode, stretch_scale parameter is ignored in favor of frame/total_frames.
        For v1 mode, frame/total_frames parameters are ignored.
        """
        height, width = image.shape[:2]
        output = np.zeros_like(image)
        
        if self.effect:
            # V2 mode: Use effect object
            t = frame / max(total_frames - 1, 1) if frame > 0 else 0
            stretch_scale = self._calculate_stretch_scale(t)
            
            # Generate factors using effect
            factors = self.effect.generate_factors(width, frame, total_frames)
            factors = factors * stretch_scale
            
            # Apply temporal smoothing
            factors = self._apply_temporal_smoothing(factors)
            
            # Warp each column using effect
            for x in range(width):
                output[:, x] = self.effect.warp_column(
                    image[:, x], factors[x], x, width, height
                )
        else:
            # V1 mode: Use legacy logic
            factors = self._generate_stretch_factors(width, stretch_scale)
            pivot = self._get_pivot_point(height)
            
            # Warp each column
            for x in range(width):
                if self.use_wave_distortion:
                    if self.interpolation == 'nearest':
                        output[:, x] = self._warp_column_wave_nearest(
                            image[:, x], factors[x], x, width, time_phase
                        )
                    else:
                        output[:, x] = self._warp_column_wave_bilinear(
                            image[:, x], factors[x], x, width, time_phase
                        )
                else:
                    if self.interpolation == 'nearest':
                        output[:, x] = self._warp_column_nearest(
                            image[:, x], factors[x], pivot
                        )
                    else:
                        output[:, x] = self._warp_column_bilinear(
                            image[:, x], factors[x], pivot
                        )
        
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
        """Create animation from input image."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        img = Image.open(input_path)
        if img.mode not in ['RGB', 'RGBA']:
            img = img.convert('RGB')
        
        img_array = np.array(img)
        
        frame_list = []
        
        if self.cumulative:
            # Cumulative mode: each frame builds on the previous
            last_frame = img_array.copy()
            
            for i in range(frames):
                if i == 0:
                    frame_list.append(self._upscale_image(img_array))
                else:
                    # Calculate incremental stretch
                    t = i / max(frames - 1, 1)
                    prev_t = (i - 1) / max(frames - 1, 1)
                    
                    current_stretch = self._calculate_stretch_scale(t)
                    prev_stretch = self._calculate_stretch_scale(prev_t)
                    incremental_stretch = current_stretch - prev_stretch
                    
                    if self.effect:
                        # V2 mode
                        warped_frame = self.warp_frame(
                            last_frame, frame=i, total_frames=frames, 
                            apply_upscale=False
                        )
                    else:
                        # V1 mode
                        time_phase = t * 2 * np.pi * self.wave_phase_speed
                        warped_frame = self.warp_frame(
                            last_frame, incremental_stretch, 
                            apply_upscale=False, time_phase=time_phase
                        )
                    
                    last_frame = warped_frame.copy()
                    frame_list.append(self._upscale_image(warped_frame))
        else:
            # Non-cumulative mode: apply increasing distortion to original
            for i in range(frames):
                t = i / max(frames - 1, 1)
                
                if i == 0 and self.start_stretch == 0:
                    frame_list.append(self._upscale_image(img_array))
                else:
                    if self.effect:
                        # V2 mode
                        warped_frame = self.warp_frame(
                            img_array, frame=i, total_frames=frames, 
                            apply_upscale=True
                        )
                    else:
                        # V1 mode
                        stretch_scale = self._calculate_stretch_scale(t)
                        time_phase = t * 2 * np.pi * self.wave_phase_speed
                        warped_frame = self.warp_frame(
                            img_array, stretch_scale, 
                            apply_upscale=True, time_phase=time_phase
                        )
                    
                    frame_list.append(warped_frame)
        
        # Save animation
        imageio.mimsave(output_path, frame_list, fps=fps)
    
    def reset(self) -> None:
        """Reset temporal state."""
        self._previous_factors = None


# Alias for backward compatibility
PixelStretcherV2 = PixelStretcher