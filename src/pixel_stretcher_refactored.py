import numpy as np
from PIL import Image
import imageio
from typing import Optional, Union, Literal
from pathlib import Path
from .distortion_effects import (
    DistortionEffect, BiasedStretchEffect, WaveDistortionEffect, 
    PivotStretchEffect, CompositeEffect
)


class PixelStretcher:
    def __init__(
        self,
        effect: Optional[DistortionEffect] = None,
        interpolation: Literal['nearest', 'bilinear'] = 'nearest',
        temporal_smoothing: float = 0.0,
        upscale: int = 1,
        cumulative: bool = False
    ):
        self.effect = effect or PivotStretchEffect()  # Default to pivot effect
        self.interpolation = interpolation
        self.temporal_smoothing = temporal_smoothing
        self.upscale = max(1, int(upscale))
        self.cumulative = cumulative
        self._previous_factors = None
        
    @classmethod
    def from_legacy_params(
        cls,
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
        stretch_bias: float = 0.0
    ):
        """Create PixelStretcher from legacy parameters for backward compatibility."""
        if use_wave_distortion:
            effect = WaveDistortionEffect(
                max_stretch=max_stretch,
                wave_amplitude=wave_amplitude,
                wave_frequency=wave_frequency,
                wave_phase_shift=wave_phase_shift,
                wave_phase_speed=wave_phase_speed,
                seed=seed
            )
        elif stretch_bias != 0.0:
            effect = BiasedStretchEffect(
                max_stretch=max_stretch,
                stretch_bias=stretch_bias,
                seed=seed
            )
        else:
            effect = PivotStretchEffect(
                max_stretch=max_stretch,
                pivot=pivot,
                seed=seed
            )
        
        return cls(
            effect=effect,
            interpolation=interpolation,
            temporal_smoothing=temporal_smoothing,
            upscale=upscale,
            cumulative=cumulative
        )
    
    def _apply_temporal_smoothing(self, factors: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to stretch factors."""
        if self.temporal_smoothing > 0 and self._previous_factors is not None:
            factors = (self.temporal_smoothing * self._previous_factors + 
                      (1 - self.temporal_smoothing) * factors)
        
        self._previous_factors = factors.copy()
        return factors
    
    def _interpolate_column(self, column: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
        """Apply interpolation to column warping."""
        height = len(column)
        
        if self.interpolation == 'nearest':
            # Round to nearest integer
            source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
            return column[source_rows]
        else:  # bilinear
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
    
    def warp_frame(self, image: np.ndarray, frame: int = 0, total_frames: int = 60, 
                  apply_upscale: bool = True) -> np.ndarray:
        height, width = image.shape[:2]
        output = np.zeros_like(image)
        
        # Generate stretch factors for each column
        factors = self.effect.generate_factors(width, frame, total_frames)
        factors = self._apply_temporal_smoothing(factors)
        
        # Warp each column
        for x in range(width):
            warped_column = self.effect.warp_column(
                image[:, x], factors[x], x, width, height
            )
            
            # Apply interpolation if needed
            if self.interpolation == 'bilinear' and hasattr(self.effect, 'get_source_rows'):
                # For effects that provide continuous source rows
                output[:, x] = self._interpolate_column(image[:, x], warped_column)
            else:
                output[:, x] = warped_column
        
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
        
        if self.cumulative:
            # For cumulative mode, work at original resolution and upscale at the end
            last_frame = img_array.copy()
            
            for i in range(frames):
                if i == 0:
                    # First frame is the original image (upscaled if needed)
                    frame_list.append(self._upscale_image(img_array))
                else:
                    # Apply distortion to the last frame
                    warped_frame = self.warp_frame(last_frame, i, frames, apply_upscale=False)
                    last_frame = warped_frame.copy()
                    # Upscale the result for output
                    frame_list.append(self._upscale_image(warped_frame))
        else:
            # Non-cumulative mode: apply increasing distortion to original image
            for i in range(frames):
                if i == 0:
                    # First frame is the original image
                    frame_list.append(self._upscale_image(img_array))
                else:
                    # Apply distortion to the original image with upscaling
                    warped_frame = self.warp_frame(img_array, i, frames, apply_upscale=True)
                    frame_list.append(warped_frame)
        
        # Save animation as video
        imageio.mimsave(
            output_path,
            frame_list,
            fps=fps
        )
    
    def reset(self) -> None:
        self._previous_factors = None