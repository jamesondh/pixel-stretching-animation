import numpy as np
from typing import Optional, Literal, Union
from abc import ABC, abstractmethod
from scipy.ndimage import rotate


class DistortionEffect(ABC):
    """Base class for all distortion effects."""
    
    @abstractmethod
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        """Generate distortion factors for each column."""
        pass
    
    @abstractmethod
    def warp_column(self, column: np.ndarray, factor: float, column_index: int, 
                   total_width: int, height: int) -> np.ndarray:
        """Apply distortion to a single column."""
        pass


class BiasedStretchEffect(DistortionEffect):
    """Directional stretching with configurable bias."""
    
    def __init__(self, max_stretch: float = 0.5, stretch_bias: float = 0.0, 
                 seed: Optional[int] = None):
        self.max_stretch = max_stretch
        self.stretch_bias = np.clip(stretch_bias, -1.0, 1.0)
        self.rng = np.random.RandomState(seed)
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
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
        
        # Apply max stretch
        return factors * self.max_stretch
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
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


class WaveDistortionEffect(DistortionEffect):
    """Wave-based distortion with animated phase."""
    
    def __init__(self, max_stretch: float = 0.5, wave_amplitude: float = 0.1,
                 wave_frequency: float = 2.0, wave_phase_shift: float = 0.0,
                 wave_phase_speed: float = 0.2, seed: Optional[int] = None):
        self.max_stretch = max_stretch
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_phase_shift = wave_phase_shift
        self.wave_phase_speed = wave_phase_speed
        self.rng = np.random.RandomState(seed)
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        # Generate random factors that will be modulated by the wave
        return (self.rng.rand(width) * 2 - 1) * self.max_stretch
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
        output_rows = np.arange(height)
        
        # Calculate wave-based displacement for each row
        normalized_y = output_rows / height  # 0 to 1
        normalized_x = column_index / total_width  # 0 to 1
        
        # Calculate time phase based on current frame
        time_phase = self.wave_phase_speed * 2 * np.pi
        
        # Create wave pattern with animated phase
        wave = np.sin(2 * np.pi * self.wave_frequency * normalized_y + 
                     self.wave_phase_shift + normalized_x * np.pi + time_phase)
        
        # Apply displacement
        displacement = wave * self.wave_amplitude * height * factor
        source_rows = output_rows + displacement
        
        # Clip to valid range and round to nearest integer
        source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
        
        return column[source_rows]


class PivotStretchEffect(DistortionEffect):
    """Original pivot-based stretching effect."""
    
    def __init__(self, max_stretch: float = 0.5, 
                 pivot: Literal['center', 'top', 'bottom'] = 'center',
                 seed: Optional[int] = None):
        self.max_stretch = max_stretch
        self.pivot = pivot
        self.rng = np.random.RandomState(seed)
    
    def _get_pivot_point(self, height: int) -> float:
        if self.pivot == 'center':
            return height / 2
        elif self.pivot == 'top':
            return 0
        else:  # bottom
            return height
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        # Generate random stretch factors
        stretch_scale = frame / max(total_frames - 1, 1) if frame > 0 else 0
        return (self.rng.rand(width) * 2 - 1) * self.max_stretch * stretch_scale
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
        output_rows = np.arange(height)
        pivot = self._get_pivot_point(height)
        
        # Original pivot-based transformation
        distances = output_rows - pivot
        source_rows = pivot + distances / (1 + factor)
        
        # Clip to valid range and round to nearest integer
        source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
        
        return column[source_rows]


class CompositeEffect(DistortionEffect):
    """Combine multiple distortion effects."""
    
    def __init__(self, effects: list[DistortionEffect], weights: Optional[list[float]] = None,
                 stretch_curves: Optional[list[str]] = None):
        self.effects = effects
        self.weights = weights or [1.0 / len(effects)] * len(effects)
        self.stretch_curves = stretch_curves or ['linear'] * len(effects)
        assert len(self.weights) == len(self.effects), "Weights must match number of effects"
        assert len(self.stretch_curves) == len(self.effects), "Stretch curves must match number of effects"
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        # Combine factors from all effects
        combined_factors = np.zeros(width)
        for effect, weight in zip(self.effects, self.weights):
            combined_factors += effect.generate_factors(width, frame, total_frames) * weight
        return combined_factors
    
    def generate_factors_with_scale(self, width: int, frame: int, total_frames: int, 
                                   stretch_scale: float, end_stretch: float) -> np.ndarray:
        """Generate factors with per-effect stretch curves."""
        combined_factors = np.zeros(width)
        t = frame / max(total_frames - 1, 1) if frame > 0 else 0
        
        for effect, weight, curve in zip(self.effects, self.weights, self.stretch_curves):
            # Calculate individual stretch scale based on curve type
            if curve == 'constant':
                # For constant curve, always use the full end_stretch value
                effect_scale = end_stretch
            elif curve == 'linear':
                # For linear, scale with the overall animation progress
                effect_scale = stretch_scale
            elif curve == 'ease_in':
                effect_scale = end_stretch * (t * t)
            elif curve == 'ease_out':
                effect_scale = end_stretch * (t * (2.0 - t))
            elif curve == 'ease_in_out':
                effect_scale = end_stretch * (t * t * (3.0 - 2.0 * t))
            else:
                effect_scale = stretch_scale
            
            # Generate factors for this effect and scale them
            factors = effect.generate_factors(width, frame, total_frames)
            combined_factors += factors * effect_scale * weight
            
        return combined_factors
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
        # For composite effects, we'll use the first effect's warp method
        # In practice, you might want to blend warping methods too
        return self.effects[0].warp_column(column, factor, column_index, total_width, height)


class HorizontalStretchEffect(DistortionEffect):
    """Horizontal stretching effect - stretches left/right instead of up/down."""
    
    def __init__(self, base_effect: DistortionEffect):
        """
        Wrap any existing effect to work horizontally.
        
        Args:
            base_effect: The underlying effect to apply horizontally
        """
        self.base_effect = base_effect
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        # For horizontal stretching, we need factors for each row instead of column
        # But since we'll transpose the image, we use height as the dimension
        # This will be called with the transposed dimensions
        return self.base_effect.generate_factors(width, frame, total_frames)
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
        # This will actually be warping rows when used with transposed image
        return self.base_effect.warp_column(column, factor, column_index, total_width, height)


class RotatedStretchEffect(DistortionEffect):
    """Apply stretching effect at an arbitrary angle."""
    
    def __init__(self, base_effect: DistortionEffect, angle: float):
        """
        Rotate the stretching effect by a given angle.
        
        Args:
            base_effect: The underlying effect to apply
            angle: Rotation angle in degrees (0 = vertical, 90 = horizontal)
        """
        self.base_effect = base_effect
        self.angle = angle
        self._factors_cache = None
        self._cache_frame = -1
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        # Cache factors for the same frame to avoid regenerating
        if frame != self._cache_frame:
            self._factors_cache = self.base_effect.generate_factors(width, frame, total_frames)
            self._cache_frame = frame
        return self._factors_cache
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
        # For rotated effect, we need to handle this differently
        # This method will be called from a modified warp_frame implementation
        return self.base_effect.warp_column(column, factor, column_index, total_width, height)


class FlowingMeltEffect(DistortionEffect):
    """Melting effect with wave-like horizontal drift - like water flowing down a river."""
    
    def __init__(self, max_stretch: float = 0.5, melt_bias: float = 0.15,
                 flow_amplitude: float = 0.1, flow_frequency: float = 2.0,
                 flow_speed: float = 0.3, flow_variation: float = 0.2,
                 edge_behavior: Literal['wrap', 'clamp', 'fade'] = 'wrap',
                 seed: Optional[int] = None):
        """
        Create a flowing melt effect where pixels melt downward while drifting horizontally.
        
        Args:
            max_stretch: Maximum vertical stretch factor
            melt_bias: Downward bias for melting (0 to 1, higher = more downward)
            flow_amplitude: Horizontal drift amplitude (0 to 1)
            flow_frequency: Number of horizontal waves
            flow_speed: Speed of horizontal wave movement
            flow_variation: Random variation in flow pattern (0 to 1)
            edge_behavior: How to handle pixels drifting off edges
            seed: Random seed for reproducibility
        """
        self.max_stretch = max_stretch
        self.melt_bias = np.clip(melt_bias, 0.0, 1.0)
        self.flow_amplitude = flow_amplitude
        self.flow_frequency = flow_frequency
        self.flow_speed = flow_speed
        self.flow_variation = np.clip(flow_variation, 0.0, 1.0)
        self.edge_behavior = edge_behavior
        self.rng = np.random.RandomState(seed)
        
        # Pre-generate random flow variations for each column
        self._flow_phases = None
        self._flow_amp_variations = None
        self._width_cache = 0
    
    def generate_factors(self, width: int, frame: int = 0, total_frames: int = 60) -> np.ndarray:
        # Calculate current time for wave animation
        time = frame / max(total_frames - 1, 1) if frame > 0 else 0
        
        # Generate vertical stretch factors with downward bias
        base_random = self.rng.rand(width)
        
        # Apply melt bias - mostly downward with occasional upward
        factors = np.zeros(width)
        bias_probability = 0.5 + self.melt_bias * 0.4  # 50% to 90% based on bias
        
        for i in range(width):
            if self.rng.rand() < bias_probability:
                factors[i] = base_random[i]  # Positive (downward stretch)
            else:
                factors[i] = -base_random[i] * 0.3  # Small upward movement
        
        # Cache width-dependent random variations
        if width != self._width_cache:
            self._width_cache = width
            self._flow_phases = self.rng.rand(width) * 2 * np.pi
            self._flow_amp_variations = 1.0 + (self.rng.rand(width) - 0.5) * self.flow_variation
        
        # Apply wave modulation to create horizontal oscillations in the melting strength
        for x in range(width):
            # Calculate wave at this column position and time
            wave = np.sin(2 * np.pi * self.flow_frequency * x / width + 
                         self._flow_phases[x] + time * self.flow_speed * 2 * np.pi)
            
            # Modulate the stretch factor with the wave
            # This creates columns that melt more or less based on the wave pattern
            wave_modulation = 1.0 + wave * self.flow_amplitude * self._flow_amp_variations[x]
            factors[x] *= wave_modulation
        
        return factors * self.max_stretch
    
    def warp_column(self, column: np.ndarray, factor: float, column_index: int,
                   total_width: int, height: int) -> np.ndarray:
        # Standard vertical melting - no horizontal displacement here
        output_rows = np.arange(height)
        
        # Vertical melting displacement
        if factor > 0:  # Downward stretch
            source_rows = output_rows - factor * height * (output_rows / height)
        else:  # Upward stretch
            source_rows = output_rows - factor * height * (1 - output_rows / height)
        
        # Clip to valid range and round to nearest integer
        source_rows = np.clip(np.round(source_rows), 0, height - 1).astype(int)
        
        return column[source_rows]
    
