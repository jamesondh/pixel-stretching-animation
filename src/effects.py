import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Callable
import noise


class StretchEffects:
    @staticmethod
    def smooth_noise(width: int, smoothness: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(seed)
        raw_noise = rng.rand(width) * 2 - 1
        
        # Apply Gaussian smoothing
        sigma = width * smoothness * 0.1
        smoothed = gaussian_filter1d(raw_noise, sigma, mode='wrap')
        
        # Normalize to [-1, 1]
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        return smoothed * 2 - 1
    
    @staticmethod
    def perlin_noise(width: int, scale: float = 0.1, octaves: int = 1, seed: int = 0) -> np.ndarray:
        noise_values = np.array([
            noise.pnoise1(x * scale, octaves=octaves, base=seed)
            for x in range(width)
        ])
        
        # Normalize to [-1, 1]
        noise_values = (noise_values - noise_values.min()) / (noise_values.max() - noise_values.min())
        return noise_values * 2 - 1
    
    @staticmethod
    def wave_pattern(
        width: int, 
        frequency: float = 0.1, 
        amplitude: float = 1.0,
        phase: float = 0.0
    ) -> np.ndarray:
        x = np.arange(width)
        wave = amplitude * np.sin(2 * np.pi * frequency * x + phase)
        return wave
    
    @staticmethod
    def combined_wave(
        width: int,
        frequencies: list = [0.05, 0.1, 0.2],
        amplitudes: list = [0.5, 0.3, 0.2],
        phases: Optional[list] = None
    ) -> np.ndarray:
        if phases is None:
            phases = [0] * len(frequencies)
        
        result = np.zeros(width)
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            result += StretchEffects.wave_pattern(width, freq, amp, phase)
        
        # Normalize to [-1, 1]
        max_val = max(abs(result.max()), abs(result.min()))
        if max_val > 0:
            result = result / max_val
        
        return result


class TemporalEffects:
    def __init__(self, smoothing_factor: float = 0.8):
        self.smoothing_factor = smoothing_factor
        self.history = None
    
    def apply_temporal_smoothing(self, current_values: np.ndarray) -> np.ndarray:
        if self.history is None:
            self.history = current_values.copy()
            return current_values
        
        smoothed = (self.smoothing_factor * self.history + 
                   (1 - self.smoothing_factor) * current_values)
        self.history = smoothed.copy()
        return smoothed
    
    def reset(self):
        self.history = None


class AnimationCurves:
    @staticmethod
    def ease_in_out(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def ease_in(t: float) -> float:
        return t * t
    
    @staticmethod
    def ease_out(t: float) -> float:
        return t * (2.0 - t)
    
    @staticmethod
    def elastic(t: float, amplitude: float = 1.0, period: float = 0.3) -> float:
        if t == 0 or t == 1:
            return t
        
        s = period / (2 * np.pi) * np.arcsin(1.0 / amplitude)
        return amplitude * np.power(2, -10 * t) * np.sin((t - s) * 2 * np.pi / period) + 1
    
    @staticmethod
    def bounce(t: float) -> float:
        if t < 1/2.75:
            return 7.5625 * t * t
        elif t < 2/2.75:
            t -= 1.5/2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5/2.75:
            t -= 2.25/2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625/2.75
            return 7.5625 * t * t + 0.984375


def create_animated_sequence(
    width: int,
    frames: int,
    effect_func: Callable,
    max_stretch: float = 0.5,
    loop: bool = True,
    easing: Optional[Callable] = None
) -> np.ndarray:
    sequence = np.zeros((frames, width))
    
    for frame in range(frames):
        if loop:
            t = frame / frames
        else:
            t = min(frame / (frames - 1), 1.0)
        
        if easing:
            t = easing(t)
        
        # Generate base effect with time-based variation
        base_effect = effect_func(width, phase=t * 2 * np.pi)
        sequence[frame] = base_effect * max_stretch
    
    return sequence