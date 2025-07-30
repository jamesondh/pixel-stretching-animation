"""
Backward-compatible wrapper for the refactored PixelStretcher.
This maintains the original API while using the new modular structure internally.
"""

from .pixel_stretcher_refactored import PixelStretcher as RefactoredPixelStretcher


class PixelStretcher(RefactoredPixelStretcher):
    def __init__(
        self,
        max_stretch: float = 0.5,
        pivot: str = 'center',
        interpolation: str = 'nearest',
        temporal_smoothing: float = 0.0,
        seed: int = None,
        upscale: int = 1,
        cumulative: bool = False,
        wave_amplitude: float = 0.1,
        wave_frequency: float = 2.0,
        wave_phase_shift: float = 0.0,
        wave_phase_speed: float = 0.2,
        use_wave_distortion: bool = False,
        stretch_bias: float = 0.0
    ):
        # Use the legacy params factory method
        instance = RefactoredPixelStretcher.from_legacy_params(
            max_stretch=max_stretch,
            pivot=pivot,
            interpolation=interpolation,
            temporal_smoothing=temporal_smoothing,
            seed=seed,
            upscale=upscale,
            cumulative=cumulative,
            wave_amplitude=wave_amplitude,
            wave_frequency=wave_frequency,
            wave_phase_shift=wave_phase_shift,
            wave_phase_speed=wave_phase_speed,
            use_wave_distortion=use_wave_distortion,
            stretch_bias=stretch_bias
        )
        
        # Copy attributes from the created instance
        self.effect = instance.effect
        self.interpolation = instance.interpolation
        self.temporal_smoothing = instance.temporal_smoothing
        self.upscale = instance.upscale
        self.cumulative = instance.cumulative
        self._previous_factors = instance._previous_factors
        
        # Store legacy parameters for compatibility
        self.max_stretch = max_stretch
        self.pivot = pivot
        self.wave_amplitude = wave_amplitude
        self.wave_frequency = wave_frequency
        self.wave_phase_shift = wave_phase_shift
        self.wave_phase_speed = wave_phase_speed
        self.use_wave_distortion = use_wave_distortion
        self.stretch_bias = stretch_bias
        self.rng = getattr(self.effect, 'rng', None)
    
    # Legacy methods for backward compatibility
    def _get_pivot_point(self, height: int) -> float:
        if hasattr(self.effect, '_get_pivot_point'):
            return self.effect._get_pivot_point(height)
        return height / 2
    
    def _generate_stretch_factors(self, width: int, stretch_scale: float = 1.0) -> np.ndarray:
        # This method is called by legacy code
        # Map to new effect's generate_factors method
        factors = self.effect.generate_factors(width, frame=1, total_frames=1)
        return factors * stretch_scale