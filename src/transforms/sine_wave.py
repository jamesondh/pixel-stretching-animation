"""
Shared sine wave transformation logic.
"""

import numpy as np
from typing import Union, Tuple, Optional, Literal


class SineWaveTransform:
    """
    Sine wave transformation that can be used for both column-based
    and frame-based distortions.
    """
    
    def __init__(
        self,
        frequency: float = 3.0,
        amplitude: float = 0.05,
        phase: float = 0.0,
        speed: float = 0.5,
        axis: Union[str, float] = 'vertical'
    ):
        """
        Initialize sine wave transform.
        
        Args:
            frequency: Number of wave cycles
            amplitude: Wave amplitude (0-1, relative to dimension)
            phase: Initial phase offset in radians
            speed: Animation speed for moving waves
            axis: Direction of wave ('vertical', 'horizontal', 'diagonal', or angle in degrees)
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.speed = speed
        self.axis = axis
        
        # Convert axis to angle in radians
        self.angle_rad = self._parse_axis(axis)
    
    def _parse_axis(self, axis: Union[str, float]) -> float:
        """Convert axis specification to angle in radians."""
        if isinstance(axis, (int, float)):
            return np.radians(axis)
        elif axis == 'vertical':
            return 0.0
        elif axis == 'horizontal':
            return np.pi / 2
        elif axis == 'diagonal':
            return np.pi / 4
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    def calculate_displacement(
        self,
        x: np.ndarray,
        y: np.ndarray,
        time: float = 0.0,
        dimension_size: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate displacement for given coordinates.
        
        Args:
            x: X coordinates (can be 1D or 2D array)
            y: Y coordinates (can be 1D or 2D array)
            time: Time parameter for animation (0-1)
            dimension_size: Size of the dimension for amplitude scaling
            
        Returns:
            Tuple of (dx, dy) displacement arrays
        """
        # Calculate position along the wave axis
        cos_angle = np.cos(self.angle_rad)
        sin_angle = np.sin(self.angle_rad)
        
        # Project coordinates onto wave axis
        position = x * cos_angle + y * sin_angle
        
        # Calculate sine wave
        wave_phase = self.phase + time * self.speed * 2 * np.pi
        wave = np.sin(position * self.frequency * 2 * np.pi / dimension_size + wave_phase)
        
        # Calculate displacement perpendicular to wave axis
        displacement = wave * self.amplitude * dimension_size
        
        # Convert to x,y displacement
        dx = displacement * (-sin_angle)
        dy = displacement * cos_angle
        
        return dx, dy
    
    def calculate_column_displacement(
        self,
        column_index: int,
        height: int,
        width: int,
        time: float = 0.0
    ) -> np.ndarray:
        """
        Calculate vertical displacement for a single column (for DistortionEffect).
        
        Args:
            column_index: Index of the column
            height: Image height
            width: Image width
            time: Time parameter for animation
            
        Returns:
            Array of vertical displacements for each pixel in the column
        """
        # Create coordinate arrays for the column
        x = np.full(height, column_index)
        y = np.arange(height)
        
        # Calculate displacement
        dx, dy = self.calculate_displacement(x, y, time, width)
        
        # For column-based effects, we typically only use vertical displacement
        if self.axis == 'horizontal' or (isinstance(self.axis, (int, float)) and abs(self.angle_rad - np.pi/2) < 0.1):
            return dx  # Use horizontal displacement for horizontal axis
        else:
            return dy  # Use vertical displacement for other axes
    
    def calculate_frame_displacement(
        self,
        width: int,
        height: int,
        time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate displacement field for entire frame (for PostProcessor).
        
        Args:
            width: Frame width
            height: Frame height
            time: Time parameter for animation
            
        Returns:
            Tuple of (dx, dy) displacement fields
        """
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Use width as the reference dimension for consistency
        return self.calculate_displacement(x, y, time, width)
    
    def get_amplitude_at_time(
        self,
        time: float,
        curve: str = 'constant',
        start_amplitude: Optional[float] = None,
        end_amplitude: Optional[float] = None
    ) -> float:
        """
        Calculate amplitude at given time based on curve type.
        
        Args:
            time: Normalized time (0-1)
            curve: Amplitude curve type
            start_amplitude: Starting amplitude (defaults to self.amplitude)
            end_amplitude: Ending amplitude (defaults to self.amplitude)
            
        Returns:
            Amplitude value at given time
        """
        if start_amplitude is None:
            start_amplitude = self.amplitude
        if end_amplitude is None:
            end_amplitude = self.amplitude
            
        if curve == 'constant':
            return self.amplitude
        elif curve == 'linear':
            return start_amplitude + (end_amplitude - start_amplitude) * time
        elif curve == 'ease_in':
            t = time * time  # Quadratic ease in
            return start_amplitude + (end_amplitude - start_amplitude) * t
        elif curve == 'ease_out':
            t = 1 - (1 - time) ** 2  # Quadratic ease out
            return start_amplitude + (end_amplitude - start_amplitude) * t
        elif curve == 'ease_in_out':
            if time < 0.5:
                t = 2 * time * time
            else:
                t = 1 - 2 * (1 - time) ** 2
            return start_amplitude + (end_amplitude - start_amplitude) * t
        else:
            raise ValueError(f"Unknown amplitude curve: {curve}")