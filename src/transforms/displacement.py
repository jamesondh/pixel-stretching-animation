"""
Shared displacement calculation utilities.
"""

import numpy as np
from typing import Tuple, Literal
from scipy import ndimage


class DisplacementCalculator:
    """
    Utilities for applying displacement fields to images.
    """
    
    @staticmethod
    def create_displacement_grid(
        width: int,
        height: int,
        dx: np.ndarray,
        dy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate grids with displacement applied.
        
        Args:
            width: Image width
            height: Image height
            dx: X displacement field
            dy: Y displacement field
            
        Returns:
            Tuple of (new_x, new_y) coordinate grids
        """
        # Create base coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply displacement
        new_x = x + dx
        new_y = y + dy
        
        return new_x, new_y
    
    @staticmethod
    def apply_displacement(
        image: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        mode: Literal['translate', 'scale', 'both'] = 'translate',
        order: int = 1,
        cval: float = 0.0
    ) -> np.ndarray:
        """
        Apply displacement field to an image.
        
        Args:
            image: Input image
            dx: X displacement field
            dy: Y displacement field
            mode: Displacement mode
            order: Interpolation order (0=nearest, 1=bilinear, 3=cubic)
            cval: Value for pixels outside boundaries
            
        Returns:
            Displaced image
        """
        height, width = image.shape[:2]
        
        if mode == 'translate':
            # Simple translation displacement
            new_x, new_y = DisplacementCalculator.create_displacement_grid(
                width, height, dx, dy
            )
        elif mode == 'scale':
            # Scale-based displacement (warping)
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            # Convert displacement to scaling factors
            scale_x = 1 + dx / width
            scale_y = 1 + dy / height
            new_x = x * scale_x
            new_y = y * scale_y
        elif mode == 'both':
            # Combination of translation and scaling
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            # Apply both translation and scaling
            scale_factor = 0.5  # Balance between modes
            new_x = x + dx * scale_factor + x * (dx / width) * (1 - scale_factor)
            new_y = y + dy * scale_factor + y * (dy / height) * (1 - scale_factor)
        else:
            raise ValueError(f"Unknown displacement mode: {mode}")
        
        # Apply displacement using map_coordinates
        if len(image.shape) == 3:
            # Color image: process each channel
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c],
                    [new_y, new_x],
                    order=order,
                    mode='constant',
                    cval=cval
                )
        else:
            # Grayscale image
            result = ndimage.map_coordinates(
                image,
                [new_y, new_x],
                order=order,
                mode='constant',
                cval=cval
            )
        
        return result
    
    @staticmethod
    def apply_column_displacement(
        column: np.ndarray,
        displacement: np.ndarray,
        interpolation: str = 'nearest'
    ) -> np.ndarray:
        """
        Apply vertical displacement to a single column.
        
        Args:
            column: Input column (height, channels)
            displacement: Vertical displacement for each pixel
            interpolation: Interpolation method ('nearest' or 'bilinear')
            
        Returns:
            Displaced column
        """
        height = len(column)
        order = 0 if interpolation == 'nearest' else 1
        
        # Create source coordinates
        source_coords = np.arange(height) - displacement
        
        # Ensure we have the right shape for multi-channel columns
        if len(column.shape) == 2:
            # Multi-channel column
            result = np.zeros_like(column)
            for c in range(column.shape[1]):
                result[:, c] = ndimage.map_coordinates(
                    column[:, c],
                    [source_coords],
                    order=order,
                    mode='constant',
                    cval=0
                )
        else:
            # Single channel column
            result = ndimage.map_coordinates(
                column,
                [source_coords],
                order=order,
                mode='constant',
                cval=0
            )
        
        return result
    
    @staticmethod
    def calculate_stretch_factors(
        displacement: np.ndarray,
        dimension_size: int
    ) -> np.ndarray:
        """
        Convert displacement values to stretch factors.
        
        Args:
            displacement: Displacement values
            dimension_size: Size of the dimension (height or width)
            
        Returns:
            Stretch factors (0-1 range)
        """
        # Normalize displacement to 0-1 range based on dimension size
        return np.abs(displacement) / dimension_size