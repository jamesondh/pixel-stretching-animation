"""
Shared edge handling utilities for transformations.
"""

import numpy as np
from typing import Literal, Tuple, Optional


class EdgeHandler:
    """
    Utilities for handling edge behavior in transformations.
    """
    
    @staticmethod
    def apply_edge_behavior(
        coords: np.ndarray,
        dimension_size: int,
        behavior: Literal['wrap', 'clamp', 'fade', 'mirror'] = 'wrap',
        fade_width: float = 0.1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply edge behavior to coordinates.
        
        Args:
            coords: Coordinate array
            dimension_size: Size of the dimension
            behavior: Edge handling behavior
            fade_width: Width of fade region (as fraction of dimension)
            
        Returns:
            Tuple of (modified_coords, alpha_mask)
            alpha_mask is only returned for 'fade' behavior
        """
        alpha_mask = None
        
        if behavior == 'wrap':
            # Wrap around edges
            coords = coords % dimension_size
        elif behavior == 'clamp':
            # Clamp to edges
            coords = np.clip(coords, 0, dimension_size - 1)
        elif behavior == 'fade':
            # Create fade mask for out-of-bounds regions
            fade_pixels = int(dimension_size * fade_width)
            alpha_mask = np.ones_like(coords, dtype=np.float32)
            
            # Fade near edges
            for i in range(fade_pixels):
                fade_factor = i / fade_pixels
                mask_low = coords == i
                mask_high = coords == (dimension_size - 1 - i)
                alpha_mask[mask_low] *= fade_factor
                alpha_mask[mask_high] *= fade_factor
            
            # Fully transparent outside bounds
            alpha_mask[coords < 0] = 0
            alpha_mask[coords >= dimension_size] = 0
            
            # Clamp coordinates after creating mask
            coords = np.clip(coords, 0, dimension_size - 1)
        elif behavior == 'mirror':
            # Mirror at edges
            coords = EdgeHandler._mirror_coords(coords, dimension_size)
        else:
            raise ValueError(f"Unknown edge behavior: {behavior}")
        
        return coords, alpha_mask
    
    @staticmethod
    def _mirror_coords(coords: np.ndarray, dimension_size: int) -> np.ndarray:
        """
        Mirror coordinates at edges.
        
        Args:
            coords: Input coordinates
            dimension_size: Size of the dimension
            
        Returns:
            Mirrored coordinates
        """
        result = coords.copy()
        
        # Handle coordinates below 0
        mask = result < 0
        result[mask] = -result[mask]
        
        # Handle coordinates above dimension_size
        mask = result >= dimension_size
        result[mask] = 2 * dimension_size - 2 - result[mask]
        
        # Handle multiple bounces
        period = 2 * (dimension_size - 1)
        result = result % period
        mask = result >= dimension_size
        result[mask] = period - result[mask]
        
        return result
    
    @staticmethod
    def apply_2d_edge_behavior(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        width: int,
        height: int,
        behavior: Literal['wrap', 'clamp', 'fade', 'mirror'] = 'wrap',
        fade_width: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply edge behavior to 2D coordinates.
        
        Args:
            x_coords: X coordinate array
            y_coords: Y coordinate array
            width: Image width
            height: Image height
            behavior: Edge handling behavior
            fade_width: Width of fade region
            
        Returns:
            Tuple of (new_x, new_y, alpha_mask)
        """
        new_x, alpha_x = EdgeHandler.apply_edge_behavior(
            x_coords, width, behavior, fade_width
        )
        new_y, alpha_y = EdgeHandler.apply_edge_behavior(
            y_coords, height, behavior, fade_width
        )
        
        # Combine alpha masks if using fade
        if behavior == 'fade' and alpha_x is not None and alpha_y is not None:
            alpha_mask = alpha_x * alpha_y
        else:
            alpha_mask = None
        
        return new_x, new_y, alpha_mask
    
    @staticmethod
    def create_edge_mask(
        width: int,
        height: int,
        edge_width: int = 10,
        mode: Literal['linear', 'quadratic', 'smooth'] = 'linear'
    ) -> np.ndarray:
        """
        Create an edge mask for smooth transitions.
        
        Args:
            width: Image width
            height: Image height
            edge_width: Width of edge region in pixels
            mode: Fade mode
            
        Returns:
            2D mask array
        """
        mask = np.ones((height, width), dtype=np.float32)
        
        # Create distance arrays from edges
        x_dist = np.minimum(np.arange(width), np.arange(width)[::-1])
        y_dist = np.minimum(np.arange(height), np.arange(height)[::-1])
        
        # Create 2D distance field
        x_mask = np.minimum(x_dist / edge_width, 1.0)
        y_mask = np.minimum(y_dist / edge_width, 1.0)
        
        # Combine masks
        for i in range(height):
            for j in range(width):
                dist = min(x_mask[j], y_mask[i])
                
                if mode == 'linear':
                    mask[i, j] = dist
                elif mode == 'quadratic':
                    mask[i, j] = dist * dist
                elif mode == 'smooth':
                    # Smooth S-curve
                    mask[i, j] = 3 * dist * dist - 2 * dist * dist * dist
        
        return mask