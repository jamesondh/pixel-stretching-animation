"""
Edge behavior utilities for handling boundary conditions in image processing.

Provides consistent edge handling across effects and post-processors.
"""

import numpy as np
from typing import Union, Literal


def apply_edge_behavior(coords: Union[int, np.ndarray], size: int, 
                       behavior: Literal['wrap', 'clamp', 'mirror', 'fade'] = 'clamp') -> Union[int, np.ndarray]:
    """
    Apply edge behavior to coordinates.
    
    Args:
        coords: Coordinate(s) to process
        size: Size of the dimension
        behavior: Edge behavior mode
        
    Returns:
        Processed coordinates
    """
    if behavior == 'wrap':
        return coords % size
    
    elif behavior == 'clamp':
        return np.clip(coords, 0, size - 1)
    
    elif behavior == 'mirror':
        # Mirror at boundaries
        if isinstance(coords, np.ndarray):
            result = coords.copy()
            # Handle negative coordinates
            negative_mask = result < 0
            result[negative_mask] = -result[negative_mask]
            
            # Handle coordinates beyond size
            while True:
                overflow_mask = result >= size
                if not np.any(overflow_mask):
                    break
                result[overflow_mask] = 2 * size - result[overflow_mask] - 2
                
                # Ensure we don't get stuck in infinite loop
                result = np.clip(result, 0, 2 * size)
                
            return np.clip(result, 0, size - 1)
        else:
            # Scalar case
            result = coords
            if result < 0:
                result = -result
            while result >= size:
                result = 2 * size - result - 2
            return np.clip(result, 0, size - 1)
    
    else:  # fade - just clamp for now, caller handles fading
        return np.clip(coords, 0, size - 1)


def create_fade_mask(coords: np.ndarray, size: int, fade_width: int = 10) -> np.ndarray:
    """
    Create a fade mask for coordinates near boundaries.
    
    Args:
        coords: Original coordinates (before clamping)
        size: Size of the dimension
        fade_width: Width of fade region in pixels
        
    Returns:
        Fade mask (0.0 to 1.0)
    """
    mask = np.ones_like(coords, dtype=float)
    
    # Fade near start boundary
    fade_start = coords < fade_width
    mask[fade_start] = coords[fade_start] / fade_width
    
    # Fade near end boundary
    fade_end = coords > size - fade_width
    mask[fade_end] = (size - coords[fade_end]) / fade_width
    
    # Completely fade out of bounds
    out_of_bounds = (coords < 0) | (coords >= size)
    mask[out_of_bounds] = 0.0
    
    return np.clip(mask, 0.0, 1.0)


def apply_2d_edge_behavior(y_coords: np.ndarray, x_coords: np.ndarray, 
                          shape: tuple, behavior: str = 'clamp') -> tuple:
    """
    Apply edge behavior to 2D coordinates.
    
    Args:
        y_coords: Y coordinates
        x_coords: X coordinates  
        shape: Image shape (height, width)
        behavior: Edge behavior mode
        
    Returns:
        Tuple of (y_coords, x_coords) after applying edge behavior
    """
    height, width = shape[:2]
    
    y_processed = apply_edge_behavior(y_coords, height, behavior)
    x_processed = apply_edge_behavior(x_coords, width, behavior)
    
    return y_processed, x_processed


def sample_with_edge_behavior(image: np.ndarray, y_coords: np.ndarray, 
                             x_coords: np.ndarray, behavior: str = 'clamp',
                             fade_width: int = 10) -> np.ndarray:
    """
    Sample from image with specified edge behavior.
    
    Args:
        image: Source image
        y_coords: Y coordinates to sample
        x_coords: X coordinates to sample
        behavior: Edge behavior mode
        fade_width: Width of fade region for 'fade' behavior
        
    Returns:
        Sampled values
    """
    height, width = image.shape[:2]
    
    if behavior == 'fade':
        # Create fade masks
        y_mask = create_fade_mask(y_coords, height, fade_width)
        x_mask = create_fade_mask(x_coords, width, fade_width)
        fade_mask = y_mask * x_mask
        
        # Clamp coordinates for sampling
        y_clamped = np.clip(y_coords, 0, height - 1).astype(int)
        x_clamped = np.clip(x_coords, 0, width - 1).astype(int)
        
        # Sample and apply fade
        if len(image.shape) == 3:
            sampled = image[y_clamped, x_clamped]
            # Apply fade to each channel
            for c in range(image.shape[2]):
                sampled[:, :, c] = sampled[:, :, c] * fade_mask
        else:
            sampled = image[y_clamped, x_clamped] * fade_mask
            
        return sampled
    
    else:
        # Apply edge behavior to coordinates
        y_processed, x_processed = apply_2d_edge_behavior(
            y_coords, x_coords, image.shape, behavior
        )
        
        # Convert to integers for indexing
        y_int = np.round(y_processed).astype(int)
        x_int = np.round(x_processed).astype(int)
        
        return image[y_int, x_int]


def warp_with_edge_behavior(image: np.ndarray, displacement_y: np.ndarray,
                           displacement_x: np.ndarray, behavior: str = 'clamp',
                           fade_width: int = 10) -> np.ndarray:
    """
    Warp image with displacement fields and edge behavior.
    
    Args:
        image: Source image
        displacement_y: Y displacement field
        displacement_x: X displacement field
        behavior: Edge behavior mode
        fade_width: Width of fade region for 'fade' behavior
        
    Returns:
        Warped image
    """
    height, width = image.shape[:2]
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    # Apply displacement
    source_y = y_grid + displacement_y
    source_x = x_grid + displacement_x
    
    # Sample with edge behavior
    return sample_with_edge_behavior(image, source_y, source_x, behavior, fade_width)