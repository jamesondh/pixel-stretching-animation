"""
Transformation utilities for image and video processing.

Provides reusable transformation functions that can be used by both
distortion effects and post-processors.
"""

import numpy as np
from scipy.ndimage import rotate as scipy_rotate
from typing import Tuple, Optional


def rotate_image(image: np.ndarray, angle: float, reshape: bool = True, 
                 order: int = 1) -> np.ndarray:
    """
    Rotate an image by a given angle.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees
        reshape: Whether to expand output to fit rotated image
        order: Interpolation order (0=nearest, 1=bilinear, 3=cubic)
        
    Returns:
        Rotated image
    """
    return scipy_rotate(image, angle, reshape=reshape, order=order)


def rotate_and_crop(image: np.ndarray, angle: float, order: int = 1) -> np.ndarray:
    """
    Rotate an image and crop back to original size.
    
    Args:
        image: Input image as numpy array
        angle: Rotation angle in degrees
        order: Interpolation order
        
    Returns:
        Rotated and cropped image
    """
    original_shape = image.shape[:2]
    
    # Rotate with reshape to avoid clipping
    rotated = rotate_image(image, angle, reshape=True, order=order)
    
    # Calculate crop to match original size
    h_diff = rotated.shape[0] - original_shape[0]
    w_diff = rotated.shape[1] - original_shape[1]
    h_start = h_diff // 2
    w_start = w_diff // 2
    
    return rotated[h_start:h_start+original_shape[0], 
                   w_start:w_start+original_shape[1]]


def transpose_image(image: np.ndarray) -> np.ndarray:
    """
    Transpose an image (swap width and height).
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Transposed image
    """
    if len(image.shape) == 3:
        return image.transpose(1, 0, 2)
    else:
        return image.T


def apply_transform_wrapper(image: np.ndarray, process_func, angle: float = 0.0) -> np.ndarray:
    """
    Apply a processing function with optional rotation.
    
    This function handles the rotation logic: rotate input, apply processing,
    rotate back to original orientation.
    
    Args:
        image: Input image
        process_func: Function that processes the image
        angle: Rotation angle in degrees
        
    Returns:
        Processed image
    """
    if angle == 0.0:
        # No rotation needed
        return process_func(image)
    
    elif angle == 90.0:
        # Use transpose for 90-degree rotation (more efficient)
        transposed = transpose_image(image)
        processed = process_func(transposed)
        return transpose_image(processed)
    
    elif angle == -90.0 or angle == 270.0:
        # Transpose and flip for -90 degree rotation
        transposed = transpose_image(image)
        flipped = np.flip(transposed, axis=0)
        processed = process_func(flipped)
        unflipped = np.flip(processed, axis=0)
        return transpose_image(unflipped)
    
    else:
        # General rotation
        rotated = rotate_image(image, angle, reshape=True)
        processed = process_func(rotated)
        return rotate_and_crop(processed, -angle)


def calculate_rotation_padding(shape: Tuple[int, int], angle: float) -> Tuple[int, int, int, int]:
    """
    Calculate padding needed after rotation to maintain original size.
    
    Args:
        shape: Original (height, width) of image
        angle: Rotation angle in degrees
        
    Returns:
        Tuple of (top, bottom, left, right) padding
    """
    angle_rad = np.radians(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    
    h, w = shape
    new_h = int(w * sin_a + h * cos_a)
    new_w = int(w * cos_a + h * sin_a)
    
    pad_h = (new_h - h) // 2
    pad_w = (new_w - w) // 2
    
    return (pad_h, pad_h, pad_w, pad_w)


def get_axis_angle(axis: str) -> float:
    """
    Convert axis name to rotation angle.
    
    Args:
        axis: Axis name ('vertical', 'horizontal', 'diagonal', etc.)
        
    Returns:
        Rotation angle in degrees
    """
    axis_angles = {
        'vertical': 0.0,
        'horizontal': 90.0,
        'diagonal': 45.0,
        'diagonal_reverse': -45.0,
    }
    return axis_angles.get(axis.lower(), 0.0)


def create_coordinate_grids(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create coordinate grids for image transformation.
    
    Args:
        shape: Image shape (height, width)
        
    Returns:
        Tuple of (y_grid, x_grid) coordinate arrays
    """
    height, width = shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    return y_grid, x_grid


def normalize_coordinates(coords: np.ndarray, size: int) -> np.ndarray:
    """
    Normalize coordinates to [0, 1] range.
    
    Args:
        coords: Coordinate array
        size: Size of the dimension
        
    Returns:
        Normalized coordinates
    """
    return coords / max(size - 1, 1)


def denormalize_coordinates(coords: np.ndarray, size: int) -> np.ndarray:
    """
    Denormalize coordinates from [0, 1] to pixel range.
    
    Args:
        coords: Normalized coordinate array
        size: Size of the dimension
        
    Returns:
        Pixel coordinates
    """
    return coords * (size - 1)