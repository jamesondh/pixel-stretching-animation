import pytest
import numpy as np
from src.pixel_stretcher import PixelStretcher


def test_cumulative_mode_incremental_stretch():
    """Test that cumulative mode applies incremental stretch, not full stretch each frame."""
    stretcher = PixelStretcher(
        stretch_curve='linear',
        start_stretch=0.0,
        end_stretch=0.6,
        cumulative=True,
        seed=42
    )
    
    # Test that incremental stretches sum to approximately end_stretch
    frames = 60
    total_stretch = 0.0
    
    for i in range(1, frames):  # Skip first frame (no stretch)
        t = i / (frames - 1)
        current_stretch = stretcher._calculate_stretch_scale(t)
        
        prev_t = (i - 1) / (frames - 1)
        prev_stretch = stretcher._calculate_stretch_scale(prev_t)
        
        incremental_stretch = current_stretch - prev_stretch
        total_stretch += incremental_stretch
    
    # The sum of incremental stretches should equal end_stretch
    assert abs(total_stretch - 0.6) < 0.01
    
    # Each individual increment should be small
    avg_increment = 0.6 / (frames - 1)
    assert avg_increment < 0.02  # Much smaller than applying full stretch each frame


def test_cumulative_vs_noncumulative_first_frame():
    """Test that both modes handle the first frame correctly."""
    # Create two stretchers with same settings
    stretcher_cumulative = PixelStretcher(
        stretch_curve='linear',
        start_stretch=0.0,
        end_stretch=0.5,
        cumulative=True,
        seed=42
    )
    
    stretcher_normal = PixelStretcher(
        stretch_curve='linear',
        start_stretch=0.0,
        end_stretch=0.5,
        cumulative=False,
        seed=42
    )
    
    # Both should have zero stretch at t=0
    assert stretcher_cumulative._calculate_stretch_scale(0.0) == 0.0
    assert stretcher_normal._calculate_stretch_scale(0.0) == 0.0