import pytest
import numpy as np
from src.pixel_stretcher import PixelStretcher


class TestStretchCurves:
    """Test the stretch curve functionality."""
    
    def test_constant_curve(self):
        """Test constant stretch curve maintains same value."""
        stretcher = PixelStretcher(
            max_stretch=0.5,
            stretch_curve='constant',
            seed=42
        )
        
        # Test at different progress points
        assert stretcher._calculate_stretch_scale(0.0) == 0.5
        assert stretcher._calculate_stretch_scale(0.5) == 0.5
        assert stretcher._calculate_stretch_scale(1.0) == 0.5
    
    def test_linear_curve(self):
        """Test linear stretch curve interpolates correctly."""
        stretcher = PixelStretcher(
            max_stretch=0.8,
            stretch_curve='linear',
            seed=42
        )
        
        assert stretcher._calculate_stretch_scale(0.0) == 0.0
        assert stretcher._calculate_stretch_scale(0.5) == 0.4
        assert stretcher._calculate_stretch_scale(1.0) == 0.8
    
    def test_ease_in_curve(self):
        """Test ease_in curve starts slow."""
        stretcher = PixelStretcher(
            max_stretch=1.0,
            stretch_curve='ease_in',
            seed=42
        )
        
        # At 25% progress, should be less than 25% of max
        assert stretcher._calculate_stretch_scale(0.25) < 0.25
        # At 50% progress, should be 25% of max (quadratic)
        assert stretcher._calculate_stretch_scale(0.5) == 0.25
        assert stretcher._calculate_stretch_scale(1.0) == 1.0
    
    def test_ease_out_curve(self):
        """Test ease_out curve starts fast."""
        stretcher = PixelStretcher(
            max_stretch=1.0,
            stretch_curve='ease_out',
            seed=42
        )
        
        # At 25% progress, should be more than 25% of max
        assert stretcher._calculate_stretch_scale(0.25) > 0.25
        assert stretcher._calculate_stretch_scale(1.0) == 1.0
    
    def test_ease_in_out_curve(self):
        """Test ease_in_out curve is smooth at both ends."""
        stretcher = PixelStretcher(
            max_stretch=1.0,
            stretch_curve='ease_in_out',
            seed=42
        )
        
        # Should be exactly 0.5 at midpoint
        assert stretcher._calculate_stretch_scale(0.5) == 0.5
        # Should start and end at correct values
        assert stretcher._calculate_stretch_scale(0.0) == 0.0
        assert stretcher._calculate_stretch_scale(1.0) == 1.0
    
    def test_custom_start_end_stretch(self):
        """Test custom start and end stretch values."""
        stretcher = PixelStretcher(
            max_stretch=0.8,  # This should be overridden
            stretch_curve='linear',
            start_stretch=0.2,
            end_stretch=0.6,
            seed=42
        )
        
        assert stretcher._calculate_stretch_scale(0.0) == 0.2
        assert stretcher._calculate_stretch_scale(0.5) == 0.4
        assert stretcher._calculate_stretch_scale(1.0) == 0.6
    
    def test_constant_with_custom_end(self):
        """Test constant curve with custom end_stretch."""
        stretcher = PixelStretcher(
            max_stretch=0.5,
            stretch_curve='constant',
            start_stretch=0.1,  # Should be ignored for constant
            end_stretch=0.3,
            seed=42
        )
        
        # Constant always uses end_stretch
        assert stretcher._calculate_stretch_scale(0.0) == 0.3
        assert stretcher._calculate_stretch_scale(0.5) == 0.3
        assert stretcher._calculate_stretch_scale(1.0) == 0.3
    
    def test_reverse_animation(self):
        """Test animation that goes from distorted to normal."""
        stretcher = PixelStretcher(
            stretch_curve='linear',
            start_stretch=0.8,
            end_stretch=0.0,
            seed=42
        )
        
        assert stretcher._calculate_stretch_scale(0.0) == 0.8
        assert stretcher._calculate_stretch_scale(0.5) == 0.4
        assert stretcher._calculate_stretch_scale(1.0) == 0.0
    
    def test_custom_curve_function(self):
        """Test using a custom curve function."""
        # Custom sine wave curve
        def sine_curve(t):
            return np.sin(t * np.pi / 2)
        
        stretcher = PixelStretcher(
            max_stretch=1.0,
            stretch_curve='custom',
            custom_curve_func=sine_curve,
            seed=42
        )
        
        # Should follow sine curve
        assert abs(stretcher._calculate_stretch_scale(0.5) - np.sin(np.pi/4)) < 0.01
        assert stretcher._calculate_stretch_scale(1.0) == 1.0
    
    def test_animation_with_constant_stretch(self):
        """Test that constant stretch creates consistent frames."""
        stretcher = PixelStretcher(
            max_stretch=0.5,
            stretch_curve='constant',
            seed=42
        )
        
        # Create a simple test image
        test_image = np.ones((10, 10, 3), dtype=np.uint8) * 128
        
        # Warp at different scales - should all be the same for constant
        frame1 = stretcher.warp_frame(test_image, stretcher._calculate_stretch_scale(0.0))
        frame2 = stretcher.warp_frame(test_image, stretcher._calculate_stretch_scale(0.5))
        frame3 = stretcher.warp_frame(test_image, stretcher._calculate_stretch_scale(1.0))
        
        # Reset stretcher state between warps to ensure consistent factors
        stretcher.reset()
        frame1_again = stretcher.warp_frame(test_image, stretcher._calculate_stretch_scale(0.0))
        
        # Frames should be identical (allowing for randomness in factors)
        assert frame1.shape == frame2.shape == frame3.shape
        assert np.array_equal(frame1, frame1_again)