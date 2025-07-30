"""
Unit tests for the main PixelStretcher class.
"""

import unittest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pixel_stretcher import PixelStretcher


class TestPixelStretcher(unittest.TestCase):
    
    def setUp(self):
        # Create a simple test image
        self.test_image = np.zeros((32, 32, 3), dtype=np.uint8)
        # Add some patterns
        self.test_image[::4, :] = [255, 0, 0]  # Red stripes
        self.test_image[:, ::4] = [0, 0, 255]  # Blue stripes
        
        # Save test image
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = Path(self.temp_dir) / "test_input.png"
        Image.fromarray(self.test_image).save(self.input_path)
    
    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_initialization(self):
        stretcher = PixelStretcher(max_stretch=0.5)
        
        self.assertEqual(stretcher.max_stretch, 0.5)
        self.assertEqual(stretcher.pivot, 'center')
        self.assertEqual(stretcher.interpolation, 'nearest')
        self.assertFalse(stretcher.cumulative)
    
    def test_parameter_validation(self):
        # Test valid parameters
        stretcher = PixelStretcher(
            max_stretch=0.7,
            pivot='top',
            interpolation='bilinear',
            upscale=4
        )
        
        self.assertEqual(stretcher.max_stretch, 0.7)
        self.assertEqual(stretcher.pivot, 'top')
        self.assertEqual(stretcher.interpolation, 'bilinear')
        self.assertEqual(stretcher.upscale, 4)
    
    def test_warp_frame(self):
        stretcher = PixelStretcher(max_stretch=0.5, seed=42)
        
        # Test single frame warping
        warped = stretcher.warp_frame(self.test_image)
        
        # Check shape is preserved (or upscaled)
        self.assertEqual(warped.shape, self.test_image.shape)
        
        # Check data type is preserved
        self.assertEqual(warped.dtype, self.test_image.dtype)
    
    def test_upscaling(self):
        stretcher = PixelStretcher(max_stretch=0.3, upscale=2)
        
        warped = stretcher.warp_frame(self.test_image)
        
        # Check upscaling worked
        self.assertEqual(warped.shape[0], self.test_image.shape[0] * 2)
        self.assertEqual(warped.shape[1], self.test_image.shape[1] * 2)
    
    def test_temporal_smoothing(self):
        stretcher = PixelStretcher(
            max_stretch=0.5,
            temporal_smoothing=0.8,
            seed=42
        )
        
        # Generate two frames
        frame1 = stretcher.warp_frame(self.test_image)
        frame2 = stretcher.warp_frame(self.test_image)
        
        # With high temporal smoothing, frames should be similar
        diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
        
        # Reset and test without smoothing
        stretcher.reset()
        stretcher.temporal_smoothing = 0.0
        
        frame3 = stretcher.warp_frame(self.test_image)
        frame4 = stretcher.warp_frame(self.test_image)
        
        diff_no_smooth = np.mean(np.abs(frame3.astype(float) - frame4.astype(float)))
        
        # Smoothed frames should be more similar
        self.assertLess(diff, diff_no_smooth)
    
    def test_create_animation(self):
        output_path = Path(self.temp_dir) / "test_output.mp4"
        
        stretcher = PixelStretcher(max_stretch=0.5)
        stretcher.create_animation(
            input_path=self.input_path,
            output_path=output_path,
            frames=10,
            fps=10
        )
        
        # Check that output file was created
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
    
    def test_wave_distortion(self):
        stretcher = PixelStretcher(
            max_stretch=0.5,
            use_wave_distortion=True,
            wave_amplitude=0.1,
            wave_frequency=2.0
        )
        
        warped = stretcher.warp_frame(self.test_image)
        
        # Check basic properties
        self.assertEqual(warped.shape, self.test_image.shape)
        self.assertEqual(warped.dtype, self.test_image.dtype)
    
    def test_bias_stretching(self):
        # Test positive bias
        stretcher_pos = PixelStretcher(
            max_stretch=0.5,
            stretch_bias=0.8,
            seed=42
        )
        
        # Test negative bias
        stretcher_neg = PixelStretcher(
            max_stretch=0.5,
            stretch_bias=-0.8,
            seed=42
        )
        
        # Both should produce valid output
        warped_pos = stretcher_pos.warp_frame(self.test_image)
        warped_neg = stretcher_neg.warp_frame(self.test_image)
        
        self.assertEqual(warped_pos.shape, self.test_image.shape)
        self.assertEqual(warped_neg.shape, self.test_image.shape)
    
    def test_cumulative_mode(self):
        output_path = Path(self.temp_dir) / "test_cumulative.mp4"
        
        stretcher = PixelStretcher(
            max_stretch=0.3,
            cumulative=True,
            seed=42
        )
        
        stretcher.create_animation(
            input_path=self.input_path,
            output_path=output_path,
            frames=5,
            fps=10
        )
        
        self.assertTrue(output_path.exists())
    
    def test_reproducibility_with_seed(self):
        # Create two stretchers with same seed
        stretcher1 = PixelStretcher(max_stretch=0.5, seed=123)
        stretcher2 = PixelStretcher(max_stretch=0.5, seed=123)
        
        # Generate frames
        frame1 = stretcher1.warp_frame(self.test_image)
        frame2 = stretcher2.warp_frame(self.test_image)
        
        # Should be identical
        np.testing.assert_array_equal(frame1, frame2)
    
    def test_different_interpolation_methods(self):
        # Test nearest neighbor
        stretcher_nearest = PixelStretcher(
            max_stretch=0.5,
            interpolation='nearest',
            seed=42
        )
        
        # Test bilinear
        stretcher_bilinear = PixelStretcher(
            max_stretch=0.5,
            interpolation='bilinear',
            seed=42
        )
        
        warped_nearest = stretcher_nearest.warp_frame(self.test_image)
        warped_bilinear = stretcher_bilinear.warp_frame(self.test_image)
        
        # Both should produce valid output
        self.assertEqual(warped_nearest.shape, self.test_image.shape)
        self.assertEqual(warped_bilinear.shape, self.test_image.shape)
        
        # Results should be different
        self.assertFalse(np.array_equal(warped_nearest, warped_bilinear))


if __name__ == '__main__':
    unittest.main()