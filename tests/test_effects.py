"""
Unit tests for distortion effects.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distortion_effects import (
    PivotStretchEffect, WaveDistortionEffect, BiasedStretchEffect, CompositeEffect
)


class TestPivotStretchEffect(unittest.TestCase):
    
    def setUp(self):
        self.effect = PivotStretchEffect(max_stretch=0.5, pivot='center', seed=42)
    
    def test_generate_factors(self):
        factors = self.effect.generate_factors(width=100, frame=30, total_frames=60)
        
        # Check shape
        self.assertEqual(len(factors), 100)
        
        # Check range
        self.assertTrue(np.all(np.abs(factors) <= 0.5))
        
        # Check reproducibility with seed
        factors2 = PivotStretchEffect(max_stretch=0.5, seed=42).generate_factors(100, 30, 60)
        np.testing.assert_array_equal(factors, factors2)
    
    def test_pivot_points(self):
        height = 100
        
        # Test center pivot
        center_effect = PivotStretchEffect(pivot='center')
        self.assertEqual(center_effect._get_pivot_point(height), 50)
        
        # Test top pivot
        top_effect = PivotStretchEffect(pivot='top')
        self.assertEqual(top_effect._get_pivot_point(height), 0)
        
        # Test bottom pivot
        bottom_effect = PivotStretchEffect(pivot='bottom')
        self.assertEqual(bottom_effect._get_pivot_point(height), 100)
    
    def test_warp_column(self):
        column = np.arange(10)
        warped = self.effect.warp_column(column, factor=0.2, column_index=0, 
                                       total_width=10, height=10)
        
        # Check shape preserved
        self.assertEqual(len(warped), len(column))
        
        # Check values are within original range
        self.assertTrue(np.all(warped >= 0))
        self.assertTrue(np.all(warped < 10))


class TestWaveDistortionEffect(unittest.TestCase):
    
    def setUp(self):
        self.effect = WaveDistortionEffect(
            max_stretch=0.5,
            wave_amplitude=0.1,
            wave_frequency=2.0,
            seed=42
        )
    
    def test_generate_factors(self):
        factors = self.effect.generate_factors(width=100)
        
        # Check shape
        self.assertEqual(len(factors), 100)
        
        # Check range
        self.assertTrue(np.all(np.abs(factors) <= 0.5))
    
    def test_wave_pattern(self):
        # Create a column of pixels
        height = 100
        column = np.ones((height, 3), dtype=np.uint8) * 128
        
        # Apply wave distortion
        warped = self.effect.warp_column(column, factor=1.0, column_index=50,
                                       total_width=100, height=height)
        
        # Check that warping creates a wave pattern
        self.assertEqual(warped.shape, column.shape)


class TestBiasedStretchEffect(unittest.TestCase):
    
    def test_positive_bias(self):
        # Positive bias should create mostly positive (downward) factors
        effect = BiasedStretchEffect(max_stretch=0.5, stretch_bias=0.8, seed=42)
        factors = effect.generate_factors(width=100)
        
        # Most factors should be positive
        positive_count = np.sum(factors > 0)
        self.assertGreater(positive_count, 70)  # At least 70% positive
    
    def test_negative_bias(self):
        # Negative bias should create mostly negative (upward) factors
        effect = BiasedStretchEffect(max_stretch=0.5, stretch_bias=-0.8, seed=42)
        factors = effect.generate_factors(width=100)
        
        # Most factors should be negative
        negative_count = np.sum(factors < 0)
        self.assertGreater(negative_count, 70)  # At least 70% negative
    
    def test_no_bias(self):
        # No bias should create roughly equal positive/negative
        effect = BiasedStretchEffect(max_stretch=0.5, stretch_bias=0.0, seed=42)
        factors = effect.generate_factors(width=1000)
        
        # Should be roughly balanced
        positive_count = np.sum(factors > 0)
        self.assertGreater(positive_count, 400)
        self.assertLess(positive_count, 600)


class TestCompositeEffect(unittest.TestCase):
    
    def test_composite_creation(self):
        wave = WaveDistortionEffect(max_stretch=0.3)
        bias = BiasedStretchEffect(max_stretch=0.5, stretch_bias=0.5)
        
        composite = CompositeEffect(
            effects=[wave, bias],
            weights=[0.7, 0.3]
        )
        
        # Test factor generation
        factors = composite.generate_factors(width=100)
        self.assertEqual(len(factors), 100)
        
        # Factors should be weighted combination
        # Maximum possible value is 0.7 * 0.3 + 0.3 * 0.5 = 0.36
        self.assertTrue(np.all(np.abs(factors) <= 0.36))
    
    def test_weight_normalization(self):
        effect1 = PivotStretchEffect(max_stretch=1.0)
        effect2 = PivotStretchEffect(max_stretch=1.0)
        
        # Weights should be normalized
        composite = CompositeEffect(
            effects=[effect1, effect2],
            weights=[2.0, 2.0]  # Will be normalized to [0.5, 0.5]
        )
        
        self.assertEqual(composite.weights, [0.5, 0.5])


if __name__ == '__main__':
    unittest.main()