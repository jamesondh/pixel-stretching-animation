"""
Unit tests for configuration management.
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    EffectConfig, AnimationConfig, OutputConfig, PixelStretchConfig,
    get_preset, PRESETS
)


class TestEffectConfig(unittest.TestCase):
    
    def test_default_values(self):
        config = EffectConfig()
        
        self.assertEqual(config.type, 'pivot')
        self.assertEqual(config.max_stretch, 0.5)
        self.assertEqual(config.pivot, 'center')
        self.assertEqual(config.stretch_bias, 0.0)
    
    def test_validation(self):
        # Valid config
        config = EffectConfig(type='wave', max_stretch=0.7)
        config.validate()  # Should not raise
        
        # Invalid effect type
        config = EffectConfig(type='invalid')
        with self.assertRaises(ValueError):
            config.validate()
        
        # Invalid max_stretch
        config = EffectConfig(max_stretch=1.5)
        with self.assertRaises(ValueError):
            config.validate()
        
        # Invalid pivot
        config = EffectConfig(pivot='invalid')
        with self.assertRaises(ValueError):
            config.validate()
        
        # Invalid stretch_bias
        config = EffectConfig(stretch_bias=2.0)
        with self.assertRaises(ValueError):
            config.validate()


class TestAnimationConfig(unittest.TestCase):
    
    def test_default_values(self):
        config = AnimationConfig()
        
        self.assertEqual(config.frames, 60)
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.interpolation, 'nearest')
        self.assertFalse(config.cumulative)
    
    def test_validation(self):
        # Valid config
        config = AnimationConfig(frames=120, fps=60)
        config.validate()  # Should not raise
        
        # Invalid frames
        config = AnimationConfig(frames=0)
        with self.assertRaises(ValueError):
            config.validate()
        
        # Invalid interpolation
        config = AnimationConfig(interpolation='invalid')
        with self.assertRaises(ValueError):
            config.validate()


class TestPixelStretchConfig(unittest.TestCase):
    
    def test_creation(self):
        config = PixelStretchConfig()
        
        self.assertIsInstance(config.effect, EffectConfig)
        self.assertIsInstance(config.animation, AnimationConfig)
        self.assertIsInstance(config.output, OutputConfig)
    
    def test_from_dict(self):
        config_dict = {
            'effect': {
                'type': 'wave',
                'max_stretch': 0.6,
                'wave_amplitude': 0.2
            },
            'animation': {
                'frames': 90,
                'fps': 30,
                'cumulative': True
            }
        }
        
        config = PixelStretchConfig.from_dict(config_dict)
        
        self.assertEqual(config.effect.type, 'wave')
        self.assertEqual(config.effect.max_stretch, 0.6)
        self.assertEqual(config.effect.wave_amplitude, 0.2)
        self.assertEqual(config.animation.frames, 90)
        self.assertTrue(config.animation.cumulative)
    
    def test_to_dict(self):
        config = PixelStretchConfig()
        config.effect.type = 'bias'
        config.effect.stretch_bias = 0.7
        config.animation.frames = 120
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['effect']['type'], 'bias')
        self.assertEqual(config_dict['effect']['stretch_bias'], 0.7)
        self.assertEqual(config_dict['animation']['frames'], 120)
    
    def test_save_and_load_json(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Create and save config
            config = PixelStretchConfig()
            config.effect.type = 'wave'
            config.effect.wave_frequency = 3.5
            config.save(temp_path)
            
            # Load config
            loaded_config = PixelStretchConfig.from_file(temp_path)
            
            self.assertEqual(loaded_config.effect.type, 'wave')
            self.assertEqual(loaded_config.effect.wave_frequency, 3.5)
        
        finally:
            temp_path.unlink()
    
    def test_validation(self):
        config = PixelStretchConfig()
        config.validate()  # Should not raise with defaults
        
        # Make invalid
        config.effect.max_stretch = 2.0
        with self.assertRaises(ValueError):
            config.validate()


class TestPresets(unittest.TestCase):
    
    def test_all_presets_exist(self):
        expected_presets = ['pixel_art', 'smooth_wave', 'melting', 'bouncy']
        
        for preset_name in expected_presets:
            self.assertIn(preset_name, PRESETS)
    
    def test_get_preset(self):
        # Get valid preset
        config = get_preset('pixel_art')
        
        self.assertIsInstance(config, PixelStretchConfig)
        self.assertEqual(config.animation.interpolation, 'nearest')
        self.assertEqual(config.animation.upscale, 4)
        
        # Get invalid preset
        with self.assertRaises(ValueError):
            get_preset('invalid_preset')
    
    def test_preset_isolation(self):
        # Modifying a preset shouldn't affect the original
        config1 = get_preset('melting')
        config1.effect.max_stretch = 0.9
        
        config2 = get_preset('melting')
        
        # Original should be unchanged
        self.assertNotEqual(config1.effect.max_stretch, config2.effect.max_stretch)
    
    def test_all_presets_valid(self):
        # All presets should pass validation
        for preset_name in PRESETS:
            config = get_preset(preset_name)
            config.validate()  # Should not raise


if __name__ == '__main__':
    unittest.main()