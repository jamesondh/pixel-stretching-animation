#!/usr/bin/env python3
"""
Comprehensive demonstration of all pixel stretching features.
This script showcases every effect type, animation mode, and rendering option.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from src.pixel_stretcher import PixelStretcher
from src.config import get_preset
from src.distortion_effects import CompositeEffect, WaveDistortionEffect, BiasedStretchEffect


def create_test_image(size=64, pattern='rainbow'):
    """Create various test patterns for demonstration."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if pattern == 'rainbow':
        # Create gradient bands
        colors = [
            [255, 0, 0],     # Red
            [255, 127, 0],   # Orange
            [255, 255, 0],   # Yellow
            [0, 255, 0],     # Green
            [0, 0, 255],     # Blue
            [75, 0, 130],    # Indigo
            [148, 0, 211]    # Violet
        ]
        
        band_height = size // len(colors)
        for i, color in enumerate(colors):
            y_start = i * band_height
            y_end = (i + 1) * band_height if i < len(colors) - 1 else size
            img[y_start:y_end, :] = color
        
        # Add white grid lines
        img[::8, :] = 255
        img[:, ::8] = 255
    
    elif pattern == 'checkerboard':
        # Create checkerboard pattern
        for i in range(size):
            for j in range(size):
                if (i // 8 + j // 8) % 2 == 0:
                    img[i, j] = [255, 100, 100]  # Light red
                else:
                    img[i, j] = [100, 100, 255]  # Light blue
    
    elif pattern == 'gradient':
        # Create smooth gradient
        for i in range(size):
            intensity = int((i / size) * 255)
            img[i, :] = [intensity, intensity, 255 - intensity]
    
    elif pattern == 'circles':
        # Create concentric circles
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if int(dist) % 10 < 5:
                    img[i, j] = [255, 200, 100]
                else:
                    img[i, j] = [100, 200, 255]
    
    return Image.fromarray(img)


def demo_basic_effects(output_dir):
    """Demonstrate all basic effect types."""
    print("\n=== Basic Effects Demo ===")
    
    # Create test image
    test_img = create_test_image(pattern='rainbow')
    test_path = output_dir / "test_rainbow.png"
    test_img.save(test_path)
    
    effects = [
        ('pivot_center', {
            'max_stretch': 0.4,
            'pivot': 'center',
            'seed': 42
        }),
        ('pivot_top', {
            'max_stretch': 0.4,
            'pivot': 'top',
            'seed': 42
        }),
        ('wave_gentle', {
            'max_stretch': 0.3,
            'use_wave_distortion': True,
            'wave_amplitude': 0.1,
            'wave_frequency': 2.0,
            'wave_phase_speed': 1.0
        }),
        ('wave_intense', {
            'max_stretch': 0.5,
            'use_wave_distortion': True,
            'wave_amplitude': 0.2,
            'wave_frequency': 4.0,
            'wave_phase_speed': 2.0
        }),
        ('bias_melt', {
            'max_stretch': 0.6,
            'stretch_bias': 0.8,
            'seed': 42
        }),
        ('bias_float', {
            'max_stretch': 0.5,
            'stretch_bias': -0.7,
            'seed': 42
        })
    ]
    
    for effect_name, params in effects:
        print(f"\nCreating {effect_name} animation...")
        stretcher = PixelStretcher(**params)
        output_path = output_dir / f"demo_{effect_name}.mp4"
        
        stretcher.create_animation(
            input_path=test_path,
            output_path=output_path,
            frames=30,
            fps=15
        )
        print(f"  ✓ Saved: {output_path.name}")


def demo_animation_modes(output_dir):
    """Demonstrate different animation modes."""
    print("\n=== Animation Modes Demo ===")
    
    # Create test image
    test_img = create_test_image(pattern='gradient')
    test_path = output_dir / "test_gradient.png"
    test_img.save(test_path)
    
    modes = [
        ('standard', {'cumulative': False}),
        ('cumulative', {'cumulative': True})
    ]
    
    for mode_name, params in modes:
        print(f"\nCreating {mode_name} mode animation...")
        stretcher = PixelStretcher(
            max_stretch=0.4,
            use_wave_distortion=True,
            wave_amplitude=0.15,
            **params
        )
        
        output_path = output_dir / f"demo_mode_{mode_name}.mp4"
        stretcher.create_animation(
            input_path=test_path,
            output_path=output_path,
            frames=30,
            fps=15
        )
        print(f"  ✓ Saved: {output_path.name}")


def demo_rendering_options(output_dir):
    """Demonstrate rendering options."""
    print("\n=== Rendering Options Demo ===")
    
    # Create small pixel art
    test_img = create_test_image(size=16, pattern='checkerboard')
    test_path = output_dir / "test_pixelart.png"
    test_img.save(test_path)
    
    options = [
        ('nearest_1x', {
            'interpolation': 'nearest',
            'upscale': 1
        }),
        ('nearest_4x', {
            'interpolation': 'nearest',
            'upscale': 4
        }),
        ('bilinear_4x', {
            'interpolation': 'bilinear',
            'upscale': 4
        }),
        ('smoothed', {
            'interpolation': 'nearest',
            'upscale': 4,
            'temporal_smoothing': 0.8
        })
    ]
    
    for option_name, params in options:
        print(f"\nCreating {option_name} rendering...")
        stretcher = PixelStretcher(
            max_stretch=0.3,
            use_wave_distortion=True,
            wave_amplitude=0.1,
            **params
        )
        
        output_path = output_dir / f"demo_render_{option_name}.mp4"
        stretcher.create_animation(
            input_path=test_path,
            output_path=output_path,
            frames=20,
            fps=10
        )
        print(f"  ✓ Saved: {output_path.name}")


def demo_presets(output_dir):
    """Demonstrate all available presets."""
    print("\n=== Presets Demo ===")
    
    # Create test image
    test_img = create_test_image(pattern='circles')
    test_path = output_dir / "test_circles.png"
    test_img.save(test_path)
    
    preset_names = ['pixel_art', 'smooth_wave', 'melting', 'bouncy']
    
    for preset_name in preset_names:
        print(f"\nApplying preset: {preset_name}")
        config = get_preset(preset_name)
        
        # Create stretcher from preset config
        stretcher = PixelStretcher(
            max_stretch=config.effect.max_stretch,
            pivot=config.effect.pivot,
            interpolation=config.animation.interpolation,
            temporal_smoothing=config.animation.temporal_smoothing,
            upscale=config.animation.upscale,
            cumulative=config.animation.cumulative,
            use_wave_distortion=(config.effect.type == 'wave'),
            wave_amplitude=config.effect.wave_amplitude,
            wave_frequency=config.effect.wave_frequency,
            stretch_bias=config.effect.stretch_bias if config.effect.type == 'bias' else 0.0
        )
        
        output_path = output_dir / f"demo_preset_{preset_name}.mp4"
        stretcher.create_animation(
            input_path=test_path,
            output_path=output_path,
            frames=config.animation.frames,
            fps=config.animation.fps
        )
        print(f"  ✓ Saved: {output_path.name}")


def demo_advanced_combinations(output_dir):
    """Demonstrate advanced effect combinations."""
    print("\n=== Advanced Combinations Demo ===")
    
    # Create test image
    test_img = create_test_image(pattern='rainbow')
    test_path = output_dir / "test_rainbow.png"
    test_img.save(test_path)
    
    # Example 1: Melting with subtle wave
    print("\nCreating melting + wave combination...")
    stretcher = PixelStretcher(
        max_stretch=0.6,
        stretch_bias=0.7,  # Melting bias
        use_wave_distortion=True,
        wave_amplitude=0.05,  # Subtle wave
        wave_frequency=1.5,
        cumulative=True,
        upscale=2
    )
    
    output_path = output_dir / "demo_advanced_melt_wave.mp4"
    stretcher.create_animation(
        input_path=test_path,
        output_path=output_path,
        frames=60,
        fps=30
    )
    print(f"  ✓ Saved: {output_path.name}")
    
    # Example 2: High frequency waves with smoothing
    print("\nCreating high-frequency smoothed waves...")
    stretcher = PixelStretcher(
        max_stretch=0.4,
        use_wave_distortion=True,
        wave_amplitude=0.15,
        wave_frequency=6.0,
        wave_phase_speed=3.0,
        temporal_smoothing=0.7,
        interpolation='bilinear'
    )
    
    output_path = output_dir / "demo_advanced_smooth_waves.mp4"
    stretcher.create_animation(
        input_path=test_path,
        output_path=output_path,
        frames=90,
        fps=60
    )
    print(f"  ✓ Saved: {output_path.name}")


def create_summary_document(output_dir):
    """Create a summary document of all demos."""
    summary_path = output_dir / "demo_summary.md"
    
    with open(summary_path, 'w') as f:
        f.write("# Pixel Stretching Animation - Demo Summary\n\n")
        f.write("This directory contains demonstrations of all features:\n\n")
        
        f.write("## Basic Effects\n")
        f.write("- `demo_pivot_center.mp4` - Center pivot stretching\n")
        f.write("- `demo_pivot_top.mp4` - Top pivot (hanging effect)\n")
        f.write("- `demo_wave_gentle.mp4` - Gentle wave distortion\n")
        f.write("- `demo_wave_intense.mp4` - Intense wave distortion\n")
        f.write("- `demo_bias_melt.mp4` - Melting effect (positive bias)\n")
        f.write("- `demo_bias_float.mp4` - Floating effect (negative bias)\n\n")
        
        f.write("## Animation Modes\n")
        f.write("- `demo_mode_standard.mp4` - Standard animation mode\n")
        f.write("- `demo_mode_cumulative.mp4` - Cumulative distortion mode\n\n")
        
        f.write("## Rendering Options\n")
        f.write("- `demo_render_nearest_1x.mp4` - Nearest neighbor, no upscaling\n")
        f.write("- `demo_render_nearest_4x.mp4` - Nearest neighbor, 4x upscale\n")
        f.write("- `demo_render_bilinear_4x.mp4` - Bilinear interpolation, 4x upscale\n")
        f.write("- `demo_render_smoothed.mp4` - Temporal smoothing applied\n\n")
        
        f.write("## Presets\n")
        f.write("- `demo_preset_pixel_art.mp4` - Pixel art preset\n")
        f.write("- `demo_preset_smooth_wave.mp4` - Smooth wave preset\n")
        f.write("- `demo_preset_melting.mp4` - Melting preset\n")
        f.write("- `demo_preset_bouncy.mp4` - Bouncy preset\n\n")
        
        f.write("## Advanced Combinations\n")
        f.write("- `demo_advanced_melt_wave.mp4` - Melting with subtle waves\n")
        f.write("- `demo_advanced_smooth_waves.mp4` - High-frequency smoothed waves\n")
    
    print(f"\n✓ Created summary: {summary_path.name}")


def main():
    """Run all demonstrations."""
    print("=" * 50)
    print("Pixel Stretching Animation - Feature Demonstration")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("../output/demos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all demos
    demo_basic_effects(output_dir)
    demo_animation_modes(output_dir)
    demo_rendering_options(output_dir)
    demo_presets(output_dir)
    demo_advanced_combinations(output_dir)
    
    # Create summary
    create_summary_document(output_dir)
    
    print("\n" + "=" * 50)
    print(f"All demonstrations complete! Check {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()