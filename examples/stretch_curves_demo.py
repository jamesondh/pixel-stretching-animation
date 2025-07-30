#!/usr/bin/env python3
"""
Demonstrates the stretch_curves feature in CompositeEffect.
Shows how different timing curves affect animation dynamics.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pixel_stretcher import PixelStretcher
from src.distortion_effects import CompositeEffect, WaveDistortionEffect, BiasedStretchEffect, PivotStretchEffect
from PIL import Image
import numpy as np


def create_demo_image():
    """Create a simple test image with gradients."""
    width, height = 200, 200
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            # Create a gradient pattern
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = int(((x + y) / (width + height)) * 255)
            pixels[x, y] = (r, g, b)
    
    return img


def demo_constant_vs_linear():
    """Show the difference between constant and linear stretch curves."""
    print("Demo 1: Constant vs Linear stretch curves")
    
    # Create effects
    wave = WaveDistortionEffect(
        max_stretch=0.3,
        wave_amplitude=0.15,
        wave_frequency=2
    )
    
    melt = BiasedStretchEffect(
        max_stretch=0.5,
        stretch_bias=0.8
    )
    
    # Create composite with different curves
    composite = CompositeEffect(
        effects=[wave, melt],
        weights=[0.5, 0.5],
        stretch_curves=['linear', 'constant']  # Wave scales up, melt stays constant
    )
    
    # Create animation
    stretcher = PixelStretcher(effect=composite)
    img = create_demo_image()
    img.save('examples/output/demo_image.png')
    
    stretcher.create_animation(
        'examples/output/demo_image.png',
        'examples/output/constant_vs_linear.mp4',
        frames=60,
        fps=30
    )
    print("Created: constant_vs_linear.mp4")
    print("  - Wave effect gradually increases (linear)")
    print("  - Melt effect stays at full strength (constant)")
    print()


def demo_easing_curves():
    """Demonstrate all easing curve types."""
    print("Demo 2: Easing curves showcase")
    
    # Create multiple wave effects with different frequencies
    effects = []
    curves = ['linear', 'ease_in', 'ease_out', 'ease_in_out', 'constant']
    
    for i, curve in enumerate(curves):
        wave = WaveDistortionEffect(
            max_stretch=0.4,
            wave_amplitude=0.1,
            wave_frequency=i + 1,  # Different frequency for each
            wave_phase_shift=i * 0.5  # Different phase for variety
        )
        effects.append(wave)
    
    # Create composite with all curve types
    composite = CompositeEffect(
        effects=effects,
        weights=[0.2] * 5,  # Equal weight for all
        stretch_curves=curves
    )
    
    # Create animation
    stretcher = PixelStretcher(effect=composite)
    img = create_demo_image()
    
    stretcher.create_animation(
        'examples/output/demo_image.png',
        'examples/output/easing_curves.mp4',
        frames=90,
        fps=30
    )
    print("Created: easing_curves.mp4")
    print("  - Linear: steady progression")
    print("  - Ease In: starts slow, accelerates")
    print("  - Ease Out: starts fast, decelerates")
    print("  - Ease In/Out: slow at both ends")
    print("  - Constant: full strength throughout")
    print()


def demo_creative_combination():
    """Creative use of stretch curves for complex animation."""
    print("Demo 3: Creative combination - Ripple that fades as melting increases")
    
    # Ripple effect that fades out
    ripple = WaveDistortionEffect(
        max_stretch=0.25,
        wave_amplitude=0.2,
        wave_frequency=4,
        wave_phase_speed=0.3
    )
    
    # Melting that increases
    melt = BiasedStretchEffect(
        max_stretch=0.6,
        stretch_bias=0.9
    )
    
    # Subtle pivot wobble throughout
    wobble = PivotStretchEffect(
        max_stretch=0.15,
        pivot='center'
    )
    
    # Combine with creative curves
    composite = CompositeEffect(
        effects=[ripple, melt, wobble],
        weights=[0.4, 0.5, 0.1],
        stretch_curves=['ease_out', 'ease_in', 'constant']
    )
    
    # Create animation with cumulative mode for dramatic effect
    stretcher = PixelStretcher(effect=composite, temporal_smoothing=0.7)
    img = create_demo_image()
    
    stretcher.create_animation(
        'examples/output/demo_image.png',
        'examples/output/creative_combination.mp4',
        frames=120,
        fps=30,
        mode='cumulative'  # For progressive melting
    )
    print("Created: creative_combination.mp4")
    print("  - Ripple effect starts strong and fades (ease_out)")
    print("  - Melting gradually takes over (ease_in)")
    print("  - Subtle wobble throughout (constant)")
    print("  - Cumulative mode creates progressive distortion")
    print()


def main():
    """Run all demonstrations."""
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    print("Stretch Curves Feature Demonstration")
    print("=" * 40)
    print()
    
    # Run demos
    demo_constant_vs_linear()
    demo_easing_curves()
    demo_creative_combination()
    
    print("All demonstrations complete!")
    print("Check the examples/output/ directory for the generated animations.")
    print()
    print("Tips for using stretch curves:")
    print("- Use 'constant' for effects that should maintain full intensity")
    print("- Use 'ease_in' for effects that should build up gradually")
    print("- Use 'ease_out' for effects that should fade away")
    print("- Use 'ease_in_out' for effects that pulse or breathe")
    print("- Combine different curves to create complex animation dynamics")


if __name__ == '__main__':
    main()