#!/usr/bin/env python3
"""
Demo of constant stretch animation - maintains consistent distortion throughout.
This solves the issue of unprocessed first frames.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pixel_stretcher import PixelStretcher
from src.config import PixelStretchConfig


def main():
    # Example 1: Constant stretch - no progression
    print("Creating constant stretch animation...")
    stretcher = PixelStretcher(
        max_stretch=0.4,
        stretch_curve='constant',  # Always applies full stretch
        interpolation='nearest',
        upscale=3,
        seed=42
    )
    
    stretcher.create_animation(
        input_path='assets/Sprite-0001.png',
        output_path='output/constant_stretch.mp4',
        frames=60,
        fps=30
    )
    print("✓ Constant stretch animation saved to output/constant_stretch.mp4")
    
    # Example 2: Constant with custom value
    print("\nCreating constant stretch with custom value...")
    stretcher2 = PixelStretcher(
        max_stretch=0.8,  # This is ignored
        stretch_curve='constant',
        end_stretch=0.3,  # This is what constant uses
        interpolation='nearest',
        upscale=3,
        seed=42
    )
    
    stretcher2.create_animation(
        input_path='assets/Sprite-0001.png',
        output_path='output/constant_custom.mp4',
        frames=60,
        fps=30
    )
    print("✓ Custom constant animation saved to output/constant_custom.mp4")
    
    # Example 3: Comparison with linear (default)
    print("\nCreating linear stretch for comparison...")
    stretcher3 = PixelStretcher(
        max_stretch=0.4,
        stretch_curve='linear',  # Default behavior
        interpolation='nearest',
        upscale=3,
        seed=42
    )
    
    stretcher3.create_animation(
        input_path='assets/Sprite-0001.png',
        output_path='output/linear_stretch.mp4',
        frames=60,
        fps=30
    )
    print("✓ Linear stretch animation saved to output/linear_stretch.mp4")
    
    # Example 4: Start with partial stretch
    print("\nCreating animation that starts partially stretched...")
    stretcher4 = PixelStretcher(
        stretch_curve='linear',
        start_stretch=0.2,  # Start at 20% distortion
        end_stretch=0.6,    # End at 60% distortion
        interpolation='nearest',
        upscale=3,
        seed=42
    )
    
    stretcher4.create_animation(
        input_path='assets/Sprite-0001.png',
        output_path='output/partial_stretch.mp4',
        frames=60,
        fps=30
    )
    print("✓ Partial stretch animation saved to output/partial_stretch.mp4")
    
    print("\nDone! Compare the animations to see the difference:")
    print("- constant_stretch.mp4: Maintains consistent distortion")
    print("- linear_stretch.mp4: Starts unprocessed, gradually distorts")
    print("- partial_stretch.mp4: Never shows unprocessed image")


if __name__ == '__main__':
    main()