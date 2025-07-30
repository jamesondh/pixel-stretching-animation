#!/usr/bin/env python3
"""
Simple example demonstrating basic pixel stretching animation.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from src.pixel_stretcher import PixelStretcher


def create_sample_sprite():
    """Create a simple pixel art sprite for demonstration."""
    # 16x16 pixel smiley face
    size = 16
    sprite = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Background
    sprite[:, :] = [100, 150, 255]  # Light blue
    
    # Face outline (circle)
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if 6 <= dist <= 7:
                sprite[i, j] = [0, 0, 0]  # Black outline
    
    # Eyes
    sprite[5:6, 5:6] = [0, 0, 0]  # Left eye
    sprite[5:6, 10:11] = [0, 0, 0]  # Right eye
    
    # Smile
    sprite[10, 5:11] = [0, 0, 0]  # Mouth line
    sprite[9, 4:5] = [0, 0, 0]  # Left corner
    sprite[9, 11:12] = [0, 0, 0]  # Right corner
    
    return sprite


def main():
    print("Simple Pixel Stretching Example")
    print("=" * 30)
    
    # Create output directory
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # Create and save sample sprite
    sprite = create_sample_sprite()
    sprite_img = Image.fromarray(sprite)
    sprite_path = output_dir / "sample_sprite.png"
    sprite_img.save(sprite_path)
    print(f"Created sample sprite: {sprite_path}")
    
    # Example 1: Basic stretching
    print("\n1. Basic stretching with default settings...")
    stretcher = PixelStretcher(
        max_stretch=0.4,
        pivot='center',
        upscale=4  # Upscale for better visibility
    )
    
    stretcher.create_animation(
        input_path=sprite_path,
        output_path=output_dir / "simple_basic.mp4",
        frames=30,
        fps=15
    )
    print("   ✓ Created: simple_basic.mp4")
    
    # Example 2: Smooth animation
    print("\n2. Smooth animation with temporal smoothing...")
    stretcher = PixelStretcher(
        max_stretch=0.5,
        temporal_smoothing=0.7,
        upscale=4
    )
    
    stretcher.create_animation(
        input_path=sprite_path,
        output_path=output_dir / "simple_smooth.mp4",
        frames=30,
        fps=15
    )
    print("   ✓ Created: simple_smooth.mp4")
    
    # Example 3: Different pivot points
    print("\n3. Comparing different pivot points...")
    for pivot in ['top', 'center', 'bottom']:
        stretcher = PixelStretcher(
            max_stretch=0.4,
            pivot=pivot,
            upscale=4
        )
        
        stretcher.create_animation(
            input_path=sprite_path,
            output_path=output_dir / f"simple_pivot_{pivot}.mp4",
            frames=20,
            fps=10
        )
        print(f"   ✓ Created: simple_pivot_{pivot}.mp4")
    
    print("\nAll examples complete! Check the output directory.")


if __name__ == "__main__":
    main()