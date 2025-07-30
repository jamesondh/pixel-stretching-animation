#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from src.pixel_stretcher import PixelStretcher


def create_pixel_art():
    # Create a small pixel art image to demonstrate upscaling
    size = 16
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a simple smiley face
    # Background
    img[:, :] = [100, 150, 255]  # Light blue
    
    # Eyes
    img[5:7, 4:6] = [0, 0, 0]  # Left eye
    img[5:7, 10:12] = [0, 0, 0]  # Right eye
    
    # Mouth
    img[10:11, 5:11] = [0, 0, 0]  # Horizontal line
    img[9:10, 4:5] = [0, 0, 0]  # Left corner
    img[9:10, 11:12] = [0, 0, 0]  # Right corner
    
    return Image.fromarray(img)


def main():
    print("Demonstrating upscaling with nearest neighbor...")
    
    # Create output directory
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # Create pixel art
    pixel_art = create_pixel_art()
    pixel_path = output_dir / "pixel_smiley.png"
    pixel_art.save(pixel_path)
    print(f"Created pixel art: {pixel_path}")
    
    # Different upscale factors to demonstrate
    upscale_factors = [1, 2, 3, 4]
    
    for factor in upscale_factors:
        print(f"\nCreating animation with {factor}x upscaling...")
        
        stretcher = PixelStretcher(
            max_stretch=0.4,
            pivot='center',
            interpolation='nearest',
            temporal_smoothing=0.1,
            upscale=factor
        )
        
        output_path = output_dir / f"upscale_{factor}x.mp4"
        stretcher.create_animation(
            input_path=pixel_path,
            output_path=output_path,
            frames=45,
            fps=24
        )
        
        print(f"Saved: {output_path} (output size: {16*factor}x{16*factor})")
    
    # Create a comparison with bilinear interpolation at different scales
    print("\nCreating comparison with bilinear interpolation...")
    
    for interp_method in ['nearest', 'bilinear']:
        stretcher = PixelStretcher(
            max_stretch=0.4,
            pivot='center',
            interpolation=interp_method,
            temporal_smoothing=0.1,
            upscale=4
        )
        
        output_path = output_dir / f"upscale_4x_{interp_method}.mp4"
        stretcher.create_animation(
            input_path=pixel_path,
            output_path=output_path,
            frames=45,
            fps=24
        )
        
        print(f"Saved {interp_method} interpolation: {output_path}")
    
    print("\nUpscaling demonstration complete!")
    print("Note: Nearest neighbor preserves sharp pixel edges, perfect for pixel art!")


if __name__ == "__main__":
    main()