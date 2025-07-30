#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pixel_stretcher import PixelStretcher
import numpy as np
from PIL import Image


def create_test_image(size=32):
    """Create a simple test pattern for demonstration."""
    # Create a gradient pattern to better visualize stretching direction
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create horizontal stripes
    for i in range(size):
        color_intensity = int((i / size) * 255)
        if i % 8 < 4:
            image[i, :] = [255, color_intensity, 100]  # Red gradient
        else:
            image[i, :] = [100, color_intensity, 255]  # Blue gradient
    
    return image


def main():
    print("Creating pixel stretch animations with different bias settings...")
    
    # Create output directory
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # Create test image
    test_image = create_test_image()
    test_path = output_dir / "test_gradient.png"
    Image.fromarray(test_image).save(test_path)
    print(f"Test image saved to {test_path}")
    
    # Create animations with different bias settings
    bias_settings = [
        (-0.8, "strong_negative_bias"),  # Strong bias towards compression (negative stretch)
        (-0.4, "mild_negative_bias"),    # Mild negative bias
        (0.0, "no_bias"),                # No bias (original behavior)
        (0.4, "mild_positive_bias"),     # Mild positive bias
        (0.8, "strong_positive_bias")    # Strong bias towards expansion (positive stretch)
    ]
    
    for bias, name in bias_settings:
        print(f"\nCreating animation with bias={bias} ({name})...")
        
        stretcher = PixelStretcher(
            max_stretch=0.5,
            stretch_bias=bias,
            upscale=4,  # Upscale for better visibility
            seed=42     # Fixed seed for reproducible results
        )
        
        output_path = output_dir / f"biased_stretch_{name}.mp4"
        stretcher.create_animation(
            input_path=test_path,
            output_path=output_path,
            frames=30,
            fps=15
        )
        print(f"Animation saved to {output_path}")
    
    # Create a comparison with cumulative mode
    print("\nCreating cumulative animation with positive bias...")
    stretcher_cumulative = PixelStretcher(
        max_stretch=0.3,
        stretch_bias=0.6,
        cumulative=True,
        upscale=4,
        seed=42
    )
    
    output_path = output_dir / "biased_stretch_cumulative.mp4"
    stretcher_cumulative.create_animation(
        input_path=test_path,
        output_path=output_path,
        frames=30,
        fps=15
    )
    print(f"Cumulative animation saved to {output_path}")
    
    print("\nAll animations created successfully!")
    print("\nBias parameter explanation:")
    print("  - stretch_bias = -1.0: Strong bias towards upward stretching")
    print("  - stretch_bias = 0.0:  No bias (random up/down stretching)")
    print("  - stretch_bias = 1.0:  Strong bias towards downward stretching (melting effect)")
    print("\nWith positive bias, pixels appear to melt downward with occasional upward movement.")
    print("With negative bias, pixels appear to float upward with occasional downward movement.")


if __name__ == "__main__":
    main()