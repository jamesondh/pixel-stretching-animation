#!/usr/bin/env python3
"""
Demonstration of the shared transform architecture.

This example shows how the same sine wave transformation can be used
in both the generation phase (as a distortion effect) and the
post-processing phase (as a post-processor).
"""

from pathlib import Path
from src.pixel_stretcher import PixelStretcher
from src.distortion_effects import SineWaveDistortionEffect
from src.post_processors import SineWavePostProcessor, PostProcessorChain
from src.transforms import SineWaveTransform


def demonstrate_shared_transforms():
    """Show how transforms are shared between effects and post-processors."""
    
    # Input and output paths
    input_path = Path("assets/Sprite-0001.png")
    if not input_path.exists():
        print(f"Please ensure {input_path} exists")
        return
    
    print("Shared Transform Architecture Demo")
    print("=================================\n")
    
    # 1. Direct use of shared transform
    print("1. Direct Transform Usage:")
    transform = SineWaveTransform(
        frequency=3.0,
        amplitude=0.1,
        phase=0.0,
        speed=0.5,
        axis='vertical'
    )
    
    # Calculate some sample displacements
    dx, dy = transform.calculate_frame_displacement(100, 100, time=0.5)
    print(f"   Sample displacement at center: dx={dx[50,50]:.3f}, dy={dy[50,50]:.3f}")
    
    # 2. Using in distortion effect
    print("\n2. Generation Phase (Distortion Effect):")
    sine_effect = SineWaveDistortionEffect(
        max_stretch=0.4,
        frequency=3.0,
        amplitude=0.15,
        axis='vertical'
    )
    
    stretcher = PixelStretcher(effect=sine_effect)
    output_generation = Path("output/shared_demo_generation.mp4")
    output_generation.parent.mkdir(exist_ok=True)
    
    stretcher.create_animation(
        str(input_path),
        str(output_generation),
        frames=30,
        fps=30
    )
    print(f"   Created: {output_generation}")
    
    # 3. Using in post-processing
    print("\n3. Post-Processing Phase:")
    sine_processor = SineWavePostProcessor(
        frequency=3.0,
        amplitude=0.05,
        axis='horizontal',  # Different axis for contrast
        speed=0.5
    )
    
    # Apply to a simple pivot animation
    stretcher_simple = PixelStretcher(max_stretch=0.3)
    output_postprocess = Path("output/shared_demo_postprocess.mp4")
    
    stretcher_simple.create_animation(
        str(input_path),
        str(output_postprocess),
        frames=30,
        fps=30,
        post_processor=sine_processor
    )
    print(f"   Created: {output_postprocess}")
    
    # 4. Combining both
    print("\n4. Combined Generation + Post-Processing:")
    output_combined = Path("output/shared_demo_combined.mp4")
    
    # Use sine wave in generation with vertical axis
    stretcher_combined = PixelStretcher(effect=sine_effect)
    
    # And post-process with horizontal sine wave
    stretcher_combined.create_animation(
        str(input_path),
        str(output_combined),
        frames=30,
        fps=30,
        post_processor=sine_processor
    )
    print(f"   Created: {output_combined}")
    
    print("\n5. Mathematical Consistency:")
    print("   Both effects use the same SineWaveTransform class")
    print("   This ensures consistent wave calculations")
    print("   Only the application method differs:")
    print("   - Distortion: Column-by-column during generation")
    print("   - Post-processing: Full frame after generation")
    
    print("\nDemo complete! Check the output/ directory for results.")


if __name__ == "__main__":
    demonstrate_shared_transforms()