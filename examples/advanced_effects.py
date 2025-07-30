#!/usr/bin/env python3

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from src.pixel_stretcher import PixelStretcher
from src.effects import StretchEffects, TemporalEffects, AnimationCurves, create_animated_sequence


class AdvancedPixelStretcher(PixelStretcher):
    def __init__(self, effect_type='wave', **kwargs):
        super().__init__(**kwargs)
        self.effect_type = effect_type
        self.frame_count = 0
        self.total_frames = 60
        
    def _generate_stretch_factors(self, width: int) -> np.ndarray:
        self.frame_count += 1
        t = self.frame_count / self.total_frames
        
        if self.effect_type == 'wave':
            # Create animated wave effect
            base = StretchEffects.wave_pattern(
                width, 
                frequency=0.1, 
                amplitude=self.max_stretch,
                phase=t * 2 * np.pi
            )
        elif self.effect_type == 'smooth':
            # Use smooth noise
            base = StretchEffects.smooth_noise(width, smoothness=0.7) * self.max_stretch
        elif self.effect_type == 'perlin':
            # Use Perlin noise
            base = StretchEffects.perlin_noise(width, scale=0.1, octaves=2) * self.max_stretch
        elif self.effect_type == 'combined':
            # Combined wave patterns
            base = StretchEffects.combined_wave(
                width,
                frequencies=[0.05, 0.1, 0.15],
                amplitudes=[0.5, 0.3, 0.2],
                phases=[t * 2 * np.pi, t * 3 * np.pi, t * 4 * np.pi]
            ) * self.max_stretch
        else:
            # Default to random
            base = super()._generate_stretch_factors(width)
        
        return base


def create_sample_image():
    # Create a colorful pixel art sample
    size = 64
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
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
    
    # Add some pixel patterns
    for x in range(0, size, 4):
        img[:, x:x+1] = 255  # White vertical lines
    
    return Image.fromarray(img)


def main():
    print("Demonstrating advanced pixel stretching effects...")
    
    # Create output directory
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample image
    sample_img = create_sample_image()
    sample_path = output_dir / "sample_rainbow.png"
    sample_img.save(sample_path)
    print(f"Created sample image: {sample_path}")
    
    # Effect types to demonstrate
    effects = {
        'wave': {'max_stretch': 0.3, 'temporal_smoothing': 0.2},
        'smooth': {'max_stretch': 0.5, 'temporal_smoothing': 0.5},
        'perlin': {'max_stretch': 0.4, 'temporal_smoothing': 0.3},
        'combined': {'max_stretch': 0.6, 'temporal_smoothing': 0.1}
    }
    
    # Generate animations for each effect
    for effect_name, params in effects.items():
        print(f"\nCreating {effect_name} effect animation...")
        
        stretcher = AdvancedPixelStretcher(
            effect_type=effect_name,
            pivot='center',
            interpolation='nearest',
            **params
        )
        
        output_path = output_dir / f"effect_{effect_name}.mp4"
        stretcher.create_animation(
            input_path=sample_path,
            output_path=output_path,
            frames=60,
            fps=30
        )
        
        print(f"Saved: {output_path}")
    
    # Create an easing demonstration
    print("\nCreating easing demonstration...")
    
    # This would require modifying the PixelStretcher to accept pre-computed sequences
    # For now, we'll create a simple bounce effect
    bounce_stretcher = AdvancedPixelStretcher(
        effect_type='wave',
        max_stretch=0.8,
        pivot='center'
    )
    
    bounce_output = output_dir / "effect_bounce.mp4"
    bounce_stretcher.create_animation(
        input_path=sample_path,
        output_path=bounce_output,
        frames=90,
        fps=30
    )
    
    print(f"Saved bounce effect: {bounce_output}")
    print("\nAll demonstrations complete!")


if __name__ == "__main__":
    main()