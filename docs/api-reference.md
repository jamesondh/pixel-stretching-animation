# API Reference

## Core Classes

### PixelStretcher

The main class for creating pixel stretching animations.

```python
from src.pixel_stretcher import PixelStretcher

stretcher = PixelStretcher(
    max_stretch=0.5,
    pivot='center',
    interpolation='nearest',
    temporal_smoothing=0.0,
    seed=None,
    upscale=1,
    cumulative=False,
    wave_amplitude=0.1,
    wave_frequency=2.0,
    wave_phase_shift=0.0,
    wave_phase_speed=0.2,
    use_wave_distortion=False,
    stretch_bias=0.0
)
```

#### Parameters

- **max_stretch** (float, 0-1): Maximum stretch factor applied to columns
- **pivot** (str): Pivot point for stretching ('center', 'top', 'bottom')
- **interpolation** (str): Pixel interpolation method ('nearest', 'bilinear')
- **temporal_smoothing** (float, 0-1): Smoothing between frames
- **seed** (int, optional): Random seed for reproducible animations
- **upscale** (int): Upscaling factor (1, 2, 3, 4, etc.)
- **cumulative** (bool): Apply distortion cumulatively to each frame
- **wave_amplitude** (float): Amplitude of wave distortion
- **wave_frequency** (float): Frequency of wave pattern
- **wave_phase_shift** (float): Initial phase offset for waves
- **wave_phase_speed** (float): Speed of wave animation
- **use_wave_distortion** (bool): Enable wave-based distortion
- **stretch_bias** (float, -1 to 1): Bias for directional stretching

#### Methods

##### create_animation()
```python
stretcher.create_animation(
    input_path='input.png',
    output_path='output.mp4',
    frames=60,
    fps=30,
    loop=0
)
```

Creates an animation from a static image.

**Parameters:**
- **input_path** (str/Path): Path to input image
- **output_path** (str/Path): Path for output video
- **frames** (int): Number of frames to generate
- **fps** (int): Frames per second
- **loop** (int): Number of loops (unused for video formats)

##### warp_frame()
```python
warped = stretcher.warp_frame(image_array, stretch_scale=1.0, apply_upscale=True, time_phase=0.0)
```

Apply distortion to a single frame.

**Parameters:**
- **image** (np.ndarray): Input image as numpy array
- **stretch_scale** (float): Scale factor for stretch intensity
- **apply_upscale** (bool): Whether to apply upscaling
- **time_phase** (float): Time-based phase for animated effects

**Returns:** Warped image as numpy array

##### reset()
```python
stretcher.reset()
```

Reset internal state (useful when processing multiple animations).

### Distortion Effects

#### DistortionEffect (Base Class)

Abstract base class for all distortion effects.

```python
from src.distortion_effects import DistortionEffect

class CustomEffect(DistortionEffect):
    def generate_factors(self, width, frame=0, total_frames=60):
        # Return array of stretch factors for each column
        pass
    
    def warp_column(self, column, factor, column_index, total_width, height):
        # Return warped column
        pass
```

#### BiasedStretchEffect

Creates directional stretching with configurable bias.

```python
from src.distortion_effects import BiasedStretchEffect

effect = BiasedStretchEffect(
    max_stretch=0.5,
    stretch_bias=0.8,  # Strong downward bias (melting)
    seed=42
)
```

#### WaveDistortionEffect

Creates wave-based distortion patterns.

```python
from src.distortion_effects import WaveDistortionEffect

effect = WaveDistortionEffect(
    max_stretch=0.5,
    wave_amplitude=0.15,
    wave_frequency=2.0,
    wave_phase_shift=0.0,
    wave_phase_speed=0.2,
    seed=42
)
```

#### PivotStretchEffect

Original pivot-based stretching.

```python
from src.distortion_effects import PivotStretchEffect

effect = PivotStretchEffect(
    max_stretch=0.5,
    pivot='center',  # or 'top', 'bottom'
    seed=42
)
```

#### CompositeEffect

Combine multiple effects.

```python
from src.distortion_effects import CompositeEffect, WaveDistortionEffect, BiasedStretchEffect

wave = WaveDistortionEffect(max_stretch=0.3)
bias = BiasedStretchEffect(max_stretch=0.5, stretch_bias=0.5)

composite = CompositeEffect(
    effects=[wave, bias],
    weights=[0.7, 0.3]  # 70% wave, 30% bias
)
```

### Animation Classes

#### AnimationEngine

Main engine for creating animations.

```python
from src.animation import AnimationEngine, CumulativeFrameGenerator

engine = AnimationEngine(
    frame_generator=CumulativeFrameGenerator()
)

engine.create_animation(
    input_path='input.png',
    output_path='output.mp4',
    warp_func=my_warp_function,
    frames=60,
    fps=30
)
```

#### Frame Generators

Different strategies for generating animation frames:

- **StandardFrameGenerator**: Apply increasing distortion to original
- **CumulativeFrameGenerator**: Each frame builds on previous
- **PingPongFrameGenerator**: Forward then backward playback

### Configuration

#### PixelStretchConfig

Configuration management for animations.

```python
from src.config import PixelStretchConfig, EffectConfig, AnimationConfig

# Create configuration
config = PixelStretchConfig(
    effect=EffectConfig(
        type='wave',
        max_stretch=0.5,
        wave_amplitude=0.2
    ),
    animation=AnimationConfig(
        frames=90,
        fps=30,
        cumulative=True
    )
)

# Save configuration
config.save('my_config.yaml')

# Load configuration
loaded_config = PixelStretchConfig.from_file('my_config.yaml')
```

#### Presets

Pre-configured settings for common use cases:

```python
from src.config import get_preset

# Get a preset configuration
config = get_preset('pixel_art')  # or 'smooth_wave', 'melting', 'bouncy'
```

## Advanced Usage Examples

### Custom Effect Implementation

```python
from src.distortion_effects import DistortionEffect
import numpy as np

class SpiralEffect(DistortionEffect):
    def __init__(self, max_stretch=0.5, spiral_tightness=2.0):
        self.max_stretch = max_stretch
        self.spiral_tightness = spiral_tightness
    
    def generate_factors(self, width, frame=0, total_frames=60):
        t = frame / max(total_frames - 1, 1)
        factors = np.zeros(width)
        
        for i in range(width):
            angle = (i / width) * 2 * np.pi * self.spiral_tightness
            factors[i] = np.sin(angle + t * 2 * np.pi) * self.max_stretch
        
        return factors
    
    def warp_column(self, column, factor, column_index, total_width, height):
        # Implementation similar to other effects
        pass
```

### Batch Processing

```python
from pathlib import Path
from src.pixel_stretcher import PixelStretcher

# Process all PNG files in a directory
input_dir = Path('input_images')
output_dir = Path('output_videos')
output_dir.mkdir(exist_ok=True)

stretcher = PixelStretcher(max_stretch=0.5, upscale=2)

for img_path in input_dir.glob('*.png'):
    output_path = output_dir / f"{img_path.stem}_stretched.mp4"
    stretcher.create_animation(img_path, output_path)
    print(f"Processed: {img_path.name}")
```

### Animation Sequences

```python
from src.animation import AnimationSequence

sequence = AnimationSequence()

# Add different effects in sequence
sequence.add_segment(
    effect_func=lambda img, t: wave_effect(img, t),
    frames=30,
    transition_frames=10
)

sequence.add_segment(
    effect_func=lambda img, t: melt_effect(img, t),
    frames=30,
    transition_frames=5
)

# Generate all frames
frames = sequence.generate_frames(base_image)
```

## Error Handling

The API includes validation for common errors:

```python
try:
    stretcher = PixelStretcher(max_stretch=1.5)  # Invalid: > 1
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    stretcher.create_animation('input.png', 'output.gif')  # Invalid: GIF not supported
except ValueError as e:
    print(f"Output format error: {e}")
```