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
- **upscale** (int): Upscaling factor (deprecated - use post-processing instead)
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
    loop=0,
    post_processor=None,
    codec='libx264',
    quality=None
)
```

Creates an animation from a static image.

**Parameters:**
- **input_path** (str/Path): Path to input image
- **output_path** (str/Path): Path for output video
- **frames** (int): Number of frames to generate
- **fps** (int): Frames per second
- **loop** (int): Number of loops (unused for video formats)
- **post_processor** (PostProcessor, optional): Post-processor to apply
- **codec** (str): Video codec to use
- **quality** (int, optional): Video quality (codec-specific)

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
    weights=[0.7, 0.3],  # 70% wave, 30% bias
    stretch_curves=['linear', 'constant']  # Optional: control timing curves
)
```

**Parameters:**
- `effects`: List of DistortionEffect instances to combine
- `weights`: Optional list of weights for each effect (normalized automatically)
- `stretch_curves`: Optional list of timing curves for each effect
  - `'constant'`: Effect maintains full intensity throughout
  - `'linear'`: Effect scales with animation progress (default)
  - `'ease_in'`: Effect starts slow and accelerates
  - `'ease_out'`: Effect starts fast and decelerates
  - `'ease_in_out'`: Effect starts and ends slowly

**Methods:**

`generate_factors(width, frame, total_frames)`: Generate combined distortion factors

`generate_factors_with_scale(width, frame, total_frames, stretch_scale, end_stretch)`: 
Generate factors with per-effect stretch curves. This method allows fine-grained control over how each effect's intensity changes throughout the animation.

```python
# Example using generate_factors_with_scale
factors = composite.generate_factors_with_scale(
    width=image.width,
    frame=current_frame,
    total_frames=60,
    stretch_scale=0.5,  # Overall animation progress
    end_stretch=1.0     # Maximum stretch value
)
```

#### FlowingMeltEffect

Creates a flowing melt effect with 2D displacement - vertical melting combined with horizontal wave-like drift.

```python
from src.distortion_effects import FlowingMeltEffect

effect = FlowingMeltEffect(
    max_stretch=0.3,
    melt_bias=0.15,      # Downward melting strength (0-1)
    flow_amplitude=0.15,  # Horizontal drift strength (0-1)
    flow_frequency=3.0,   # Number of horizontal waves
    flow_speed=0.5,       # Speed of wave movement
    flow_variation=0.3,   # Randomness in flow (0-1)
    edge_behavior='wrap', # 'wrap', 'clamp', or 'fade'
    seed=42
)
```

**Parameters:**
- `max_stretch`: Maximum vertical stretch factor (0-1)
- `melt_bias`: Strength of downward melting (0-1)
- `flow_amplitude`: Horizontal drift strength (0-1)
- `flow_frequency`: Number of horizontal wave patterns
- `flow_speed`: Speed of horizontal wave movement
- `flow_variation`: Random variation in flow pattern (0-1)
- `edge_behavior`: How pixels behave at edges
  - `'wrap'`: Pixels wrap around edges
  - `'clamp'`: Pixels stick to edges
  - `'fade'`: Pixels fade out at edges

**Special Methods:**

`calculate_horizontal_displacement(row, column, width, height, time)`: Calculate horizontal displacement for a pixel at given position and time.

**Usage Example:**

```python
from src.pixel_stretcher import PixelStretcher
from src.distortion_effects import FlowingMeltEffect

# Create flowing melt effect
flowing_melt = FlowingMeltEffect(
    max_stretch=0.4,
    melt_bias=0.2,
    flow_amplitude=0.2,
    flow_frequency=2.0,
    flow_speed=0.6,
    edge_behavior='fade'
)

# Use with PixelStretcher
stretcher = PixelStretcher(effect=flowing_melt, cumulative=True)
stretcher.create_animation('input.png', 'output.mp4', frames=200, fps=30)
```

#### SineWaveDistortionEffect

Creates smooth sine wave distortions using shared transform utilities.

```python
from src.distortion_effects import SineWaveDistortionEffect

effect = SineWaveDistortionEffect(
    max_stretch=0.5,
    frequency=3.0,      # Number of wave cycles
    amplitude=0.15,     # Wave amplitude relative to max_stretch
    phase=0.0,          # Initial phase offset
    speed=0.2,          # Animation speed
    axis='vertical',    # Wave direction
    seed=42
)
```

**Key Features:**
- Uses the same `SineWaveTransform` as the post-processor
- Supports all axis options (vertical, horizontal, diagonal, angles)
- Smooth mathematical waves without randomness
- Efficient column-based processing

**Usage Example:**

```python
from src.pixel_stretcher import PixelStretcher
from src.distortion_effects import SineWaveDistortionEffect

# Create matching generation and post-processing effects
sine_effect = SineWaveDistortionEffect(
    max_stretch=0.4,
    frequency=3.0,
    axis='vertical'
)

stretcher = PixelStretcher(effect=sine_effect)
stretcher.create_animation('input.png', 'output.mp4')
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

## Post-Processing Classes

### PostProcessor (Base Class)

Abstract base class for all post-processing effects.

```python
from src.post_processors import PostProcessor
from abc import abstractmethod

class PostProcessor(ABC):
    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_index: int, 
                     total_frames: int, **kwargs) -> np.ndarray:
        """Process a single frame."""
        pass
```

### SineWavePostProcessor

Applies sinusoidal displacement to frames.

```python
from src.post_processors import SineWavePostProcessor

processor = SineWavePostProcessor(
    axis='vertical',
    frequency=3.0,
    amplitude=0.05,
    phase=0.0,
    speed=0.5,
    displacement_mode='translate',
    edge_behavior='wrap',
    amplitude_curve='constant',
    start_amplitude=None,
    end_amplitude=None,
    interpolation='bilinear',
    preserve_palette=False
)
```

#### Parameters

- **axis** (str/float): Wave direction ('vertical', 'horizontal', 'diagonal', or angle in degrees)
- **frequency** (float): Number of wave cycles across the image
- **amplitude** (float, 0-1): Displacement strength relative to image size
- **phase** (float): Initial phase offset in radians
- **speed** (float): Animation speed for moving waves
- **displacement_mode** (str): How pixels are displaced ('translate', 'scale', 'both')
- **edge_behavior** (str): Edge handling ('wrap', 'clamp', 'fade', 'mirror')
- **amplitude_curve** (str): Amplitude animation curve
- **start_amplitude** (float, optional): Starting amplitude for animation
- **end_amplitude** (float, optional): Ending amplitude for animation
- **interpolation** (str): Interpolation method ('nearest' for sharp pixels, 'bilinear' for smooth)
- **preserve_palette** (bool): When True with nearest interpolation, preserves exact original colors

### UpscalePostProcessor

Scales up video resolution.

```python
from src.post_processors import UpscalePostProcessor

upscaler = UpscalePostProcessor(
    scale_factor=2,
    method='nearest'
)
```

#### Parameters

- **scale_factor** (int): Integer scaling factor (2, 3, 4, etc.)
- **method** (str): Interpolation method ('nearest' or 'bilinear')

### PostProcessorChain

Chains multiple post-processors together.

```python
from src.post_processors import PostProcessorChain, SineWavePostProcessor, UpscalePostProcessor

chain = PostProcessorChain([
    SineWavePostProcessor(axis='horizontal', frequency=3.0),
    SineWavePostProcessor(axis='vertical', frequency=2.0),
    UpscalePostProcessor(scale_factor=2)
])

# Process a frame
processed = chain.process_frame(frame, 0, 60)
```

### VideoProcessor

Processes existing video files.

```python
from src.video_processor import VideoProcessor
from src.post_processors import SineWavePostProcessor

processor = VideoProcessor()
post_processor = SineWavePostProcessor(frequency=4.0, amplitude=0.08)

processor.process_video(
    input_path='input.mp4',
    output_path='output.mp4',
    post_processor=post_processor,
    fps=30,
    codec='libx264',
    show_progress=True
)
```

#### Methods

##### process_video()
Process a video file with post-processing effects.

**Parameters:**
- **input_path** (str/Path): Input video path
- **output_path** (str/Path): Output video path
- **post_processor** (PostProcessor): Post-processor to apply
- **fps** (int, optional): Output FPS (defaults to input FPS)
- **codec** (str): Video codec
- **quality** (int, optional): Video quality
- **show_progress** (bool): Show progress bar

##### process_video_stream()
Process video in streaming mode for lower memory usage.

**Parameters:**
Same as `process_video()` plus:
- **batch_size** (int): Number of frames to process at once

### PostProcessorFactory

Factory for creating post-processors from configuration.

```python
from src.post_processing_factory import PostProcessorFactory

# Create from configuration
config = [
    {
        'type': 'sine_wave',
        'axis': 'horizontal',
        'frequency': 3.0
    },
    {
        'type': 'upscale',
        'scale_factor': 2
    }
]

processor = PostProcessorFactory.create_chain(config)
```

## Shared Transform Utilities

### SineWaveTransform

Shared sine wave transformation logic used by both distortion effects and post-processors.

```python
from src.transforms import SineWaveTransform

transform = SineWaveTransform(
    frequency=3.0,
    amplitude=0.05,
    phase=0.0,
    speed=0.5,
    axis='vertical'  # or 'horizontal', 'diagonal', or angle in degrees
)
```

#### Methods

- `calculate_displacement(x, y, time, dimension_size)`: Calculate 2D displacement
- `calculate_column_displacement(column_index, height, width, time)`: For distortion effects
- `calculate_frame_displacement(width, height, time)`: For post-processors
- `get_amplitude_at_time(time, curve, start_amplitude, end_amplitude)`: Amplitude curves

### DisplacementCalculator

Utilities for applying displacement fields to images.

```python
from src.transforms import DisplacementCalculator

# Apply displacement to entire image
result = DisplacementCalculator.apply_displacement(
    image, dx, dy,
    mode='translate',  # or 'scale', 'both'
    order=1  # interpolation order
)

# Apply to single column (for distortion effects)
warped_column = DisplacementCalculator.apply_column_displacement(
    column, displacement,
    interpolation='bilinear'
)
```

### EdgeHandler

Handle edge behavior for transformations.

```python
from src.transforms import EdgeHandler

# Apply edge behavior to coordinates
new_coords, alpha_mask = EdgeHandler.apply_edge_behavior(
    coords, dimension_size,
    behavior='wrap',  # or 'clamp', 'fade', 'mirror'
    fade_width=0.1
)

# 2D edge handling
new_x, new_y, alpha = EdgeHandler.apply_2d_edge_behavior(
    x_coords, y_coords, width, height,
    behavior='fade'
)
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
from src.post_processors import UpscalePostProcessor

# Process all PNG files in a directory
input_dir = Path('input_images')
output_dir = Path('output_videos')
output_dir.mkdir(exist_ok=True)

stretcher = PixelStretcher(max_stretch=0.5)
upscaler = UpscalePostProcessor(scale_factor=2)

for img_path in input_dir.glob('*.png'):
    output_path = output_dir / f"{img_path.stem}_stretched.mp4"
    stretcher.create_animation(img_path, output_path, post_processor=upscaler)
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