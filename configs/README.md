# Configuration Files

This directory contains example configuration files for the pixel stretching animation tool.

## Available Examples

### example_wave.yaml
A wave-based animation with smooth, flowing distortions. Features:
- Bilinear interpolation for smooth results
- Temporal smoothing for fluid motion
- Moderate wave parameters for organic movement

### example_melting.yaml
Creates a melting/dripping effect using biased stretching. Features:
- Strong downward bias (0.85) for melting appearance
- Cumulative mode for progressive deformation
- Nearest neighbor interpolation for pixel art

### example_pixel_art.json
Optimized settings for pixel art animations. Features:
- 4x upscaling with nearest neighbor
- Center pivot for balanced distortion
- Lower frame rate for retro feel

## Usage

### With CLI
```bash
pixel-stretch animate input.png output.mp4 --config configs/example_wave.yaml
```

### With Python API
```python
from src.config import PixelStretchConfig
from src.pixel_stretcher import PixelStretcher

# Load configuration
config = PixelStretchConfig.from_file('configs/example_wave.yaml')

# Create stretcher from config
stretcher = PixelStretcher(
    max_stretch=config.effect.max_stretch,
    # ... other parameters from config
)
```

## Creating Custom Configurations

1. Copy one of the example files
2. Modify parameters as needed
3. Validate with: `pixel-stretch animate --config your_config.yaml --dry-run`

## Configuration Schema

```yaml
effect:
  type: [pivot|wave|bias]  # Effect type
  max_stretch: 0.0-1.0     # Maximum stretch intensity
  # Additional parameters depend on effect type

animation:
  frames: integer          # Number of frames
  fps: integer            # Frames per second
  interpolation: [nearest|bilinear]
  temporal_smoothing: 0.0-1.0
  upscale: integer        # Upscale factor
  cumulative: boolean     # Cumulative distortion
  frame_generator: [standard|cumulative|pingpong]

output:
  format: [mp4|avi|mov]   # Output format
  codec: string           # Video codec (optional)
  quality: integer        # Quality setting (optional)
```