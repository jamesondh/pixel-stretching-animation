# Pixel Stretching Animation

A modular Python library for creating mesmerizing pixel-stretching animations from static images. This tool applies various distortion effects to create dynamic animations with highly customizable parameters.

## Features

- **Multiple distortion effects**:
  - Pivot-based stretching (original effect)
  - Wave distortion with animated phase
  - Directional bias (melting/floating effects)
  - Composite effects (combine multiple distortions)
  - Horizontal and arbitrary angle stretching
- **Advanced animation modes**:
  - Standard (increasing distortion)
  - Cumulative (building on previous frames)
  - Ping-pong (forward and backward)
- **Rendering options**:
  - Nearest neighbor or bilinear interpolation
  - Integer upscaling for pixel art
  - Temporal smoothing for fluid motion
- **Modern architecture**:
  - Modular effect system
  - Configuration file support (YAML/JSON)
  - Feature registry for extensibility
  - Comprehensive CLI with subcommands
- **Output formats**: MP4, AVI, or MOV (video only)

## Installation

```bash
# Clone the repository
git clone https://github.com/jamesondh/pixel-stretching-animation.git
cd pixel-stretching-animation

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Basic animation
pixel-stretch animate input.png output.mp4

# Using presets
pixel-stretch animate input.png output.mp4 --preset pixel_art

# Wave effect with custom parameters
pixel-stretch animate input.png output.mp4 --effect wave --wave-amplitude 0.2

# Melting effect
pixel-stretch animate input.png output.mp4 --effect bias --stretch-bias 0.8 --cumulative

# Horizontal stretching
pixel-stretch animate input.png output.mp4 --effect pivot --axis horizontal

# Diagonal stretching at 45 degrees
pixel-stretch animate input.png output.mp4 --effect wave --axis 45

# Using configuration file
pixel-stretch animate input.png output.mp4 --config configs/example_wave.yaml

# List available effects and presets
pixel-stretch effects
pixel-stretch presets
```

### Python API

```python
from src.pixel_stretcher import PixelStretcher

# Create a stretcher instance
stretcher = PixelStretcher(
    max_stretch=0.5,      # Maximum stretch factor (0-1)
    pivot='center',       # Pivot point: 'center', 'top', or 'bottom'
    interpolation='nearest',  # 'nearest' for pixel art, 'bilinear' for smooth
    temporal_smoothing=0.3    # Smooth transitions between frames
)

# Generate animation
stretcher.create_animation(
    'input.png',
    'output.mp4',
    frames=60,
    fps=30
)
```

## Using Configuration Files

```yaml
# config.yaml
effect:
  type: wave
  max_stretch: 0.4
  wave_amplitude: 0.15
  wave_frequency: 2.5

animation:
  frames: 60
  fps: 30
  temporal_smoothing: 0.7
  upscale: 2
```

```bash
pixel-stretch animate input.png output.mp4 --config config.yaml
```

## Examples

The `examples/` directory contains demonstration scripts:

- `simple_stretch.py`: Basic usage examples
- `demo_all_features.py`: Comprehensive feature demonstration
- `advanced_effects.py`: Custom effect implementations
- `biased_stretch.py`: Directional stretching examples
- `upscale_demo.py`: Upscaling demonstrations

Run the comprehensive demo:

```bash
python examples/demo_all_features.py
```

This creates animations demonstrating all effects, modes, and options.

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Features Guide](docs/features.md) - Detailed feature descriptions
- [API Reference](docs/api-reference.md) - Complete API documentation
- [CLI Reference](docs/cli-reference.md) - Command-line interface guide
- [Effects Guide](docs/effects-guide.md) - In-depth effect documentation

## Project Structure

```
pixel-stretching-animation/
├── src/
│   ├── pixel_stretcher.py      # Main animation class
│   ├── distortion_effects.py   # Modular effect system
│   ├── animation.py            # Animation engine
│   ├── config.py               # Configuration management
│   ├── cli.py                  # CLI with subcommands
│   └── registry.py             # Feature registry
├── docs/                       # Comprehensive documentation
├── configs/                    # Example configuration files
├── examples/                   # Usage examples
└── tests/                      # Unit tests
```

## Available Presets

- **pixel_art**: Optimized for pixel art with 4x upscaling
- **smooth_wave**: Flowing wave animation with smoothing
- **melting**: Downward melting effect with cumulative mode
- **bouncy**: Ping-pong animation for bouncing effects

Use presets with: `pixel-stretch animate input.png output.mp4 --preset <name>`

## Advanced Usage

### Custom Effects

```python
from src.distortion_effects import DistortionEffect
import numpy as np

class CustomEffect(DistortionEffect):
    def generate_factors(self, width, frame=0, total_frames=60):
        # Your custom factor generation
        return np.random.rand(width) * self.max_stretch

    def warp_column(self, column, factor, column_index, total_width, height):
        # Your custom warping logic
        return warped_column
```

### Batch Processing

```python
from pathlib import Path
from src.pixel_stretcher import PixelStretcher
from src.config import get_preset

# Process multiple images with a preset
for img_path in Path('input_dir').glob('*.png'):
    config = get_preset('pixel_art')
    stretcher = PixelStretcher(**config.to_dict())
    stretcher.create_animation(
        img_path,
        f'output/{img_path.stem}_animated.mp4'
    )
```

## Requirements

- Python 3.8+
- NumPy
- Pillow
- imageio + imageio-ffmpeg
- scipy
- noise
- click
- PyYAML (optional, for YAML configs)

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## License

MIT License - feel free to use in your projects!

## Contributing

Contributions are welcome! The modular architecture makes it easy to:

- Add new distortion effects
- Create custom frame generators
- Implement new animation modes

See the documentation for development guidelines.
