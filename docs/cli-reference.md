# CLI Reference

The Pixel Stretching Animation tool provides a comprehensive command-line interface with multiple subcommands.

## Installation

After installing the package, the `pixel-stretch` command becomes available:

```bash
pip install -e .
pixel-stretch --version
```

## Main Commands

### animate

Create a pixel-stretching animation from an input image.

```bash
pixel-stretch animate INPUT_PATH OUTPUT_PATH [OPTIONS]
```

#### Arguments
- **INPUT_PATH**: Path to the input image (PNG, JPEG, etc.)
- **OUTPUT_PATH**: Path for the output video (must be .mp4, .avi, or .mov)

#### Options

**Basic Options:**
- `--frames, -f INTEGER`: Number of frames to generate (default: 60)
- `--fps INTEGER`: Frames per second (default: 30)
- `--max-stretch, -s FLOAT`: Maximum stretch factor 0-1 (default: 0.5)
- `--seed INTEGER`: Random seed for reproducibility

**Effect Options:**
- `--effect [pivot|wave|bias]`: Effect type to use (default: pivot)
- `--pivot [center|top|bottom]`: Pivot point for stretching (default: center)
- `--stretch-bias, -b FLOAT`: Bias stretching direction -1 to 1 (default: 0.0)

**Wave Effect Options:**
- `--wave-amplitude FLOAT`: Wave amplitude for wave effect (default: 0.1)
- `--wave-frequency FLOAT`: Wave frequency for wave effect (default: 2.0)

**Rendering Options:**
- `--interpolation [nearest|bilinear]`: Interpolation method (default: nearest)
- `--temporal-smoothing, -t FLOAT`: Temporal smoothing factor 0-1 (default: 0.0)
- `--upscale, -u INTEGER`: Upscale factor (default: 1)
- `--cumulative, -C`: Apply distortion cumulatively to each frame

**Configuration Options:**
- `--preset, -p [pixel_art|smooth_wave|melting|bouncy]`: Use a preset configuration
- `--config, -c PATH`: Load configuration from file

#### Examples

Basic usage:
```bash
pixel-stretch animate input.png output.mp4
```

With custom parameters:
```bash
pixel-stretch animate input.png output.mp4 --frames 120 --fps 60 --max-stretch 0.7
```

Using wave effect:
```bash
pixel-stretch animate input.png output.mp4 --effect wave --wave-amplitude 0.2
```

Melting effect with bias:
```bash
pixel-stretch animate input.png output.mp4 --effect bias --stretch-bias 0.8 --cumulative
```

Using a preset:
```bash
pixel-stretch animate input.png output.mp4 --preset pixel_art
```

Loading from config file:
```bash
pixel-stretch animate input.png output.mp4 --config my_animation.yaml
```

### effects

List available effects and their parameters.

```bash
pixel-stretch effects
```

Output:
```
Available effects:

  pivot: Original pivot-based stretching
    Parameters: max_stretch, pivot (center/top/bottom)

  wave: Wave-based distortion with animated phase
    Parameters: max_stretch, wave_amplitude, wave_frequency, wave_phase_shift, wave_phase_speed

  bias: Directional stretching with configurable bias
    Parameters: max_stretch, stretch_bias (-1 to 1)
```

### presets

List available presets with their configurations.

```bash
pixel-stretch presets
```

Output:
```
Available presets:

  pixel_art:
    Effect: pivot
    Max stretch: 0.4
    Frames: 60 @ 30 fps
    Features: 4x upscale

  smooth_wave:
    Effect: wave
    Max stretch: 0.3
    Frames: 60 @ 60 fps
    Features: smoothed

  melting:
    Effect: bias
    Max stretch: 0.6
    Frames: 90 @ 30 fps
    Features: cumulative

  bouncy:
    Effect: pivot
    Max stretch: 0.7
    Frames: 60 @ 30 fps
    Features: none
```

### create-config

Create a configuration file template.

```bash
pixel-stretch create-config OUTPUT_PATH [OPTIONS]
```

#### Arguments
- **OUTPUT_PATH**: Path for the configuration file

#### Options
- `--preset, -p [pixel_art|smooth_wave|melting|bouncy]`: Base configuration on a preset
- `--format, -f [json|yaml]`: Output format (default: yaml)

#### Examples

Create default configuration:
```bash
pixel-stretch create-config my_config.yaml
```

Create config based on preset:
```bash
pixel-stretch create-config my_config.yaml --preset smooth_wave
```

Create JSON config:
```bash
pixel-stretch create-config my_config.json --format json
```

### preview

Generate a quick preview of an effect.

```bash
pixel-stretch preview INPUT_PATH [OPTIONS]
```

#### Arguments
- **INPUT_PATH**: Path to the input image

#### Options
- `--size, -s INTEGER`: Preview size in pixels (default: 64)
- `--effect [pivot|wave|bias]`: Effect type to preview (default: pivot)
- `--max-stretch FLOAT`: Maximum stretch factor (default: 0.5)

#### Examples

Quick preview:
```bash
pixel-stretch preview input.png
```

Preview with specific effect:
```bash
pixel-stretch preview input.png --effect wave --size 128
```

## Configuration Files

Configuration files can be in YAML or JSON format.

### YAML Example

```yaml
effect:
  type: wave
  max_stretch: 0.5
  wave_amplitude: 0.15
  wave_frequency: 2.5
  seed: 42

animation:
  frames: 90
  fps: 30
  interpolation: bilinear
  temporal_smoothing: 0.7
  upscale: 2
  cumulative: false

output:
  format: mp4
  codec: libx264
```

### JSON Example

```json
{
  "effect": {
    "type": "bias",
    "max_stretch": 0.6,
    "stretch_bias": 0.8
  },
  "animation": {
    "frames": 60,
    "fps": 30,
    "cumulative": true
  },
  "output": {
    "format": "mp4"
  }
}
```

## Advanced Usage

### Batch Processing with Shell

Process multiple files:
```bash
for img in *.png; do
    pixel-stretch animate "$img" "output/${img%.png}.mp4" --preset pixel_art
done
```

### Combining with ffmpeg

Add audio to animation:
```bash
pixel-stretch animate input.png temp.mp4
ffmpeg -i temp.mp4 -i audio.mp3 -c:v copy -c:a aac final.mp4
```

### Parameter Exploration

Test different stretch values:
```bash
for stretch in 0.3 0.5 0.7; do
    pixel-stretch animate input.png "output_${stretch}.mp4" --max-stretch $stretch
done
```

## Tips and Best Practices

1. **Start with presets**: Use `--preset` to get good default settings
2. **Preview first**: Use the `preview` command to test effects quickly
3. **Save configurations**: Use `create-config` to save successful parameter combinations
4. **Batch processing**: Use shell scripts for processing multiple images
5. **Upscale pixel art**: Always use `--upscale` with pixel art for better quality

## Troubleshooting

### Common Issues

**Output format error:**
```
Error: Output must be .mp4, .avi, or .mov
```
Solution: Ensure output file has correct extension

**Invalid parameter range:**
```
Error: max_stretch must be between 0 and 1
```
Solution: Check parameter documentation for valid ranges

**Missing dependencies:**
```
Error: imageio-ffmpeg not installed
```
Solution: Install with `pip install imageio-ffmpeg`