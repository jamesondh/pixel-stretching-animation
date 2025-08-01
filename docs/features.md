# Pixel Stretching Animation - Features

## Core Features

### 1. Distortion Effects

#### Pivot-based Stretching
The original effect that stretches pixels around a configurable pivot point.

**Parameters:**
- `max_stretch` (0-1): Maximum stretch intensity
- `pivot` (center/top/bottom): The anchor point for stretching

**Use cases:**
- Classic pixel art distortion
- Symmetrical warping effects
- Controlled deformation animations

#### Wave Distortion
Creates sinusoidal wave patterns that flow through the image.

**Parameters:**
- `max_stretch` (0-1): Overall distortion intensity
- `wave_amplitude` (0-1): Height of the wave peaks
- `wave_frequency` (float): Number of wave cycles
- `wave_phase_shift` (radians): Initial wave offset
- `wave_phase_speed` (float): Animation speed

**Use cases:**
- Flowing, organic animations
- Water-like effects
- Dynamic, rhythmic distortions

#### Biased Stretching
Directional stretching that creates melting or floating effects.

**Parameters:**
- `max_stretch` (0-1): Maximum stretch intensity
- `stretch_bias` (-1 to 1): Direction preference
  - Negative values: Upward bias (floating effect)
  - Positive values: Downward bias (melting effect)
  - Zero: No bias (random direction)

**Use cases:**
- Melting animations
- Anti-gravity effects
- Directional pixel flow

### 2. Animation Modes

#### Standard Mode
Each frame applies increasing distortion to the original image.
- Smooth progression from original to maximum distortion
- Predictable, repeatable results
- Best for loop animations

#### Cumulative Mode
Each frame builds upon the previous frame's distortion.
- Creates cascading, evolving effects
- More chaotic and organic results
- Ideal for transformation sequences

#### Ping-Pong Mode
Animation plays forward then backward.
- Creates seamless loops
- Natural breathing/pulsing effects
- No jarring transitions

### 3. Animation Control

#### Stretch Curves
Control how the distortion progresses over time with different curve types.

**Curve Types:**
- **Linear**: Smooth, constant progression from start to end
- **Constant**: Maintains the same distortion level throughout
- **Ease In**: Starts slow, accelerates (creates anticipation)
- **Ease Out**: Starts fast, decelerates (creates settling effect)
- **Ease In-Out**: S-curve for smooth start and end (professional look)

**Parameters:**
- `stretch_curve`: Choose the progression type
- `start_stretch` (0-1): Initial distortion level (default: 0)
- `end_stretch` (0-1): Final distortion level (default: max_stretch)

**Use cases:**
- Constant stretch for static distortion effects
- Custom start/end for partial animations
- Reverse animations (high to low distortion)
- Professional easing for polished results

### 4. Rendering Options

#### Interpolation Methods
- **Nearest Neighbor**: Preserves hard pixel edges (best for pixel art)
- **Bilinear**: Smooth transitions between pixels (better for photos)

#### Upscaling (Now part of Post-Processing)
- Integer scaling (2x, 3x, 4x, etc.)
- Available as a post-processor for flexible pipeline placement
- Supports both nearest and bilinear interpolation
- Perfect for low-resolution pixel art

#### Temporal Smoothing
- Blends between frames for fluid motion
- Reduces jitter and sudden changes
- Value from 0 (no smoothing) to 1 (maximum smoothing)

### 5. Output Formats

Supported video formats:
- **MP4** (recommended): Best compatibility and compression
- **AVI**: Uncompressed option for maximum quality
- **MOV**: QuickTime format for Mac ecosystems

### 6. Post-Processing System

The post-processing system allows you to apply additional effects after the main animation generation. This separation enables:
- Processing existing video files
- Chaining multiple effects
- Non-destructive workflows

#### Sine Wave Post-Processor
Applies sinusoidal displacement to frames for ripple and wave effects.

**Parameters:**
- `axis`: Direction of the wave (vertical, horizontal, diagonal, or angle in degrees)
- `frequency`: Number of wave cycles across the image
- `amplitude`: Displacement strength (0-1, relative to image size)
- `phase`: Initial phase offset
- `speed`: Animation speed for moving waves
- `displacement_mode`: How pixels are displaced (translate, scale, or both)
- `edge_behavior`: How to handle pixels at edges (wrap, clamp, fade, mirror)
- `amplitude_curve`: How amplitude changes over time (constant, linear, ease_in, ease_out, ease_in_out)
- `start_amplitude` / `end_amplitude`: Amplitude animation range

**Use cases:**
- Water ripple effects
- Heat distortion
- Dream-like sequences
- Glitch effects

#### Upscale Post-Processor
Scales up video resolution with configurable interpolation.

**Parameters:**
- `scale_factor`: Integer scaling (2, 3, 4, etc.)
- `method`: Interpolation method (nearest or bilinear)

#### Post-Processor Chaining
Multiple post-processors can be chained together in sequence:
```yaml
post_processing:
  enabled: true
  processors:
    - type: sine_wave
      axis: horizontal
      frequency: 3.0
    - type: sine_wave
      axis: vertical
      frequency: 2.0
    - type: upscale
      scale_factor: 2
```

### 7. Advanced Features

#### Composite Effects
Combine multiple distortion effects with configurable weights.

#### Animation Sequences
Chain different effects together with smooth transitions.

#### Configuration Presets
Pre-configured settings for common use cases:
- `pixel_art`: Optimized for pixel art with upscaling
- `smooth_wave`: Flowing wave animation with smoothing
- `melting`: Downward melting effect
- `bouncy`: Ping-pong animation for bouncing effects

#### Batch Processing
Process multiple images with the same settings (via scripting).

#### Video Processing
Process existing videos without regeneration using the `process-video` command.

### 8. Shared Transform Architecture

The codebase uses a unified transform system that allows effects to be shared between generation and post-processing phases:

#### Shared Transform Utilities
- **SineWaveTransform**: Mathematical sine wave calculations
- **DisplacementCalculator**: Apply displacement fields to images
- **EdgeHandler**: Consistent edge behavior across all effects

#### Benefits
- Code reuse between distortion effects and post-processors
- Consistent behavior across different processing phases
- Easier to add new effects that work in both contexts
- Optimized algorithms used everywhere

#### Example: Sine Wave
The sine wave effect exists in both forms:
- **SineWaveDistortionEffect**: Applied during generation (column-by-column)
- **SineWavePostProcessor**: Applied after generation (full frames)
- Both use the same `SineWaveTransform` for calculations

## Feature Comparison

| Feature | Pivot | Wave | Bias | Sine Wave | Composite |
|---------|-------|------|------|-----------|-----------|
| Directional Control | Limited | Medium | High | High | Variable |
| Smoothness | Medium | High | Medium | Very High | Variable |
| Predictability | High | Medium | High | Very High | Low |
| CPU Usage | Low | Medium | Low | Low | High |
| Best For | Pixel Art | Organic | Melting | Precise | Complex |

## Performance Considerations

- **Frame Count**: More frames = smoother animation but larger file size
- **Resolution**: Higher resolution = more processing time
- **Upscaling**: Increases output resolution without quality loss
- **Effects**: Wave effect is more CPU-intensive than pivot or bias

## Tips for Best Results

1. **For Pixel Art**:
   - Use nearest neighbor interpolation
   - Apply 2x-4x upscaling
   - Keep max_stretch below 0.5
   - Try constant stretch for consistent glitch effects

2. **For Smooth Animations**:
   - Enable temporal smoothing (0.5-0.8)
   - Use higher FPS (30-60)
   - Consider bilinear interpolation

3. **For Dramatic Effects**:
   - Use cumulative mode
   - Higher max_stretch values (0.6-0.8)
   - Combine with bias for directional flow
   - Try reverse animations (start_stretch > end_stretch)

4. **For Professional Results**:
   - Use ease_in_out curve for smooth animations
   - Set start_stretch > 0 to avoid static first frame
   - Experiment with partial ranges (0.2 to 0.6)