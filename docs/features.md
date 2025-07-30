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

### 3. Rendering Options

#### Interpolation Methods
- **Nearest Neighbor**: Preserves hard pixel edges (best for pixel art)
- **Bilinear**: Smooth transitions between pixels (better for photos)

#### Upscaling
- Integer scaling (2x, 3x, 4x, etc.)
- Uses nearest neighbor to maintain pixel integrity
- Perfect for low-resolution pixel art

#### Temporal Smoothing
- Blends between frames for fluid motion
- Reduces jitter and sudden changes
- Value from 0 (no smoothing) to 1 (maximum smoothing)

### 4. Output Formats

Supported video formats:
- **MP4** (recommended): Best compatibility and compression
- **AVI**: Uncompressed option for maximum quality
- **MOV**: QuickTime format for Mac ecosystems

### 5. Advanced Features

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

## Feature Comparison

| Feature | Pivot | Wave | Bias | Composite |
|---------|-------|------|------|-----------|
| Directional Control | Limited | Medium | High | Variable |
| Smoothness | Medium | High | Medium | Variable |
| Predictability | High | Medium | High | Low |
| CPU Usage | Low | Medium | Low | High |
| Best For | Pixel Art | Organic | Melting | Complex |

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

2. **For Smooth Animations**:
   - Enable temporal smoothing (0.5-0.8)
   - Use higher FPS (30-60)
   - Consider bilinear interpolation

3. **For Dramatic Effects**:
   - Use cumulative mode
   - Higher max_stretch values (0.6-0.8)
   - Combine with bias for directional flow