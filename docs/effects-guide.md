# Effects Guide

This guide provides detailed information about each distortion effect, including visual examples, parameter tuning tips, and creative applications.

## Table of Contents

1. [Pivot Effect](#pivot-effect)
2. [Wave Effect](#wave-effect)
3. [Bias Effect](#bias-effect)
4. [Flowing Melt Effect](#flowing-melt-effect)
5. [Composite Effects](#composite-effects)
6. [Effect Combinations](#effect-combinations)
7. [Axis Control](#axis-control)

## Pivot Effect

The original and most versatile effect. Stretches pixels around a fixed pivot point.

### How It Works

The pivot effect treats each column of pixels independently, stretching them away from or toward a pivot point. The stretch amount is randomized per column, creating an organic distortion pattern.

### Parameters

- **max_stretch** (0-1): Controls the maximum displacement of pixels
  - 0.1-0.3: Subtle wobble
  - 0.4-0.6: Moderate distortion
  - 0.7-1.0: Extreme warping

- **pivot** (center/top/bottom): The anchor point
  - `center`: Stretches outward from middle (symmetric)
  - `top`: Stretches downward (hanging effect)
  - `bottom`: Stretches upward (rising effect)

### Visual Characteristics

- Random, noise-like pattern
- Each column moves independently
- Maintains recognizable shapes at low values
- Creates abstract patterns at high values

### Best Practices

```bash
# Subtle breathing effect
pixel-stretch animate input.png output.mp4 \
  --effect pivot \
  --pivot center \
  --max-stretch 0.3 \
  --temporal-smoothing 0.8

# Hanging/dripping effect
pixel-stretch animate input.png output.mp4 \
  --effect pivot \
  --pivot top \
  --max-stretch 0.5

# Rising smoke effect
pixel-stretch animate input.png output.mp4 \
  --effect pivot \
  --pivot bottom \
  --max-stretch 0.4 \
  --cumulative
```

### Creative Applications

1. **Pixel Art Enhancement**: Low stretch values (0.2-0.3) add life to static sprites
2. **Glitch Effects**: High values with no smoothing create digital artifacts
3. **Transition Effects**: Ramp up stretch over time for scene transitions
4. **Background Animation**: Subtle movement for menu backgrounds

## Wave Effect

Creates flowing, sinusoidal distortions that ripple through the image.

### How It Works

The wave effect applies a sine wave pattern to the vertical displacement of pixels. The wave can be animated to create flowing motion, and multiple parameters control the wave's shape and behavior.

### Parameters

- **max_stretch** (0-1): Overall intensity multiplier
- **wave_amplitude** (0-1): Height of wave peaks
  - 0.05-0.1: Gentle ripples
  - 0.15-0.3: Moderate waves
  - 0.4+: Dramatic undulation

- **wave_frequency** (float): Number of complete waves
  - 1-2: Long, smooth waves
  - 3-5: Medium frequency
  - 6+: Rapid oscillation

- **wave_phase_speed** (float): Animation speed
  - 0.5-1: Slow, hypnotic
  - 2-3: Normal flow
  - 4+: Rapid movement

### Visual Characteristics

- Smooth, predictable patterns
- Horizontal coherence (rows move together)
- Natural, organic motion
- Works well with both pixel art and photos

### Best Practices

```bash
# Gentle water ripple
pixel-stretch animate water.png ripple.mp4 \
  --effect wave \
  --wave-amplitude 0.08 \
  --wave-frequency 3 \
  --temporal-smoothing 0.6

# Flag waving effect
pixel-stretch animate flag.png waving.mp4 \
  --effect wave \
  --wave-amplitude 0.15 \
  --wave-frequency 2 \
  --max-stretch 0.4

# Psychedelic waves
pixel-stretch animate abstract.png trippy.mp4 \
  --effect wave \
  --wave-amplitude 0.3 \
  --wave-frequency 5 \
  --fps 60
```

### Creative Applications

1. **Water Simulation**: Low amplitude, high frequency for realistic water
2. **Cloth/Flag Animation**: Medium settings for fabric movement
3. **Dream Sequences**: High amplitude with slow speed
4. **UI Effects**: Subtle waves for menu items or backgrounds

## Bias Effect

Directional stretching that creates melting, dripping, or floating effects.

### How It Works

The bias effect applies preferential stretching in one direction. Unlike random stretching, most pixels move in the same direction with occasional counter-movement for organic feel.

### Parameters

- **max_stretch** (0-1): Maximum displacement amount
- **stretch_bias** (-1 to 1): Direction and strength
  - -1.0: Strong upward bias (anti-gravity)
  - -0.5: Moderate upward tendency
  - 0.0: No bias (random)
  - 0.5: Moderate downward tendency
  - 1.0: Strong downward bias (melting)

### Visual Characteristics

- Directional flow
- Cohesive movement pattern
- Natural gravity-like effects
- Excellent for transformation animations

### Best Practices

```bash
# Melting effect
pixel-stretch animate ice.png melting.mp4 \
  --effect bias \
  --stretch-bias 0.8 \
  --max-stretch 0.6 \
  --cumulative \
  --frames 90

# Anti-gravity float
pixel-stretch animate character.png floating.mp4 \
  --effect bias \
  --stretch-bias -0.7 \
  --max-stretch 0.4

# Subtle downward drift
pixel-stretch animate leaves.png falling.mp4 \
  --effect bias \
  --stretch-bias 0.5 \
  --max-stretch 0.3 \
  --temporal-smoothing 0.7
```

### Creative Applications

1. **Melting Objects**: Ice, candles, or any melting transformation
2. **Gravity Effects**: Falling particles, rain, or snow
3. **Magical Effects**: Floating objects, levitation
4. **Disintegration**: Objects breaking apart and falling

## Flowing Melt Effect

Advanced 2D displacement effect that combines vertical melting with horizontal wave-like drift, creating fluid dynamics similar to water flowing down a river.

### How It Works

The flowing melt effect applies two-dimensional displacement:
1. **Vertical melting**: Pixels are stretched downward with a bias, similar to the bias effect
2. **Horizontal drift**: Pixels flow left and right in wave patterns that vary with depth
3. **Depth-dependent flow**: Pixels that have melted further (lower in the image) flow more strongly

### Parameters

- **max_stretch** (0-1): Maximum vertical displacement
  - 0.1-0.3: Subtle flowing effect
  - 0.4-0.6: Moderate fluid motion
  - 0.7-1.0: Strong flowing distortion

- **melt_bias** (0-1): Strength of downward melting
  - 0.0-0.1: Minimal melting, mostly horizontal flow
  - 0.15-0.3: Balanced melting and flow
  - 0.5-1.0: Strong melting with flow

- **flow_amplitude** (0-1): Horizontal drift strength
  - 0.05-0.1: Subtle sideways movement
  - 0.15-0.25: Noticeable wave-like flow
  - 0.3-0.5: Strong horizontal displacement

- **flow_frequency** (1-10): Number of horizontal waves
  - 1-2: Large, slow waves
  - 3-5: Medium wave patterns
  - 6-10: Many small ripples

- **flow_speed** (0-2): Speed of wave movement
  - 0.1-0.3: Slow, calm flow
  - 0.5-0.8: Moderate flow speed
  - 1.0-2.0: Rapid flow

- **flow_variation** (0-1): Randomness in flow pattern
  - 0.0: Uniform waves
  - 0.2-0.4: Natural variation
  - 0.5-1.0: Chaotic, turbulent flow

- **edge_behavior** (wrap/clamp/fade): How pixels behave at edges
  - `wrap`: Pixels wrap around (continuous flow)
  - `clamp`: Pixels stick to edges
  - `fade`: Pixels disappear at edges

### Visual Characteristics

- Fluid, water-like motion
- Natural flow patterns that follow gravity
- Depth-dependent movement (stronger flow at bottom)
- Combines melting with lateral displacement
- Creates organic, flowing animations

### Best Practices

```bash
# Basic flowing melt
pixel-stretch animate lava.png flowing_lava.mp4 \
  --config configs/example_flowing_melt.yaml

# Dramatic water flow
pixel-stretch animate waterfall.png water_flow.mp4 \
  --config configs/example_flowing_melt_dramatic.yaml

# Custom flowing effect via CLI
pixel-stretch animate ice.png melting_ice.mp4 \
  --effect flowing_melt \
  --max-stretch 0.35 \
  --melt-bias 0.2 \
  --flow-amplitude 0.2 \
  --flow-frequency 2.5 \
  --flow-speed 0.6 \
  --edge-behavior wrap \
  --cumulative \
  --frames 200
```

### Creative Applications

1. **Liquid Simulations**: Water, lava, honey, or any viscous fluid
2. **Melting with Movement**: Ice melting and flowing away
3. **Digital Rain Effects**: Matrix-style cascading characters with drift
4. **Magical Dissolve**: Objects dissolving into flowing particles
5. **Environmental Effects**: Rain on windows, flowing sand, or snow melt
6. **Glitch Art**: Digital corruption with directional flow

### Performance Considerations

The flowing melt effect is computationally more intensive than single-axis effects due to:
- 2D displacement calculations
- Per-pixel horizontal displacement
- Bilinear interpolation for smooth flow

For better performance:
- Use lower resolution images during testing
- Reduce upscale factor
- Limit frame count for previews
- Consider using `edge_behavior: clamp` for faster processing

### Configuration Examples

See the included example configurations:
- `configs/example_flowing_melt.yaml`: Balanced flowing melt effect
- `configs/example_flowing_melt_dramatic.yaml`: Strong flow with fade edges

## Composite Effects

Combine multiple effects for complex animations.

### Implementation

Currently requires Python API:

```python
from src.distortion_effects import CompositeEffect, WaveDistortionEffect, BiasedStretchEffect

# Create individual effects
wave = WaveDistortionEffect(
    max_stretch=0.3,
    wave_amplitude=0.1,
    wave_frequency=2
)

melt = BiasedStretchEffect(
    max_stretch=0.4,
    stretch_bias=0.7
)

# Combine with weights
composite = CompositeEffect(
    effects=[wave, melt],
    weights=[0.6, 0.4]  # 60% wave, 40% melt
)
```

### Advanced Parameters

#### Stretch Curves

The `stretch_curves` parameter allows you to control how each effect's intensity changes over time:

```python
# Create composite with different timing curves
composite = CompositeEffect(
    effects=[wave, melt],
    weights=[0.6, 0.4],
    stretch_curves=['ease_in', 'constant']  # Wave eases in, melt stays constant
)
```

Available curve types:
- **constant**: Effect maintains full intensity throughout animation
- **linear**: Effect scales linearly with animation progress (default)
- **ease_in**: Effect starts slow and accelerates (quadratic)
- **ease_out**: Effect starts fast and decelerates
- **ease_in_out**: Effect starts and ends slowly with faster middle

This is particularly useful for creating complex animations where different effects need different timing:

```python
# Example: Wave that fades in while melting stays constant
ripple = WaveDistortionEffect(max_stretch=0.2, wave_frequency=3)
melt = BiasedStretchEffect(max_stretch=0.5, stretch_bias=0.9)

composite = CompositeEffect(
    effects=[ripple, melt],
    weights=[0.3, 0.7],
    stretch_curves=['ease_in_out', 'constant']
)

# The ripple effect will gradually fade in and out
# while the melting effect remains at full strength
```

### Use Cases

1. **Ocean Waves + Gravity**: Realistic water motion
2. **Wave + Random**: Organic, less predictable patterns
3. **Multiple Waves**: Complex interference patterns
4. **Bias + Pivot**: Directional flow with random elements

## Effect Combinations

### Sequential Effects

Create animations that transition between effects:

```python
# Start with gentle wave, transition to melting
sequence = AnimationSequence()
sequence.add_segment(wave_effect, frames=30, transition_frames=10)
sequence.add_segment(melt_effect, frames=30)
```

### Layered Processing

Apply effects in stages:

1. First pass: Subtle wave for base movement
2. Second pass: Add directional bias
3. Third pass: Fine random distortion

### Parameter Animation

Vary parameters over time for dynamic effects:

```python
# Increasing intensity
for i in range(frames):
    stretch = i / frames * max_stretch
    frame = warp_with_stretch(image, stretch)
```

## Effect Selection Guide

| Effect | Best For | Avoid When |
|--------|----------|------------|
| Pivot | General distortion, pixel art | Need smooth motion |
| Wave | Flowing animations, water | Need random chaos |
| Bias | Melting, gravity effects | Need symmetric distortion |

## Performance Tips

1. **Preview First**: Test with low resolution
2. **Start Small**: Begin with subtle parameters
3. **Batch Process**: Reuse effect configurations
4. **Cache Results**: Save intermediate frames

## Troubleshooting Effects

### Too Chaotic
- Reduce max_stretch
- Increase temporal_smoothing
- Use fewer frames

### Too Predictable
- Add slight randomization
- Combine multiple effects
- Vary parameters over time

### Performance Issues
- Reduce image resolution
- Lower frame count
- Disable upscaling during preview

## Axis Control

Control the direction of stretching with the `--axis` parameter. All effects support axis transformation.

### How It Works

By default, all effects apply vertical stretching (up/down). The axis parameter allows you to:
- Apply effects horizontally (left/right)
- Apply effects at any arbitrary angle
- Create unique directional distortions

### Axis Options

- **vertical** (default): Standard up/down stretching
- **horizontal**: Left/right stretching
- **angle** (degrees): Any angle from -180 to 180

### Examples

```bash
# Horizontal wave effect (left-right motion)
pixel-stretch animate ocean.png waves.mp4 \
  --effect wave \
  --axis horizontal \
  --wave-amplitude 0.1

# Diagonal melting at 45 degrees
pixel-stretch animate ice.png diagonal_melt.mp4 \
  --effect bias \
  --axis 45 \
  --stretch-bias 0.8 \
  --cumulative

# Horizontal pivot stretch (sideways wobble)
pixel-stretch animate character.png wobble.mp4 \
  --effect pivot \
  --axis horizontal \
  --pivot center

# Rotated wave at -30 degrees
pixel-stretch animate fabric.png rotated_wave.mp4 \
  --effect wave \
  --axis -30 \
  --wave-frequency 3
```

### Creative Applications

1. **Horizontal Effects**
   - Wind effects on flags or hair
   - Sideways melting or morphing
   - Horizontal wave patterns for water surfaces

2. **Diagonal Effects**
   - Rain or snow falling at an angle
   - Slanted melting effects
   - Dynamic motion blur alternatives

3. **Custom Angles**
   - Match the angle of objects in the scene
   - Create spiral-like distortions
   - Simulate perspective-based stretching

### Technical Notes

- Horizontal mode transposes the image internally for processing
- Arbitrary angles use rotation transforms with proper anti-aliasing
- Performance is similar across all axis modes
- All other effect parameters work normally with axis transformations

### Best Practices

1. **Match Content**: Align the axis with natural directions in your image
2. **Combine with Effects**: Different axes work better with different effects
   - Horizontal + Wave: Great for flags and water
   - 45° + Bias: Interesting diagonal melting
   - -45° + Pivot: Dynamic corner stretching
3. **Experiment**: Non-standard angles can create unique, unexpected results