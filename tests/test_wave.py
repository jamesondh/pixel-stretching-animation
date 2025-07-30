#!/usr/bin/env python3
from src.pixel_stretcher import PixelStretcher
from pathlib import Path

# Test wave-based distortion
input_image = Path("assets/sample_images/noise.png")

print("Creating wave-based animation with oscillation...")
stretcher = PixelStretcher(
    max_stretch=0.5,
    seed=42,
    cumulative=False,
    use_wave_distortion=True,
    wave_amplitude=0.15,
    wave_frequency=3.0,
    wave_phase_shift=0.0,
    wave_phase_speed=2.0,  # Wave completes 2 cycles during animation
    interpolation='bilinear'
)
stretcher.create_animation(
    input_path=input_image,
    output_path="output/test_wave.mp4",
    frames=30,
    fps=15
)

print("Creating cumulative wave-based animation with oscillation...")
stretcher_cumulative = PixelStretcher(
    max_stretch=0.3,
    seed=42,
    cumulative=True,
    use_wave_distortion=True,
    wave_amplitude=0.1,
    wave_frequency=2.0,
    wave_phase_shift=0.0,
    wave_phase_speed=1.5,  # Wave completes 1.5 cycles during animation
    interpolation='bilinear'
)
stretcher_cumulative.create_animation(
    input_path=input_image,
    output_path="output/test_wave_cumulative.mp4",
    frames=30,
    fps=15
)

print("Done! Check output/test_wave.mp4 and output/test_wave_cumulative.mp4")