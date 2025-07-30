#!/usr/bin/env python3
from src.pixel_stretcher import PixelStretcher
from pathlib import Path

# Test both modes
input_image = Path("assets/sample_images/noise.png")

# Non-cumulative mode (default)
print("Creating non-cumulative animation...")
stretcher_normal = PixelStretcher(
    max_stretch=0.5,
    seed=42,
    upscale=2
)
stretcher_normal.create_animation(
    input_path=input_image,
    output_path="output/test_normal.mp4",
    frames=30,
    fps=15
)

# Cumulative mode
print("Creating cumulative animation...")
stretcher_cumulative = PixelStretcher(
    max_stretch=0.5,
    seed=42,
    upscale=2,
    cumulative=True
)
stretcher_cumulative.create_animation(
    input_path=input_image,
    output_path="output/test_cumulative.mp4",
    frames=30,
    fps=15
)

print("Done! Check output/test_normal.mp4 and output/test_cumulative.mp4")