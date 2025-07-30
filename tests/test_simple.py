#!/usr/bin/env python3
from src.pixel_stretcher import PixelStretcher
from pathlib import Path

# Test cumulative mode with fewer frames
input_image = Path("assets/sample_images/noise.png")

print("Creating cumulative animation with max_stretch=0.3...")
stretcher = PixelStretcher(
    max_stretch=0.3,
    seed=42,
    cumulative=True
)
stretcher.create_animation(
    input_path=input_image,
    output_path="output/test_cumulative_simple.mp4",
    frames=20,
    fps=10
)

print("Done! Check output/test_cumulative_simple.mp4")