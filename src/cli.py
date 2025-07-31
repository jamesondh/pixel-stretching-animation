"""
Enhanced CLI interface with subcommands and better organization.
"""

import click
from pathlib import Path
from typing import Optional
import json
import yaml

from .pixel_stretcher import PixelStretcher
from .config import PixelStretchConfig, get_preset, PRESETS
from .distortion_effects import (
    BiasedStretchEffect, WaveDistortionEffect, PivotStretchEffect,
    HorizontalStretchEffect, RotatedStretchEffect
)
from .animation import AnimationEngine, StandardFrameGenerator, CumulativeFrameGenerator, PingPongFrameGenerator
from .effect_factory import create_effect_from_config


@click.group()
@click.version_option()
def cli():
    """Pixel Stretching Animation Tool - Create mesmerizing animations from static images."""
    pass


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--preset', '-p', type=click.Choice(list(PRESETS.keys())), 
              help='Use a preset configuration')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Load configuration from file')
@click.option('--frames', '-f', default=60, help='Number of frames to generate')
@click.option('--fps', default=30, help='Frames per second')
@click.option('--max-stretch', '-s', default=0.5, help='Maximum stretch factor (0-1)')
@click.option('--effect', type=click.Choice(['pivot', 'wave', 'bias', 'flowing_melt']), 
              default='pivot', help='Effect type to use')
@click.option('--pivot', type=click.Choice(['center', 'top', 'bottom']), 
              default='center', help='Pivot point for stretching')
@click.option('--interpolation', type=click.Choice(['nearest', 'bilinear']), 
              default='nearest', help='Interpolation method')
@click.option('--temporal-smoothing', '-t', default=0.0, 
              help='Temporal smoothing factor (0-1)')
@click.option('--seed', type=int, default=None, help='Random seed for reproducibility')
@click.option('--upscale', '-u', default=1, type=int, help='Upscale factor')
@click.option('--cumulative', '-C', is_flag=True, 
              help='Apply distortion cumulatively to each frame')
@click.option('--wave-amplitude', default=0.1, help='Wave amplitude (for wave effect)')
@click.option('--wave-frequency', default=2.0, help='Wave frequency (for wave effect)')
@click.option('--stretch-bias', '-b', default=0.0, 
              help='Bias stretching direction (-1 to 1, for bias effect)')
@click.option('--stretch-curve', type=click.Choice(['linear', 'constant', 'ease_in', 'ease_out', 'ease_in_out']),
              default='linear', help='Animation curve for stretch progression')
@click.option('--start-stretch', type=float, default=None,
              help='Starting stretch value (defaults to 0)')
@click.option('--end-stretch', type=float, default=None,
              help='Ending stretch value (defaults to max-stretch)')
@click.option('--axis', type=str, default='vertical',
              help='Stretch axis: "vertical", "horizontal", or angle in degrees (e.g., "45")')
def animate(input_path, output_path, preset, config, frames, fps, max_stretch, 
           effect, pivot, interpolation, temporal_smoothing, seed, upscale, 
           cumulative, wave_amplitude, wave_frequency, stretch_bias,
           stretch_curve, start_stretch, end_stretch, axis):
    """Create a pixel-stretching animation from an input image."""
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Load configuration
    if config:
        cfg = PixelStretchConfig.from_file(config)
        click.echo(f"Loaded configuration from {config}")
    elif preset:
        cfg = get_preset(preset)
        click.echo(f"Using preset: {preset}")
    else:
        # Build configuration from command line arguments
        cfg = PixelStretchConfig()
        cfg.effect.type = effect
        cfg.effect.max_stretch = max_stretch
        cfg.effect.pivot = pivot
        cfg.effect.seed = seed
        cfg.effect.wave_amplitude = wave_amplitude
        cfg.effect.wave_frequency = wave_frequency
        cfg.effect.stretch_bias = stretch_bias
        cfg.effect.stretch_curve = stretch_curve
        cfg.effect.start_stretch = start_stretch
        cfg.effect.end_stretch = end_stretch
        
        cfg.animation.frames = frames
        cfg.animation.fps = fps
        cfg.animation.interpolation = interpolation
        cfg.animation.temporal_smoothing = temporal_smoothing
        cfg.animation.upscale = upscale
        cfg.animation.cumulative = cumulative
    
    # Validate configuration
    cfg.validate()
    
    # Validate output format
    if output_path.suffix.lower() not in ['.mp4', '.avi', '.mov']:
        raise click.BadParameter("Output must be .mp4, .avi, or .mov")
    
    click.echo(f"Creating animation from {input_path}")
    click.echo(f"Effect: {cfg.effect.type}, {cfg.animation.frames} frames at {cfg.animation.fps} fps")
    
    # Create effect from configuration with axis
    effect = create_effect_from_config(cfg.effect, axis=axis)
    
    # Create stretcher with the effect
    stretcher = PixelStretcher(
        effect=effect,
        interpolation=cfg.animation.interpolation,
        temporal_smoothing=cfg.animation.temporal_smoothing,
        upscale=cfg.animation.upscale,
        cumulative=cfg.animation.cumulative,
        stretch_curve=cfg.effect.stretch_curve,
        start_stretch=cfg.effect.start_stretch if cfg.effect.start_stretch is not None else 0.0,
        end_stretch=cfg.effect.end_stretch if cfg.effect.end_stretch is not None else cfg.effect.max_stretch
    )
    
    # Generate animation
    stretcher.create_animation(
        input_path=input_path,
        output_path=output_path,
        frames=cfg.animation.frames,
        fps=cfg.animation.fps
    )
    
    click.echo(f"Animation saved to {output_path}")


@cli.command()
def effects():
    """List available effects and their parameters."""
    effects_info = {
        'pivot': {
            'description': 'Original pivot-based stretching',
            'parameters': ['max_stretch', 'pivot (center/top/bottom)']
        },
        'wave': {
            'description': 'Wave-based distortion with animated phase',
            'parameters': ['max_stretch', 'wave_amplitude', 'wave_frequency', 
                         'wave_phase_shift', 'wave_phase_speed']
        },
        'bias': {
            'description': 'Directional stretching with configurable bias',
            'parameters': ['max_stretch', 'stretch_bias (-1 to 1)']
        },
        'composite': {
            'description': 'Combine multiple effects with weighted blending',
            'parameters': ['effects (list of effect configs)', 'weights (optional blend ratios)']
        }
    }
    
    click.echo("Available effects:\n")
    for effect_name, info in effects_info.items():
        click.echo(f"  {effect_name}: {info['description']}")
        click.echo(f"    Parameters: {', '.join(info['parameters'])}\n")
    
    click.echo("Axis options:")
    click.echo("  All effects can be applied along different axes using the --axis parameter:")
    click.echo("    vertical    : Default vertical stretching (up/down)")
    click.echo("    horizontal  : Horizontal stretching (left/right)")
    click.echo("    <degrees>   : Any angle in degrees (e.g., 45, -30, 90)\n")
    click.echo("  Examples:")
    click.echo("    --axis horizontal        # Left-right stretching")
    click.echo("    --axis 45               # Diagonal stretching at 45°")
    click.echo("    --axis -30              # Diagonal stretching at -30°")


@cli.command()
def presets():
    """List available presets."""
    click.echo("Available presets:\n")
    for name, preset in PRESETS.items():
        click.echo(f"  {name}:")
        click.echo(f"    Effect: {preset.effect.type}")
        click.echo(f"    Max stretch: {preset.effect.max_stretch}")
        click.echo(f"    Frames: {preset.animation.frames} @ {preset.animation.fps} fps")
        click.echo(f"    Features: ", nl=False)
        features = []
        if preset.animation.cumulative:
            features.append("cumulative")
        if preset.animation.upscale > 1:
            features.append(f"{preset.animation.upscale}x upscale")
        if preset.animation.temporal_smoothing > 0:
            features.append("smoothed")
        click.echo(", ".join(features) if features else "none")
        click.echo()


@cli.command()
@click.argument('output_path', type=click.Path())
@click.option('--preset', '-p', type=click.Choice(list(PRESETS.keys())), 
              help='Base configuration on a preset')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), 
              default='yaml', help='Output format')
def create_config(output_path, preset, format):
    """Create a configuration file template."""
    output_path = Path(output_path)
    
    # Start with preset or default
    if preset:
        cfg = get_preset(preset)
        click.echo(f"Creating config based on preset: {preset}")
    else:
        cfg = PixelStretchConfig()
        click.echo("Creating default configuration")
    
    # Ensure correct extension
    if format == 'yaml' and output_path.suffix not in ['.yaml', '.yml']:
        output_path = output_path.with_suffix('.yaml')
    elif format == 'json' and output_path.suffix != '.json':
        output_path = output_path.with_suffix('.json')
    
    cfg.save(output_path)
    click.echo(f"Configuration saved to {output_path}")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--size', '-s', default=64, help='Preview size (width and height)')
@click.option('--effect', type=click.Choice(['pivot', 'wave', 'bias', 'flowing_melt']), 
              default='pivot', help='Effect type to preview')
@click.option('--max-stretch', default=0.5, help='Maximum stretch factor')
def preview(input_path, size, effect, max_stretch):
    """Generate a quick preview of the effect."""
    from PIL import Image
    import numpy as np
    
    input_path = Path(input_path)
    
    # Load and resize image for preview
    img = Image.open(input_path)
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    if img.mode not in ['RGB', 'RGBA']:
        img = img.convert('RGB')
    
    # Create preview with 10 frames
    preview_path = input_path.parent / f"preview_{effect}_{input_path.stem}.mp4"
    
    stretcher = PixelStretcher(
        max_stretch=max_stretch,
        use_wave_distortion=(effect == 'wave'),
        stretch_bias=0.7 if effect == 'bias' else 0.0
    )
    
    stretcher.create_animation(
        input_path=input_path,
        output_path=preview_path,
        frames=10,
        fps=10
    )
    
    click.echo(f"Preview saved to {preview_path}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()