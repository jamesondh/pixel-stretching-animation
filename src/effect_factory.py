"""
Factory for creating distortion effects from configuration.
"""

from typing import Dict, Any, List, Optional
from .distortion_effects import (
    DistortionEffect, BiasedStretchEffect, WaveDistortionEffect, 
    PivotStretchEffect, CompositeEffect, HorizontalStretchEffect, RotatedStretchEffect,
    FlowingMeltEffect
)
from .config import EffectConfig


def create_effect_from_config(config: EffectConfig, axis: Optional[str] = None) -> DistortionEffect:
    """Create a distortion effect instance from configuration."""
    # Create base effect
    if config.type == 'pivot':
        base_effect = PivotStretchEffect(
            max_stretch=config.max_stretch,
            pivot=config.pivot,
            seed=config.seed
        )
    elif config.type == 'wave':
        base_effect = WaveDistortionEffect(
            max_stretch=config.max_stretch,
            wave_amplitude=config.wave_amplitude,
            wave_frequency=config.wave_frequency,
            wave_phase_shift=config.wave_phase_shift,
            wave_phase_speed=config.wave_phase_speed,
            seed=config.seed
        )
    elif config.type == 'bias':
        base_effect = BiasedStretchEffect(
            max_stretch=config.max_stretch,
            stretch_bias=config.stretch_bias,
            seed=config.seed
        )
    elif config.type == 'composite':
        base_effect = create_composite_effect(config)
    elif config.type == 'flowing_melt':
        base_effect = FlowingMeltEffect(
            max_stretch=config.max_stretch,
            melt_bias=getattr(config, 'melt_bias', 0.15),
            flow_amplitude=getattr(config, 'flow_amplitude', 0.1),
            flow_frequency=getattr(config, 'flow_frequency', 2.0),
            flow_speed=getattr(config, 'flow_speed', 0.3),
            flow_variation=getattr(config, 'flow_variation', 0.2),
            edge_behavior=getattr(config, 'edge_behavior', 'wrap'),
            seed=config.seed
        )
    else:
        raise ValueError(f"Unknown effect type: {config.type}")
    
    # Apply axis transformation if specified
    if axis and axis != 'vertical':
        if axis == 'horizontal':
            return HorizontalStretchEffect(base_effect)
        else:
            try:
                # Try to parse as angle
                angle = float(axis)
                return RotatedStretchEffect(base_effect, angle)
            except ValueError:
                raise ValueError(f"Invalid axis: {axis}. Must be 'vertical', 'horizontal', or a number (angle in degrees)")
    
    return base_effect


def create_composite_effect(config: EffectConfig) -> CompositeEffect:
    """Create a composite effect from configuration."""
    if not config.effects:
        raise ValueError("Composite effect requires 'effects' list")
    
    sub_effects = []
    stretch_curves = []
    for effect_dict in config.effects:
        # Create a temporary EffectConfig for each sub-effect
        sub_config = EffectConfig()
        
        # Copy common parameters from parent config if not specified
        effect_dict.setdefault('max_stretch', config.max_stretch)
        effect_dict.setdefault('seed', config.seed)
        
        # Extract stretch_curve for this effect
        stretch_curve = effect_dict.get('stretch_curve', 'linear')
        stretch_curves.append(stretch_curve)
        
        # Update sub_config with effect parameters
        for key, value in effect_dict.items():
            if hasattr(sub_config, key):
                setattr(sub_config, key, value)
        
        # Create the sub-effect
        sub_effect = create_effect_from_config(sub_config)
        sub_effects.append(sub_effect)
    
    # Use provided weights or default to equal weights
    weights = config.weights if config.weights else None
    
    return CompositeEffect(effects=sub_effects, weights=weights, stretch_curves=stretch_curves)


def create_effect_from_dict(effect_dict: Dict[str, Any]) -> DistortionEffect:
    """Create a distortion effect from a dictionary configuration."""
    config = EffectConfig(**effect_dict)
    return create_effect_from_config(config)