"""
Configuration and parameter validation module.
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml


@dataclass
class EffectConfig:
    """Configuration for distortion effects."""
    type: str = 'pivot'  # 'pivot', 'wave', 'bias', 'composite', 'flowing_melt'
    max_stretch: float = 0.5
    
    # Pivot effect parameters
    pivot: str = 'center'  # 'center', 'top', 'bottom'
    
    # Wave effect parameters
    wave_amplitude: float = 0.1
    wave_frequency: float = 2.0
    wave_phase_shift: float = 0.0
    wave_phase_speed: float = 0.2
    
    # Bias effect parameters
    stretch_bias: float = 0.0
    
    # Flowing melt effect parameters
    melt_bias: float = 0.15
    flow_amplitude: float = 0.1
    flow_frequency: float = 2.0
    flow_speed: float = 0.3
    flow_variation: float = 0.2
    edge_behavior: str = 'wrap'  # 'wrap', 'clamp', 'fade'
    
    # Common parameters
    seed: Optional[int] = None
    
    # Stretch curve parameters
    stretch_curve: str = 'linear'  # 'linear', 'constant', 'ease_in', 'ease_out', 'ease_in_out'
    start_stretch: Optional[float] = None
    end_stretch: Optional[float] = None
    
    # Composite effect parameters
    effects: Optional[List[Dict[str, Any]]] = None
    weights: Optional[List[float]] = None
    
    def validate(self):
        """Validate effect configuration."""
        if self.type not in ['pivot', 'wave', 'bias', 'composite', 'flowing_melt', 'sine_wave']:
            raise ValueError(f"Invalid effect type: {self.type}")
        
        if not 0 <= self.max_stretch <= 1:
            raise ValueError("max_stretch must be between 0 and 1")
        
        if self.pivot not in ['center', 'top', 'bottom']:
            raise ValueError(f"Invalid pivot: {self.pivot}")
        
        if not -1 <= self.stretch_bias <= 1:
            raise ValueError("stretch_bias must be between -1 and 1")
        
        # Validate flowing melt parameters
        if not 0 <= self.melt_bias <= 1:
            raise ValueError("melt_bias must be between 0 and 1")
        
        if not 0 <= self.flow_amplitude <= 1:
            raise ValueError("flow_amplitude must be between 0 and 1")
        
        if not 0 <= self.flow_variation <= 1:
            raise ValueError("flow_variation must be between 0 and 1")
        
        if self.edge_behavior not in ['wrap', 'clamp', 'fade']:
            raise ValueError(f"Invalid edge_behavior: {self.edge_behavior}")
        
        if self.stretch_curve not in ['linear', 'constant', 'ease_in', 'ease_out', 'ease_in_out']:
            raise ValueError(f"Invalid stretch_curve: {self.stretch_curve}")
        
        if self.start_stretch is not None and not 0 <= self.start_stretch <= 1:
            raise ValueError("start_stretch must be between 0 and 1")
        
        if self.end_stretch is not None and not 0 <= self.end_stretch <= 1:
            raise ValueError("end_stretch must be between 0 and 1")
        
        # Validate composite effect configuration
        if self.type == 'composite':
            if not self.effects or len(self.effects) < 2:
                raise ValueError("Composite effect must have at least 2 sub-effects")
            
            if self.weights:
                if len(self.weights) != len(self.effects):
                    raise ValueError("Number of weights must match number of effects")
                if any(w < 0 for w in self.weights):
                    raise ValueError("Weights must be non-negative")
                if sum(self.weights) == 0:
                    raise ValueError("At least one weight must be greater than 0")


@dataclass
class AnimationConfig:
    """Configuration for animation parameters."""
    frames: int = 60
    fps: int = 30
    interpolation: str = 'nearest'  # 'nearest' or 'bilinear'
    temporal_smoothing: float = 0.0
    upscale: int = 1
    cumulative: bool = False
    frame_generator: str = 'standard'  # 'standard', 'cumulative', 'pingpong'
    
    def validate(self):
        """Validate animation configuration."""
        if self.frames < 1:
            raise ValueError("frames must be at least 1")
        
        if self.fps < 1:
            raise ValueError("fps must be at least 1")
        
        if self.interpolation not in ['nearest', 'bilinear']:
            raise ValueError(f"Invalid interpolation: {self.interpolation}")
        
        if not 0 <= self.temporal_smoothing <= 1:
            raise ValueError("temporal_smoothing must be between 0 and 1")
        
        if self.upscale < 1:
            raise ValueError("upscale must be at least 1")
        
        if self.frame_generator not in ['standard', 'cumulative', 'pingpong']:
            raise ValueError(f"Invalid frame_generator: {self.frame_generator}")


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing effects."""
    enabled: bool = False
    processors: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate(self):
        """Validate post-processing configuration."""
        for i, proc in enumerate(self.processors):
            if 'type' not in proc:
                raise ValueError(f"Post-processor {i} must have a 'type' field")
            
            # Validate specific processor types
            proc_type = proc['type']
            if proc_type == 'upscale':
                scale_factor = proc.get('scale_factor', 2)
                if not isinstance(scale_factor, int) or scale_factor < 1:
                    raise ValueError(f"Upscale scale_factor must be an integer >= 1")
                
                method = proc.get('method', 'nearest')
                if method not in ['nearest', 'bilinear', 'cubic']:
                    raise ValueError(f"Invalid upscale method: {method}")
            
            elif proc_type == 'sine_wave':
                axis = proc.get('axis', 'vertical')
                if axis not in ['vertical', 'horizontal', 'diagonal', 'angle']:
                    raise ValueError(f"Invalid sine_wave axis: {axis}")
                
                if axis == 'angle' and 'angle' not in proc:
                    raise ValueError("sine_wave with axis='angle' requires 'angle' parameter")
                
                displacement_mode = proc.get('displacement_mode', 'translate')
                if displacement_mode not in ['translate', 'scale', 'both']:
                    raise ValueError(f"Invalid displacement_mode: {displacement_mode}")
                
                edge_behavior = proc.get('edge_behavior', 'wrap')
                if edge_behavior not in ['wrap', 'clamp', 'fade', 'mirror']:
                    raise ValueError(f"Invalid edge_behavior: {edge_behavior}")
                
                amplitude_curve = proc.get('amplitude_curve', 'constant')
                if amplitude_curve not in ['constant', 'linear', 'ease_in', 'ease_out', 'ease_in_out']:
                    raise ValueError(f"Invalid amplitude_curve: {amplitude_curve}")


@dataclass
class OutputConfig:
    """Configuration for output parameters."""
    format: str = 'mp4'  # 'mp4', 'avi', 'mov'
    codec: str = 'libx264'
    quality: Optional[int] = None
    
    def validate(self):
        """Validate output configuration."""
        if self.format not in ['mp4', 'avi', 'mov']:
            raise ValueError(f"Invalid output format: {self.format}")


@dataclass
class PixelStretchConfig:
    """Complete configuration for pixel stretching."""
    effect: EffectConfig = field(default_factory=EffectConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def validate(self):
        """Validate all configuration sections."""
        self.effect.validate()
        self.animation.validate()
        self.post_processing.validate()
        self.output.validate()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PixelStretchConfig':
        """Create configuration from dictionary."""
        effect_dict = config_dict.get('effect', {})
        animation_dict = config_dict.get('animation', {})
        post_processing_dict = config_dict.get('post_processing', {})
        output_dict = config_dict.get('output', {})
        
        return cls(
            effect=EffectConfig(**effect_dict),
            animation=AnimationConfig(**animation_dict),
            post_processing=PostProcessingConfig(**post_processing_dict),
            output=OutputConfig(**output_dict)
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'PixelStretchConfig':
        """Load configuration from file (JSON or YAML)."""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            'effect': {
                k: v for k, v in self.effect.__dict__.items() 
                if v is not None
            },
            'animation': self.animation.__dict__,
            'output': {
                k: v for k, v in self.output.__dict__.items() 
                if v is not None
            }
        }
        
        # Only include post_processing if enabled
        if self.post_processing.enabled:
            result['post_processing'] = {
                'enabled': True,
                'processors': self.post_processing.processors
            }
        
        return result
    
    def save(self, config_path: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)


# Preset configurations
PRESETS = {
    'pixel_art': PixelStretchConfig(
        effect=EffectConfig(type='pivot', max_stretch=0.4),
        animation=AnimationConfig(
            interpolation='nearest',
            upscale=4,
            fps=30
        )
    ),
    'smooth_wave': PixelStretchConfig(
        effect=EffectConfig(
            type='wave',
            max_stretch=0.3,
            wave_amplitude=0.15,
            wave_frequency=2.5
        ),
        animation=AnimationConfig(
            interpolation='bilinear',
            temporal_smoothing=0.7,
            fps=60
        )
    ),
    'melting': PixelStretchConfig(
        effect=EffectConfig(
            type='bias',
            max_stretch=0.6,
            stretch_bias=0.8
        ),
        animation=AnimationConfig(
            cumulative=True,
            frames=90,
            fps=30
        )
    ),
    'bouncy': PixelStretchConfig(
        effect=EffectConfig(type='pivot', max_stretch=0.7),
        animation=AnimationConfig(
            frame_generator='pingpong',
            frames=60,
            fps=30
        )
    )
}


def get_preset(name: str) -> PixelStretchConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    
    # Return a copy to avoid modifying the preset
    preset = PRESETS[name]
    return PixelStretchConfig.from_dict(preset.to_dict())