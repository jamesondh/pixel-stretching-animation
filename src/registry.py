"""
Feature registry system for automatic discovery and documentation.
"""

from typing import Dict, Any, Type, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import click

from .distortion_effects import (
    DistortionEffect, PivotStretchEffect, WaveDistortionEffect, BiasedStretchEffect
)


@dataclass
class ParameterInfo:
    """Information about a parameter."""
    name: str
    type: type
    default: Any
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None


@dataclass
class FeatureInfo:
    """Information about a registered feature."""
    name: str
    class_type: Type
    description: str
    category: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


class FeatureRegistry:
    """Central registry for all effects and features."""
    
    def __init__(self):
        self._effects: Dict[str, FeatureInfo] = {}
        self._frame_generators: Dict[str, FeatureInfo] = {}
        self._processors: Dict[str, FeatureInfo] = {}
        self._categories = {
            'effects': self._effects,
            'frame_generators': self._frame_generators,
            'processors': self._processors
        }
        
        # Register built-in features
        self._register_builtin_effects()
    
    def register_effect(self, name: str, effect_class: Type[DistortionEffect], 
                       description: str, parameters: List[ParameterInfo],
                       examples: Optional[List[Dict[str, Any]]] = None):
        """Register a distortion effect."""
        self._effects[name] = FeatureInfo(
            name=name,
            class_type=effect_class,
            description=description,
            category='effects',
            parameters=parameters,
            examples=examples or []
        )
    
    def register_frame_generator(self, name: str, generator_class: Type,
                                description: str, parameters: List[ParameterInfo]):
        """Register a frame generator."""
        self._frame_generators[name] = FeatureInfo(
            name=name,
            class_type=generator_class,
            description=description,
            category='frame_generators',
            parameters=parameters
        )
    
    def get_effect(self, name: str) -> Optional[FeatureInfo]:
        """Get effect information by name."""
        return self._effects.get(name)
    
    def get_all_effects(self) -> Dict[str, FeatureInfo]:
        """Get all registered effects."""
        return self._effects.copy()
    
    def create_effect(self, name: str, **kwargs) -> DistortionEffect:
        """Create an effect instance with given parameters."""
        effect_info = self.get_effect(name)
        if not effect_info:
            raise ValueError(f"Unknown effect: {name}")
        
        # Filter kwargs to only include valid parameters
        valid_params = {p.name for p in effect_info.parameters}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return effect_info.class_type(**filtered_kwargs)
    
    def generate_cli_options(self, effect_name: str) -> List[click.Option]:
        """Generate Click options for an effect."""
        effect_info = self.get_effect(effect_name)
        if not effect_info:
            return []
        
        options = []
        for param in effect_info.parameters:
            option_kwargs = {
                'help': param.description
            }
            
            if param.default is not None:
                option_kwargs['default'] = param.default
            
            if param.type == bool:
                option_kwargs['is_flag'] = True
            elif param.type == int:
                option_kwargs['type'] = click.INT
            elif param.type == float:
                option_kwargs['type'] = click.FLOAT
            elif param.choices:
                option_kwargs['type'] = click.Choice(param.choices)
            
            option_name = f"--{param.name.replace('_', '-')}"
            options.append(click.option(option_name, **option_kwargs))
        
        return options
    
    def generate_documentation(self) -> str:
        """Generate documentation for all registered features."""
        doc_lines = ["# Auto-generated Feature Documentation\n"]
        
        for category, features in self._categories.items():
            if not features:
                continue
                
            doc_lines.append(f"\n## {category.title()}\n")
            
            for name, info in features.items():
                doc_lines.append(f"### {name}\n")
                doc_lines.append(f"{info.description}\n")
                
                if info.parameters:
                    doc_lines.append("**Parameters:**\n")
                    for param in info.parameters:
                        param_desc = f"- `{param.name}` ({param.type.__name__})"
                        if param.default is not None:
                            param_desc += f" = {param.default}"
                        param_desc += f": {param.description}"
                        
                        if param.min_value is not None or param.max_value is not None:
                            param_desc += f" (range: {param.min_value}-{param.max_value})"
                        elif param.choices:
                            param_desc += f" (choices: {', '.join(param.choices)})"
                        
                        doc_lines.append(param_desc)
                    doc_lines.append("")
                
                if info.examples:
                    doc_lines.append("**Examples:**\n")
                    for example in info.examples:
                        doc_lines.append(f"```python")
                        doc_lines.append(f"{name}_effect = {info.class_type.__name__}(")
                        for key, value in example.items():
                            doc_lines.append(f"    {key}={repr(value)},")
                        doc_lines.append(")")
                        doc_lines.append("```\n")
        
        return "\n".join(doc_lines)
    
    def _register_builtin_effects(self):
        """Register all built-in effects."""
        # Pivot effect
        self.register_effect(
            name='pivot',
            effect_class=PivotStretchEffect,
            description='Original pivot-based stretching effect',
            parameters=[
                ParameterInfo(
                    name='max_stretch',
                    type=float,
                    default=0.5,
                    description='Maximum stretch factor',
                    min_value=0.0,
                    max_value=1.0
                ),
                ParameterInfo(
                    name='pivot',
                    type=str,
                    default='center',
                    description='Pivot point for stretching',
                    choices=['center', 'top', 'bottom']
                ),
                ParameterInfo(
                    name='seed',
                    type=int,
                    default=None,
                    description='Random seed for reproducibility'
                )
            ],
            examples=[
                {'max_stretch': 0.3, 'pivot': 'center'},
                {'max_stretch': 0.5, 'pivot': 'top', 'seed': 42}
            ]
        )
        
        # Wave effect
        self.register_effect(
            name='wave',
            effect_class=WaveDistortionEffect,
            description='Wave-based distortion with animated phase',
            parameters=[
                ParameterInfo(
                    name='max_stretch',
                    type=float,
                    default=0.5,
                    description='Maximum stretch factor',
                    min_value=0.0,
                    max_value=1.0
                ),
                ParameterInfo(
                    name='wave_amplitude',
                    type=float,
                    default=0.1,
                    description='Amplitude of the wave',
                    min_value=0.0,
                    max_value=1.0
                ),
                ParameterInfo(
                    name='wave_frequency',
                    type=float,
                    default=2.0,
                    description='Number of wave cycles',
                    min_value=0.1,
                    max_value=10.0
                ),
                ParameterInfo(
                    name='wave_phase_shift',
                    type=float,
                    default=0.0,
                    description='Initial phase offset in radians'
                ),
                ParameterInfo(
                    name='wave_phase_speed',
                    type=float,
                    default=0.2,
                    description='Speed of wave animation'
                ),
                ParameterInfo(
                    name='seed',
                    type=int,
                    default=None,
                    description='Random seed for reproducibility'
                )
            ],
            examples=[
                {'max_stretch': 0.3, 'wave_amplitude': 0.15, 'wave_frequency': 2.5},
                {'max_stretch': 0.5, 'wave_amplitude': 0.2, 'wave_frequency': 1.0, 'wave_phase_speed': 0.5}
            ]
        )
        
        # Bias effect
        self.register_effect(
            name='bias',
            effect_class=BiasedStretchEffect,
            description='Directional stretching with configurable bias',
            parameters=[
                ParameterInfo(
                    name='max_stretch',
                    type=float,
                    default=0.5,
                    description='Maximum stretch factor',
                    min_value=0.0,
                    max_value=1.0
                ),
                ParameterInfo(
                    name='stretch_bias',
                    type=float,
                    default=0.0,
                    description='Bias for stretch direction (-1=up, 1=down)',
                    min_value=-1.0,
                    max_value=1.0
                ),
                ParameterInfo(
                    name='seed',
                    type=int,
                    default=None,
                    description='Random seed for reproducibility'
                )
            ],
            examples=[
                {'max_stretch': 0.6, 'stretch_bias': 0.8},  # Melting effect
                {'max_stretch': 0.4, 'stretch_bias': -0.7}  # Anti-gravity effect
            ]
        )


# Global registry instance
registry = FeatureRegistry()


def discover_custom_effects(module_path: str):
    """Discover and register custom effects from a module."""
    import importlib
    import pkgutil
    
    module = importlib.import_module(module_path)
    
    for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
        submodule = importlib.import_module(f"{module_path}.{modname}")
        
        for name, obj in inspect.getmembers(submodule):
            if (inspect.isclass(obj) and 
                issubclass(obj, DistortionEffect) and 
                obj != DistortionEffect):
                
                # Auto-register the effect
                effect_name = name.lower().replace('effect', '').replace('distortion', '')
                
                # Extract parameters from __init__ signature
                sig = inspect.signature(obj.__init__)
                parameters = []
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else type(param.default)
                    parameters.append(ParameterInfo(
                        name=param_name,
                        type=param_type,
                        default=param.default if param.default != inspect.Parameter.empty else None,
                        description=f"Parameter {param_name}"  # Would need docstring parsing for better descriptions
                    ))
                
                registry.register_effect(
                    name=effect_name,
                    effect_class=obj,
                    description=obj.__doc__ or f"Custom {effect_name} effect",
                    parameters=parameters
                )