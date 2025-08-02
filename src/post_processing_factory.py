"""
Factory for creating post-processor instances from configuration.
"""

from typing import Dict, Any, List, Optional
from .post_processors import (
    PostProcessor, PostProcessorChain, IdentityPostProcessor,
    UpscalePostProcessor, SineWavePostProcessor
)


class PostProcessorFactory:
    """Factory for creating post-processors from configuration."""
    
    @staticmethod
    def create_processor(processor_type: str, config: Optional[Dict[str, Any]] = None) -> PostProcessor:
        """
        Create a single post-processor instance.
        
        Args:
            processor_type: Type of processor to create
            config: Configuration dictionary
            
        Returns:
            PostProcessor instance
        """
        if config is None:
            config = {}
            
        # Simple factory pattern matching the effect_factory approach
        if processor_type == 'identity':
            return IdentityPostProcessor()
        elif processor_type == 'upscale':
            return UpscalePostProcessor(
                scale_factor=config.get('scale_factor', 2),
                method=config.get('method', 'nearest')
            )
        elif processor_type == 'sine_wave':
            return SineWavePostProcessor(
                axis=config.get('axis', 'vertical'),
                angle=config.get('angle'),
                frequency=config.get('frequency', 3.0),
                amplitude=config.get('amplitude', 0.05),
                phase=config.get('phase', 0.0),
                speed=config.get('speed', 0.5),
                displacement_mode=config.get('displacement_mode', 'translate'),
                edge_behavior=config.get('edge_behavior', 'wrap'),
                amplitude_curve=config.get('amplitude_curve', 'constant'),
                start_amplitude=config.get('start_amplitude'),
                end_amplitude=config.get('end_amplitude'),
                interpolation=config.get('interpolation', 'bilinear'),
                preserve_palette=config.get('preserve_palette', False),
                amplitude_gradient=config.get('amplitude_gradient', 'none'),
                gradient_curve=config.get('gradient_curve', 'linear'),
                gradient_start=config.get('gradient_start', 0.0),
                gradient_end=config.get('gradient_end', 1.0)
            )
        else:
            raise ValueError(f"Unknown post-processor type: {processor_type}")
    
    @staticmethod
    def create_chain(processors_config: List[Dict[str, Any]]) -> PostProcessor:
        """
        Create a chain of post-processors from configuration.
        
        Args:
            processors_config: List of processor configurations
            
        Returns:
            PostProcessorChain instance or single processor if only one
        """
        if not processors_config:
            return IdentityPostProcessor()
        
        processors = []
        for proc_config in processors_config:
            processor_type = proc_config.get('type')
            if not processor_type:
                raise ValueError("Post-processor configuration must include 'type'")
            
            # Remove 'type' from config before passing to creator
            config_copy = proc_config.copy()
            config_copy.pop('type', None)
            
            processor = PostProcessorFactory.create_processor(processor_type, config_copy)
            processors.append(processor)
        
        if len(processors) == 1:
            return processors[0]
        else:
            return PostProcessorChain(processors)
    
    @staticmethod
    def list_available() -> List[str]:
        """List available post-processor types."""
        return ['identity', 'upscale', 'sine_wave']