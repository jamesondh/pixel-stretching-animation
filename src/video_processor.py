"""
Video processing module for applying post-processing effects to existing videos.
"""

import numpy as np
import imageio
from typing import Union, List, Optional, Iterator
from pathlib import Path
from tqdm import tqdm
from .post_processors import PostProcessor
from .post_processing_factory import PostProcessorFactory


class VideoProcessor:
    """Process existing videos with post-processing effects."""
    
    def __init__(self, post_processor: Optional[PostProcessor] = None):
        """
        Initialize video processor.
        
        Args:
            post_processor: PostProcessor instance to apply to frames
        """
        self.post_processor = post_processor
    
    def load_video_frames(self, input_path: Union[str, Path]) -> Iterator[np.ndarray]:
        """
        Load video frames as an iterator.
        
        Args:
            input_path: Path to input video
            
        Yields:
            Video frames as numpy arrays
        """
        input_path = Path(input_path)
        reader = imageio.get_reader(input_path)
        
        try:
            for frame in reader:
                yield frame
        finally:
            reader.close()
    
    def get_video_info(self, input_path: Union[str, Path]) -> dict:
        """
        Get video information.
        
        Args:
            input_path: Path to input video
            
        Returns:
            Dictionary with video metadata
        """
        input_path = Path(input_path)
        reader = imageio.get_reader(input_path)
        
        try:
            meta = reader.get_meta_data()
            info = {
                'fps': meta.get('fps', 30),
                'size': meta.get('size', (0, 0)),
                'duration': meta.get('duration', 0),
                'nframes': reader.count_frames() if hasattr(reader, 'count_frames') else None
            }
            return info
        finally:
            reader.close()
    
    def process_video(self, input_path: Union[str, Path], output_path: Union[str, Path],
                     post_processor: Optional[PostProcessor] = None,
                     fps: Optional[int] = None, codec: str = 'libx264',
                     quality: Optional[int] = None, show_progress: bool = True) -> None:
        """
        Process a video file with post-processing effects.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            post_processor: Optional PostProcessor to use (overrides instance processor)
            fps: Output FPS (defaults to input FPS)
            codec: Video codec to use
            quality: Video quality (codec-specific)
            show_progress: Whether to show progress bar
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Use provided processor or instance processor
        processor = post_processor or self.post_processor
        if not processor:
            raise ValueError("No post-processor specified")
        
        # Get video info
        info = self.get_video_info(input_path)
        total_frames = info.get('nframes')
        
        if fps is None:
            fps = info['fps']
        
        # Prepare output writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine save parameters based on format
        save_kwargs = {'fps': fps}
        if output_path.suffix.lower() == '.mp4':
            save_kwargs['codec'] = codec
            if quality:
                save_kwargs['quality'] = quality
        
        # Process frames
        processed_frames = []
        frame_iterator = self.load_video_frames(input_path)
        
        if show_progress and total_frames:
            frame_iterator = tqdm(frame_iterator, total=total_frames, desc="Processing frames")
        
        for i, frame in enumerate(frame_iterator):
            # Apply post-processing
            processed_frame = processor.process_frame(
                frame, i, total_frames if total_frames else i + 1
            )
            processed_frames.append(processed_frame)
        
        # Save processed video
        imageio.mimsave(output_path, processed_frames, **save_kwargs)
        
        if show_progress:
            print(f"Saved processed video to: {output_path}")
    
    def process_video_stream(self, input_path: Union[str, Path], output_path: Union[str, Path],
                            post_processor: Optional[PostProcessor] = None,
                            fps: Optional[int] = None, codec: str = 'libx264',
                            quality: Optional[int] = None, show_progress: bool = True,
                            batch_size: int = 100) -> None:
        """
        Process a video file in streaming mode (lower memory usage).
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            post_processor: Optional PostProcessor to use
            fps: Output FPS
            codec: Video codec
            quality: Video quality
            show_progress: Whether to show progress
            batch_size: Number of frames to process at once
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Use provided processor or instance processor
        processor = post_processor or self.post_processor
        if not processor:
            raise ValueError("No post-processor specified")
        
        # Get video info
        info = self.get_video_info(input_path)
        total_frames = info.get('nframes')
        
        if fps is None:
            fps = info['fps']
        
        # Prepare output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open writer
        writer_kwargs = {'fps': fps}
        if output_path.suffix.lower() == '.mp4':
            writer_kwargs['codec'] = codec
            if quality:
                writer_kwargs['quality'] = quality
        
        writer = imageio.get_writer(output_path, **writer_kwargs)
        
        try:
            # Process frames in batches
            frame_iterator = self.load_video_frames(input_path)
            
            if show_progress and total_frames:
                frame_iterator = tqdm(frame_iterator, total=total_frames, desc="Processing frames")
            
            batch = []
            frame_count = 0
            
            for frame in frame_iterator:
                # Apply post-processing
                processed_frame = processor.process_frame(
                    frame, frame_count, total_frames if total_frames else frame_count + 1
                )
                batch.append(processed_frame)
                frame_count += 1
                
                # Write batch when full
                if len(batch) >= batch_size:
                    for f in batch:
                        writer.append_data(f)
                    batch = []
            
            # Write remaining frames
            for f in batch:
                writer.append_data(f)
        
        finally:
            writer.close()
        
        if show_progress:
            print(f"Saved processed video to: {output_path}")
    
    @classmethod
    def from_config(cls, config: dict) -> 'VideoProcessor':
        """
        Create VideoProcessor from configuration.
        
        Args:
            config: Configuration dictionary with 'post_processing' section
            
        Returns:
            VideoProcessor instance
        """
        post_proc_config = config.get('post_processing', {})
        
        if post_proc_config.get('enabled') and post_proc_config.get('processors'):
            processor = PostProcessorFactory.create_chain(post_proc_config['processors'])
        else:
            processor = None
        
        return cls(processor)