import whisper
import tempfile
import os
import time
import shutil
from typing import Tuple


class WhisperService:
    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize Whisper service with model configuration.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)  
            compute_type: Compute precision (not used with OpenAI Whisper)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model lazily."""
        if self.model is None:
            print(f"Loading Whisper model: {self.model_name}")
            
            # Ensure ffmpeg is available in PATH
            self._setup_ffmpeg_path()
            
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"Model loaded on device: {self.device}")
    
    def _setup_ffmpeg_path(self):
        """Ensure ffmpeg is available in PATH."""
        # Common ffmpeg locations
        ffmpeg_paths = [
            "/opt/homebrew/bin/ffmpeg",  # Homebrew on Apple Silicon
            "/usr/local/bin/ffmpeg",     # Homebrew on Intel
            "/usr/bin/ffmpeg",           # System package managers
        ]
        
        # Check if ffmpeg is already in PATH
        if shutil.which("ffmpeg"):
            return
            
        # Try to find ffmpeg and add its directory to PATH
        for ffmpeg_path in ffmpeg_paths:
            if os.path.exists(ffmpeg_path):
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                current_path = os.environ.get("PATH", "")
                if ffmpeg_dir not in current_path:
                    os.environ["PATH"] = f"{ffmpeg_dir}:{current_path}"
                    print(f"Added {ffmpeg_dir} to PATH for ffmpeg")
                return
        
        print("Warning: ffmpeg not found in common locations")
    
    def transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio file bytes
            
        Returns:
            Tuple of (transcribed_text, duration_seconds)
        """
        start_time = time.time()
        
        # Create temporary file for audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            temp_filename = temp_file.name
        
        try:
            # Transcribe the audio file
            result = self.model.transcribe(temp_filename)
            text = result["text"].strip()
            
            # Calculate processing duration
            duration = time.time() - start_time
            
            return text, duration
            
        except Exception as e:
            error_msg = str(e)
            print(f"Transcription error: {e}")
            
            # Provide helpful error message for missing ffmpeg
            if "ffmpeg" in error_msg.lower():
                raise Exception(
                    "ffmpeg is required for audio processing. "
                    "Install it using: brew install ffmpeg (requires Homebrew) "
                    "or download from https://ffmpeg.org/download.html"
                )
            
            raise Exception(f"Failed to transcribe audio: {error_msg}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
