"""
Whisper speech-to-text transcription service.

This module provides a service wrapper for OpenAI-compatible Whisper API endpoints,
enabling audio transcription with performance tracking and error handling.
"""
import tempfile
import os
import time
import io
import numpy as np
import soundfile as sf
from scipy import signal
from openai import OpenAI
from typing import Tuple

class WhisperService:
    """
    Service for transcribing audio using OpenAI-compatible Whisper API.
    
    This service manages connections to a Whisper transcription endpoint
    (e.g., whisper.cpp FastAPI server) and handles audio file processing,
    temporary file management, and performance monitoring.
    
    Attributes:
        model_name: Name of the Whisper model to use for transcription
        whisper_url: Base URL of the Whisper API endpoint
        client: OpenAI client instance for API communication
        
    Example:
        >>> service = WhisperService(
        ...     model_name="whisper-large-v3-turbo-quantized",
        ...     whisper_url="http://localhost:8080/v1"
        ... )
        >>> text, duration = service.transcribe(audio_bytes)
        >>> print(f"Transcribed in {duration:.2f}s: {text}")
    """
    def __init__(self, model_name: str = "base", whisper_url: str = "http://localhost:8080/v1"):
        """
        Initialize Whisper transcription service.
        
        Creates a new WhisperService instance configured to connect to an
        OpenAI-compatible Whisper API endpoint. The client is automatically
        loaded during initialization and is ready for transcription operations.
        
        Args:
            model_name: Name or identifier of the Whisper model to use.
                Common values include "base", "small", "medium", "large",
                or custom model names like "whisper-large-v3-turbo-quantized".
                Defaults to "base".
            whisper_url: Base URL of the Whisper API endpoint following
                OpenAI's API format (should end with /v1). 
                Defaults to "http://localhost:8080/v1".
        """
        self.model_name = model_name
        self.whisper_url = whisper_url
        self.load_client()
    
    def load_client(self):
        """
        Initialize the OpenAI client for API communication.
        
        Creates an OpenAI client instance configured to communicate with
        the Whisper API endpoint. Uses a placeholder API key since many
        self-hosted Whisper services don't require authentication.
        
        Note:
            This method is automatically called during __init__(). The client
            is stored as an instance attribute for reuse across multiple transcriptions.
            
        Raises:
            Exception: If the client fails to initialize or the endpoint
                is unreachable (though exceptions are not explicitly caught).
        """
        self.client = OpenAI(
            base_url=self.whisper_url,
            api_key="fake"
        )

    def transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        """
        Transcribe audio data to text using the Whisper model.
        
        Processes raw audio bytes by writing them to a temporary WAV file,
        sending the file to the Whisper API endpoint, and returning the
        transcribed text along with processing duration. Automatically
        cleans up temporary files after transcription.
        
        Args:
            audio_data: Raw audio file bytes in any format supported by soundfile
                (WAV, FLAC, OGG, etc.). The audio will be automatically converted
                to Whisper-compatible format (16-bit PCM, mono, 16kHz).
                
        Returns:
            A tuple containing:
                - transcribed_text (str): The transcribed text from the audio
                - duration_seconds (float): Time taken to process the transcription
                
        Raises:
            Exception: If transcription fails due to network errors, invalid
                audio format, API errors, or other processing issues. The
                exception message includes details about the failure.
                
        Note:
            The audio data is automatically converted to Whisper-compatible format
            (16-bit PCM, mono, 16kHz) and written to a temporary WAV file before
            being sent to the Whisper API. The temporary file is automatically
            cleaned up after transcription.
            
        Example:
            >>> with open("recording.wav", "rb") as f:
            ...     audio_bytes = f.read()
            >>> text, duration = service.transcribe(audio_bytes)
            >>> print(f"Transcription ({duration:.2f}s): {text}")
        """
        start_time = time.time()
        temp_filename = None
        
        # Convert audio data to Whisper-compatible WAV format
        # Whisper expects: 16-bit PCM, mono channel, 16kHz sample rate
        try:
            # Load audio from bytes using soundfile (supports WAV, FLAC, OGG, etc.)
            audio_data_io = io.BytesIO(audio_data)
            audio_data_io.seek(0)
            
            # Read audio file (soundfile auto-detects format)
            audio_array, sample_rate = sf.read(audio_data_io)
            
            # Convert to mono if stereo/multi-channel
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = signal.resample(audio_array, num_samples)
                sample_rate = 16000
            
            # Normalize to int16 range (-32768 to 32767)
            if audio_array.dtype != np.int16:
                # Normalize to [-1, 1] range first
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    # Already in float format, just ensure it's in [-1, 1]
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                else:
                    # Convert to float and normalize
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array.astype(np.float32) / max_val
                
                # Convert to int16
                audio_array = (audio_array * 32767).astype(np.int16)
            
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_filename = temp_file.name
            
            # Write as WAV file (16-bit PCM, mono, 16kHz)
            sf.write(temp_filename, audio_array, sample_rate, subtype='PCM_16')
            
        except Exception as e:
            raise Exception(f"Failed to convert audio to WAV format: {str(e)}")

        try:
            # Open the file in read mode for the API
            with open(temp_filename, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model_name,  # match your deployed model name
                    file=audio_file,
                    language="en"
                )
            
            # Calculate processing duration
            duration = time.time() - start_time
            
            return transcript.text, duration
            
        except Exception as e:
            error_msg = str(e)
            raise Exception(f"Failed to transcribe audio: {error_msg}")
        finally:
            # Clean up temporary file
            if temp_filename and os.path.exists(temp_filename):
                os.unlink(temp_filename)
