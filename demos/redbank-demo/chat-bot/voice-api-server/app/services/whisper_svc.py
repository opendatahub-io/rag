# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Whisper speech-to-text transcription service.

This module provides a service wrapper for OpenAI-compatible Whisper API endpoints,
enabling audio transcription with performance tracking and error handling.
"""

import tempfile
import os
import time
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

    def __init__(
        self, model_name: str = "base", whisper_url: str = "http://localhost:8080/v1"
    ):
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
        self.client = OpenAI(base_url=self.whisper_url, api_key="fake")

    def transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        """
        Transcribe audio data to text using the Whisper model.

        Processes raw audio bytes by converting them to WAV format and sending
        to the Whisper API endpoint. Supports multiple input formats: mp3, mp4,
        mpeg, mpga, m4a, wav, webm, flac, ogg.

        Args:
            audio_data: Raw audio file bytes in any supported format.
                Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac, ogg.
                The audio will be automatically converted to Whisper-compatible
                format (16-bit PCM, mono, 16kHz).

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

        try:
            # Create a temporary file and write audio data to it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_data)

            # Open the file in read mode for the API
            with open(temp_filename, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model_name,  # match your deployed model name
                    file=audio_file,
                    language="en",
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
