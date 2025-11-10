import io
import wave
import struct
import math

class TTSService:
    def __init__(self, url, route, voice):
        self.url = url
        self.route = route
        self.voice = voice
    
    def synthesize(self, text):
        """Generate a simple sine wave audio for the given text"""
        # Generate a 2-second sine wave at 440 Hz as a placeholder
        sample_rate = 22050
        duration = 2.0
        frequency = 440.0

        frames = []
        for i in range(int(sample_rate * duration)):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames.append(struct.pack('<h', value))

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b''.join(frames))

        wav_buffer.seek(0)
        return wav_buffer.getvalue()
