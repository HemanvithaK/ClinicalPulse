import os
import tempfile
import wave
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

MODEL_SIZE = "base"  # options: tiny, base, small, medium, large-v3


class SpeechToText:
    def __init__(self):
        print("Loading Whisper model...")
        self.model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",  # efficient on CPU
        )
        print(f"Whisper {MODEL_SIZE} loaded")

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file and return the text."""
        print(f"Transcribing: {audio_path}")
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
        )
        text = " ".join([seg.text.strip() for seg in segments])
        print(f"Transcribed: {text}")
        return text.strip()

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw audio bytes (from WebSocket stream)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(f, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_bytes)

        try:
            return self.transcribe_file(tmp_path)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    stt = SpeechToText()
    print("STT module loaded successfully")
    print("Ready to transcribe audio files or bytes")