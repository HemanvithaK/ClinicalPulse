import os
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class TextToSpeech:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.voice = "alloy"  # options: alloy, echo, fable, onyx, nova, shimmer
        print("TTS module ready")

    def synthesize(self, text: str, output_path: str = None) -> bytes:
        """Convert text to speech, return audio bytes."""
        print(f"Synthesizing {len(text)} chars...")

        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text,
        )

        audio_bytes = response.content

        if output_path:
            Path(output_path).write_bytes(audio_bytes)
            print(f"Saved to {output_path}")

        return audio_bytes

    def synthesize_streaming(self, text: str):
        """Stream audio chunks for lower latency."""
        with self.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=self.voice,
            input=text,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                yield chunk


if __name__ == "__main__":
    tts = TextToSpeech()

    test_text = (
        "ClinicalPulse found 3 relevant trials. "
        "The top result is a Phase 3 breast cancer trial "
        "using tamoxifen, enrolled 6000 patients, "
        "conducted by the Institute of Cancer Research in the UK."
    )

    audio = tts.synthesize(test_text, output_path="test_output.mp3")
    print(f"Generated {len(audio)} bytes of audio")
    print("Open test_output.mp3 to hear the result")