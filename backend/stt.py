from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play, save
import os
import requests 
load_dotenv()

elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

# def stt(media_file):
#     pass

def tts():

    # pass
    audio = elevenlabs.text_to_speech.convert(
        text="My name is shahriyar mughal. I am Prime minister of poonch",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="pcm_16000",
    )

    save(audio, "output.wav")

# tts()


from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def transcribe_local_file(file_path: str):
    """
    Transcribe a local MP3 or WAV file using ElevenLabs Speech-to-Text API
    
    Args:
        file_path: Path to your local audio file (MP3, WAV, etc.)
    """
    with open(file_path, "rb") as audio_file:
        transcription = client.speech_to_text.convert(
            file=audio_file,  # ✅ Pass open file object, not path string
            model_id="scribe_v1",
            language_code="eng",
            # diarize=True,
            # tag_audio_events=True
        )
    return transcription.text

# Usage
result = transcribe_local_file("output.mp3")  # or "output.wav"
print(result)
