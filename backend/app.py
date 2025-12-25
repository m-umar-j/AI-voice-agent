from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import save
import os
from elevenlabs.core.api_error import ApiError
from requests.exceptions import HTTPError
from typing import Optional
from pydantic import BaseModel
import logging
import base64
load_dotenv()
app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptResponse(BaseModel):
    text: str
    full_transcript: dict

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "JBFqnCBsd6RMkjVDRZzb"
    model_id: Optional[str] = "eleven_multilingual_v2"

class TTSResponse(BaseModel):
    audio_bytes: bytes
    audio_b64: str
    size_bytes: int
class TTSMetadataResponse(BaseModel):
    audio_b64: str      # Only base64 string (JSON serializable)
    size_bytes: int
    duration_estimate: float

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

@app.post("/get-transcript", response_model=TranscriptResponse)
def get_transcript(audio_file: UploadFile = File(...)):
    """STT: Blocks until transcription complete"""
    # if not audio_file.content_type.startswith('audio/'):
    #     raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        logger.info(f"Transcribing audio: {audio_file.filename}, size: {audio_file.size}")
        
        transcription = client.speech_to_text.convert(
            file=audio_file.file,
            model_id="scribe_v1",
            language_code="eng",
            diarize=True,
            tag_audio_events=True
        )
        
        logger.info("Transcription successful")
        return TranscriptResponse(
            text=transcription.text,
            full_transcript=transcription.model_dump()
        )
    
    except ApiError as e:
        logger.error(f"ElevenLabs API error: {e.status_code} - {e.body}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e.body.get('detail', str(e))}")
    
    except Exception as e:
        logger.error(f"Unexpected STT error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/text-to-speech", response_model=TTSMetadataResponse)
def text_to_speech(request: TTSRequest):
    """TTS: Blocks until audio fully generated"""
    try:
        logger.info(f"Generating TTS for text length: {len(request.text)}")
        
        audio_generator = client.text_to_speech.convert(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id,
            output_format="mp3_44100_128",
        )
        audio_bytes = b''.join(audio_generator)
        
        logger.info(f"TTS generated: {len(audio_bytes)} bytes")
        
        return TTSMetadataResponse(
            audio_b64=base64.b64encode(audio_bytes).decode('utf-8'),
            size_bytes=len(audio_bytes),
            duration_estimate=len(request.text) / 150.0  # ~150 wpm estimate
        )
    
    except ApiError as e:
        logger.error(f"ElevenLabs TTS API error: {e.status_code} - {e.body}")
        if e.status_code == 402:
            raise HTTPException(status_code=402, detail="Insufficient credits")
        raise HTTPException(status_code=400, detail=f"TTS failed: {e.body.get('detail', str(e))}")
    
    except Exception as e:
        logger.error(f"Unexpected TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.post("/text-to-speech/download")
def text_to_speech_download(request: TTSRequest):
    try:
        audio_generator = client.text_to_speech.convert(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model_id,
            output_format="mp3_44100_128",
        )
        audio_bytes = b''.join(audio_generator)
        
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))