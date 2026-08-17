import base64

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.speech import synthesize_speech, transcribe_audio


router = APIRouter(tags=["speech"])
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(min_length=2, max_length=35)


@router.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...), language: str = Form(...)):
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio must be 10 MB or smaller")
    try:
        transcript = transcribe_audio(content, language, audio.content_type or "application/octet-stream")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"transcript": transcript}


@router.post("/text-to-speech")
def text_to_speech(request: TextToSpeechRequest):
    try:
        audio = synthesize_speech(request.text, request.language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"audio": base64.b64encode(audio).decode("ascii"), "mimeType": "audio/mpeg"}
