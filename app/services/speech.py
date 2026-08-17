import base64
import os

import httpx
from dotenv import load_dotenv


load_dotenv()

GOOGLE_SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
REQUEST_TIMEOUT_SECONDS = 60.0


def _api_key() -> str:
    key = os.getenv("GOOGLE_SPEECH_API_KEY") or os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_SPEECH_API_KEY or GOOGLE_TRANSLATE_API_KEY is missing")
    return key


def _language_code(language: str) -> str:
    value = language.strip()
    aliases = {
        "arabic": "ar-SA", "ar": "ar-SA",
        "english": "en-US", "en": "en-US",
        "french": "fr-FR", "fr": "fr-FR",
        "german": "de-DE", "de": "de-DE",
        "spanish": "es-ES", "es": "es-ES",
        "japanese": "ja-JP", "ja": "ja-JP",
    }
    return aliases.get(value.casefold(), value)


def transcribe_audio(content: bytes, language: str, mime_type: str) -> str:
    if not content:
        raise ValueError("Audio file is empty")

    config: dict[str, object] = {
        "languageCode": _language_code(language),
        "enableAutomaticPunctuation": True,
        "model": "latest_short",
    }
    normalized_mime = mime_type.split(";", 1)[0].casefold()
    encoding = {
        "audio/webm": "WEBM_OPUS",
        "audio/ogg": "OGG_OPUS",
        "audio/flac": "FLAC",
        "audio/wav": "LINEAR16",
        "audio/x-wav": "LINEAR16",
    }.get(normalized_mime)
    if encoding:
        config["encoding"] = encoding
    if normalized_mime in {"audio/webm", "audio/ogg"}:
        # MediaRecorder encodes Opus at 48 kHz. Google requires this value
        # explicitly for WEBM_OPUS/OGG_OPUS instead of accepting zero/omitted.
        config["sampleRateHertz"] = 48_000

    response = httpx.post(
        GOOGLE_SPEECH_URL,
        params={"key": _api_key()},
        json={
            "config": config,
            "audio": {"content": base64.b64encode(content).decode("ascii")},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    _raise_google_error(response, "Speech-to-Text")

    results = response.json().get("results", [])
    transcript = " ".join(
        item["alternatives"][0]["transcript"]
        for item in results
        if item.get("alternatives")
    ).strip()
    if not transcript:
        raise RuntimeError("No speech could be recognized in the recording")
    return transcript


def synthesize_speech(text: str, language: str) -> bytes:
    if not text.strip():
        raise ValueError("Text is empty")

    response = httpx.post(
        GOOGLE_TTS_URL,
        params={"key": _api_key()},
        json={
            "input": {"text": text},
            "voice": {"languageCode": _language_code(language)},
            "audioConfig": {"audioEncoding": "MP3"},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    _raise_google_error(response, "Text-to-Speech")

    audio_content = response.json().get("audioContent")
    if not audio_content:
        raise RuntimeError("Google Text-to-Speech returned no audio")
    return base64.b64decode(audio_content)


def _raise_google_error(response: httpx.Response, service: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            message = response.json()["error"]["message"]
        except (KeyError, TypeError, ValueError):
            message = response.text or str(exc)
        raise RuntimeError(f"Google {service} API error: {message}") from exc
