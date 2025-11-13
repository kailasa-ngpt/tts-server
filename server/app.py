import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .models import TTSRequest, TTSResponse
from .inference import load_model, audio_to_wav_bytes


logger = logging.getLogger("tts-server")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

app = FastAPI(title="TTS Inference Server", version="0.1.0")

# CORS for browser usage; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global backend instance
backend = None


@app.on_event("startup")
def _startup() -> None:
    global backend
    logger.info("Loading TTS model backend...")
    try:
        backend = load_model()
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        # Keep starting; requests will fail with 503 until resolved


@app.get("/health")
def health():
    status = "ready" if backend is not None else "loading_failed"
    return {"status": status}


@app.post("/tts")
def tts_json(body: TTSRequest):
    if backend is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        audio, sr = backend.synthesize(
            text=body.text,
            language=body.language,
            voice=body.voice,
            speaker_wav_bytes=None,
        )
        wav_bytes = audio_to_wav_bytes(audio, sr)
        # Return raw WAV bytes
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/tts failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts-multipart")
async def tts_multipart(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    speaker_wav: Optional[UploadFile] = File(None),
):
    if backend is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    speaker_bytes: Optional[bytes] = None
    if speaker_wav is not None:
        speaker_bytes = await speaker_wav.read()

    try:
        audio, sr = backend.synthesize(
            text=text,
            language=language,
            voice=voice,
            speaker_wav_bytes=speaker_bytes,
        )
        wav_bytes = audio_to_wav_bytes(audio, sr)
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/tts-multipart failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
