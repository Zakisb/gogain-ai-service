"""
Audio transcription endpoint compatible with the legacy virtual coach contract.

This converts uploaded audio to text using OpenAI's transcription API and returns
the transcript so the frontend can forward it to the chat workflow.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional
import json

# ✅ Load .env BEFORE importing config
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# ✅ Now safe to import config — vars will be populated
from app.core.config import OPENAI_API_KEY, TRANSCRIPTION_MODEL, TRANSCRIPTION_UPLOAD_DIR
from app.api.chat import (
    VirtualCoachChatRequest,
    handle_virtual_coach_chat,
    emit_vc_event,
)

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI

# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
router = APIRouter()

_upload_dir = Path(TRANSCRIPTION_UPLOAD_DIR)
if not _upload_dir.is_absolute():
    _upload_dir = _PROJECT_ROOT / _upload_dir


def _ensure_upload_dir() -> None:
    try:
        _upload_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # defensive path setup
        logger.error("❌ Failed to create upload directory '%s': %s", _upload_dir, exc)
        raise HTTPException(status_code=500, detail="Unable to prepare upload directory")


_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY not configured or empty.")
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        try:
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized successfully.")
        except Exception as exc:  # defensive initialization
            logger.error("❌ Failed to initialize OpenAI client: %s", exc)
            raise HTTPException(status_code=500, detail="Unable to initialize transcription client")
    return _openai_client


@router.post("/virtual-coach/transcribe-audio")
async def transcribe_audio(
    audio: Optional[UploadFile] = File(default=None),
    file: Optional[UploadFile] = File(default=None),
    request: Request = None,
) -> Any:
    """
    Transcribe an uploaded audio file to text using OpenAI Whisper-compatible models.

    Mirrors the legacy Node backend contract:
        - Expects multipart/form-data with field name 'audio' or 'file'
        - Returns JSON: { "transcription": "<text>" }
    """

    # Accept either bound UploadFile or fallback to manual parsing
    form_data = None
    upload: Optional[UploadFile] = audio or file
    if upload is None and request is not None:
        form_data = await request.form()
        candidate = form_data.get("audio") or form_data.get("file")
        upload = candidate if isinstance(candidate, UploadFile) else None

    if upload is None or upload.filename is None:
        raise HTTPException(status_code=400, detail="No audio file uploaded")

    _ensure_upload_dir()

    saved_path: Optional[Path] = None
    try:
        saved_path = await _persist_upload(upload)
        transcription_text = await _transcribe_file(saved_path)
        logger.info("✅ Transcribed audio file '%s' (%s)", upload.filename, saved_path.name)

        # Determine if we should auto-forward to the chat workflow
        user_id: Optional[str] = None
        conversation_history = []
        if request is not None:
            try:
                if form_data is None:
                    form_data = await request.form()
                user_id = form_data.get("userId") or form_data.get("user_id")
                history_raw = form_data.get("conversationHistory")
                if history_raw:
                    try:
                        conversation_history = json.loads(history_raw)
                    except Exception:
                        conversation_history = []
            except Exception:
                user_id = None
                conversation_history = []

        if user_id:
            vc_request = VirtualCoachChatRequest(
                userId=user_id,
                message=transcription_text,
                conversationHistory=conversation_history or [],
            )
            chat_response = await handle_virtual_coach_chat(vc_request)

            body_iterator = chat_response.body_iterator

            async def combined_stream():
                yield emit_vc_event("transcription", transcription_text)
                if body_iterator is not None:
                    async for chunk in body_iterator:
                        yield chunk

            headers_dict = dict(chat_response.headers.items()) if chat_response.headers else {}
            return StreamingResponse(
                combined_stream(),
                status_code=chat_response.status_code,
                media_type=chat_response.media_type,
                headers=headers_dict,
                background=chat_response.background,
            )

        # Legacy compatibility: return transcript when no user id is supplied
        return {"transcription": transcription_text}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("❌ Error while transcribing audio: %s", exc)
        raise HTTPException(status_code=500, detail="An error occurred while transcribing")
    finally:
        if saved_path:
            try:
                os.remove(saved_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("⚠️ Failed to remove temporary audio file '%s': %s", saved_path, exc)


async def _persist_upload(audio: UploadFile) -> Path:
    """Persist the incoming upload to disk and return the path."""
    original_suffix = Path(audio.filename).suffix or ".wav"
    timestamp = int(time.time() * 1000)
    safe_suffix = original_suffix if original_suffix.startswith(".") else f".{original_suffix}"
    filename = f"audio-{timestamp}{safe_suffix}"
    destination = _upload_dir / filename

    try:
        with destination.open("wb") as buffer:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    finally:
        await audio.close()

    return destination


async def _transcribe_file(path: Path) -> str:
    """Call OpenAI's transcription API on the provided file path."""
    client = _get_openai_client()

    def _call_openai() -> Any:
        with path.open("rb") as audio_file:
            return client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
            )

    response = await asyncio.to_thread(_call_openai)
    text = getattr(response, "text", None)
    if text is None and isinstance(response, dict):
        text = response.get("text")

    if not text:
        logger.warning("⚠️ Transcription returned empty text for file '%s'", path.name)
        return ""

    return text.strip()


__all__ = ["router"]
