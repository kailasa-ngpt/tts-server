from typing import Optional
from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: Optional[str] = Field(None, description="Language code, e.g., 'en'")
    voice: Optional[str] = Field(None, description="Named voice or preset if backend supports it")
    rate: Optional[float] = Field(None, description="Speech rate multiplier (backend-specific)")
    pitch: Optional[float] = Field(None, description="Pitch adjustment (backend-specific)")


class TTSResponse(BaseModel):
    sample_rate: int
    num_samples: int
    format: str = "wav"

