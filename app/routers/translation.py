from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.translator import (
    translate_json_values,
    translate_text,
)

router = APIRouter()


class TranslationRequest(BaseModel):
    text: str
    target: str
    source: str | None = None


class JsonTranslationRequest(BaseModel):
    data: Any
    target: str
    source: str | None = None
    ignore_keys: list[str] = Field(default_factory=list)


@router.post("/translate")
def translate(request: TranslationRequest):

    result = translate_text(
        text=request.text,
        target=request.target,
        source=request.source,
    )

    return {
        "translation": result
    }


@router.post("/translate-json")
def translate_json(request: JsonTranslationRequest):

    translated = translate_json_values(
        obj=request.data,
        target=request.target,
        source=request.source,
        ignore_keys=set(request.ignore_keys),
    )

    return {
        "data": translated
    }
