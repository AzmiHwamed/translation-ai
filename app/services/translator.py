import copy
import os
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

# Google Cloud Translation uses ISO-style language codes.
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Arabic": "ar",
    "Japanese": "ja",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh-CN",
}

GLOSSARY: dict[tuple[str, str], dict[str, str]] = {
    ("English", "Arabic"): {
        "pet supplies": "مستلزمات الحيوانات الأليفة",
        "office supplies": "مستلزمات مكتبية",
    },
}

# The API allows multiple strings in one request. Keeping batches below 30K
# characters avoids oversized requests while still making JSON translation fast.
MAX_BATCH_STRINGS = 100
MAX_BATCH_CHARACTERS = 25_000
REQUEST_TIMEOUT_SECONDS = 30.0

_translation_cache: dict[tuple[str, str | None, str], str] = {}


def _api_key() -> str:
    key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_TRANSLATE_API_KEY is missing. Add it to the .env file."
        )
    return key


def _language_code(language: str | None) -> str | None:
    if language is None:
        return None

    value = language.strip()
    if not value:
        return None

    # Continue accepting the old API's names, but also accept Google codes.
    return LANGUAGES.get(value, LANGUAGES.get(value.title(), value))


def _glossary_translation(
    text: str,
    source: str | None,
    target: str,
) -> str | None:
    if source is None:
        return None
    normalized = " ".join(text.casefold().split())
    source_name = source.title()
    target_name = target.title()
    return GLOSSARY.get((source_name, target_name), {}).get(normalized)


def _google_translate_batch(
    texts: list[str],
    source: str | None,
    target: str,
) -> list[str]:
    if not texts:
        return []

    payload: dict[str, Any] = {
        "q": texts,
        "target": _language_code(target),
        "format": "text",
        "model": "nmt",
    }
    source_code = _language_code(source)
    if source_code:
        payload["source"] = source_code

    try:
        response = httpx.post(
            GOOGLE_TRANSLATE_URL,
            params={"key": _api_key()},
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        translations = response.json()["data"]["translations"]
    except httpx.HTTPStatusError as exc:
        try:
            message = exc.response.json()["error"]["message"]
        except (KeyError, TypeError, ValueError):
            message = exc.response.text or str(exc)
        raise RuntimeError(f"Google Translation API error: {message}") from exc
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Google Translation API request failed: {exc}") from exc

    if len(translations) != len(texts):
        raise RuntimeError("Google Translation API returned an unexpected result count")

    return [item["translatedText"] for item in translations]


def _translate_batch_uncached(
    texts: list[str],
    source: str | None,
    target: str,
) -> list[str]:
    results: list[str | None] = [None] * len(texts)
    api_texts: list[str] = []
    api_indexes: list[int] = []

    for index, text in enumerate(texts):
        glossary_value = _glossary_translation(text, source, target)
        if glossary_value is not None:
            results[index] = glossary_value
        else:
            api_indexes.append(index)
            api_texts.append(text)

    for index, translated in zip(
        api_indexes, _google_translate_batch(api_texts, source, target)
    ):
        results[index] = translated

    return results  # type: ignore[return-value]


def _chunks(items: list[tuple[int, str]]):
    chunk: list[tuple[int, str]] = []
    characters = 0

    for item in items:
        item_characters = len(item[1])
        if chunk and (
            len(chunk) >= MAX_BATCH_STRINGS
            or characters + item_characters > MAX_BATCH_CHARACTERS
        ):
            yield chunk
            chunk = []
            characters = 0

        chunk.append(item)
        characters += item_characters

    if chunk:
        yield chunk


def translate_texts_batch(
    texts: list[str],
    source: str | None,
    target: str,
) -> list[str]:
    if source and _language_code(source) == _language_code(target):
        return texts

    results: list[str | None] = [None] * len(texts)
    misses: list[tuple[int, str]] = []

    for index, text in enumerate(texts):
        cache_key = (text, source, target)
        cached = _translation_cache.get(cache_key)
        if cached is None:
            misses.append((index, text))
        else:
            results[index] = cached

    for chunk in _chunks(misses):
        chunk_texts = [text for _, text in chunk]
        translated = _translate_batch_uncached(chunk_texts, source, target)

        for (index, original), value in zip(chunk, translated):
            results[index] = value
            _translation_cache[(original, source, target)] = value

    return results  # type: ignore[return-value]


def translate_text(
    text: str,
    target: str,
    source: str | None = None,
) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    return translate_texts_batch([text], source, target)[0]


def _collect_paths(
    obj: Any,
    path: list,
    paths: list[list],
    strings: list[str],
    ignore_keys: set[str],
):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in ignore_keys:
                _collect_paths(value, path + [key], paths, strings, ignore_keys)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            _collect_paths(item, path + [index], paths, strings, ignore_keys)
    elif isinstance(obj, str) and obj.strip():
        paths.append(path)
        strings.append(obj)


def _set_path(obj: Any, path: list, value: Any):
    target = obj
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def translate_json_values(
    obj: Any,
    target: str,
    ignore_keys: set[str],
    source: str | None = None,
):
    result = copy.deepcopy(obj)
    paths: list[list] = []
    strings: list[str] = []
    _collect_paths(result, [], paths, strings, ignore_keys)

    if not strings:
        return result

    # When source is omitted Google detects it per string automatically.
    translated = translate_texts_batch(strings, source, target)
    for path, value in zip(paths, translated):
        if path:
            _set_path(result, path, value)
        else:
            result = value

    return result
