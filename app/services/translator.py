import copy
from typing import Any

from lingua import Language, LanguageDetectorBuilder
import torch

_model = None
_tokenizer = None
_device = None


def get_translation_model():

    global _model
    global _tokenizer
    global _device

    if _model is None:

        from app.models.translator_model import (
            model,
            tokenizer,
            device
        )

        _model = model
        _tokenizer = tokenizer
        _device = device

    return _model, _tokenizer, _device


LANGUAGES = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Arabic": "arb_Arab",
    "Japanese": "jpn_Jpan",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Chinese": "zho_Hans",
}

# Reviewed translations for short catalogue labels that lack enough context for
# reliable machine translation. Keys are normalized by _glossary_translation.
GLOSSARY: dict[tuple[str, str], dict[str, str]] = {
    ("English", "Arabic"): {
        "pet supplies": "مستلزمات الحيوانات الأليفة",
        "office supplies": "مستلزمات مكتبية",
    },
}

LINGUA_TO_NAME = {
    Language.ENGLISH: "English",
    Language.FRENCH: "French",
    Language.ARABIC: "Arabic",
    Language.JAPANESE: "Japanese",
    Language.GERMAN: "German",
    Language.SPANISH: "Spanish",
    Language.CHINESE: "Chinese",
}

# Max number of strings sent to model.generate() in one batch.
# Tune based on GPU memory - lower if you hit OOM, raise if you have headroom.
BATCH_SIZE = 32
# Simple in-memory cache: (text, source, target) -> translation.
# Countries/static data barely change, so this avoids re-translating
# the same strings on every request. Swap for Redis if you need it
# to survive restarts or be shared across processes.
_translation_cache: dict[tuple[str, str, str], str] = {}


detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.FRENCH,
    Language.ARABIC,
    Language.JAPANESE,
    Language.GERMAN,
    Language.SPANISH,
    Language.CHINESE,
).build()


def detect_language(text: str) -> str:
    language = detector.detect_language_of(text)

    if language is None:
        return "English"

    return LINGUA_TO_NAME.get(language, "English")


def _glossary_translation(
    text: str,
    source: str,
    target: str,
) -> str | None:
    """Return a reviewed exact-match translation, if one is available."""

    normalized = " ".join(text.casefold().split())
    return GLOSSARY.get((source, target), {}).get(normalized)


def translate_text(
    text: str,
    target: str,
    source: str | None = None,
) -> str:
    """Single-string translation. Kept for the /translate endpoint."""

    if not isinstance(text, str):
        return text

    if not text.strip():
        return text

    if source is None:
        source = detect_language(text)

    if source == target:
        return text

    return _translate_batch_uncached([text], source, target)[0]


def _translate_batch_uncached(
    texts: list[str],
    source: str,
    target: str,
) -> list[str]:
    """Runs a single batched forward pass through the model."""

    if not texts:
        return []

    results: list[str | None] = [None] * len(texts)
    model_texts: list[str] = []
    model_indexes: list[int] = []

    for index, text in enumerate(texts):
        glossary_value = _glossary_translation(text, source, target)
        if glossary_value is not None:
            results[index] = glossary_value
        else:
            model_indexes.append(index)
            model_texts.append(text)

    if not model_texts:
        return results  # type: ignore[return-value]

    model, tokenizer, device = get_translation_model()


    tokenizer.src_lang = LANGUAGES[source]


    inputs = tokenizer(
        model_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )


    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                LANGUAGES[target]
            ),
            num_beams=5,
            do_sample=False,
            max_new_tokens=512,
            early_stopping=True,
        )

    model_translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    for index, translation in zip(model_indexes, model_translations):
        results[index] = translation

    return results  # type: ignore[return-value]


def translate_texts_batch(
    texts: list[str],
    source: str,
    target: str,
) -> list[str]:
    """
    Translates a list of strings, using the cache where possible and
    only running the model on cache misses, chunked to BATCH_SIZE.
    """

    if source == target:
        return texts

    results: list[str | None] = [None] * len(texts)
    misses: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        cached = _translation_cache.get((text, source, target))
        if cached is not None:
            results[i] = cached
        else:
            misses.append((i, text))

    for start in range(0, len(misses), BATCH_SIZE):
        chunk = misses[start : start + BATCH_SIZE]
        chunk_texts = [t for _, t in chunk]

        translated = _translate_batch_uncached(chunk_texts, source, target)

        for (i, original_text), translation in zip(chunk, translated):
            results[i] = translation
            _translation_cache[(original_text, source, target)] = translation

    return results  # type: ignore[return-value]


def _collect_strings(obj: Any, strings: list[str]):
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_strings(value, strings)

    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, strings)

    elif isinstance(obj, str):
        if obj.strip():
            strings.append(obj)


def _collect_paths(
    obj: Any,
    path: list,
    paths: list[list],
    strings: list[str],
    ignore_keys: set[str],
):
    """Walks the structure, recording a path + string for every
    translatable leaf so we can batch-translate then reassemble."""

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ignore_keys:
                continue
            _collect_paths(value, path + [key], paths, strings, ignore_keys)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _collect_paths(item, path + [i], paths, strings, ignore_keys)

    elif isinstance(obj, str):
        if obj.strip():
            paths.append(path)
            strings.append(obj)


def _set_path(obj: Any, path: list, value: Any):
    target = obj
    for p in path[:-1]:
        target = target[p]
    target[path[-1]] = value


def translate_json_values(
    obj: Any,
    target: str,
    ignore_keys: set[str],
    source: str | None = None,
):
    """
    Translate only JSON string values, in a single batched model call
    per request (instead of one call per string).

    Keys and ignored keys remain unchanged.
    If source is None, it's detected once from a sample of the strings.
    """

    if source is None:
        sample_strings: list[str] = []
        _collect_strings(obj, sample_strings)
        sample = " ".join(sample_strings[:10])
        source = detect_language(sample) if sample.strip() else "English"

    result = copy.deepcopy(obj)

    paths: list[list] = []
    strings: list[str] = []
    _collect_paths(result, [], paths, strings, ignore_keys)

    if not strings:
        return result

    translated = translate_texts_batch(strings, source, target)

    for path, value in zip(paths, translated):
        _set_path(result, path, value)

    return result
