#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pymupdf as fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk

try:
    import cv2
except ImportError:  # pragma: no cover - handled in GUI
    cv2 = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
HISTORY_DIR = Path.home() / "Library" / "Application Support" / "Automatizador de Termos"
HISTORY_FILE = HISTORY_DIR / "historico.json"
NAME_DATABASE_FILE = APP_DIR / "nomes_brasileiros.json"
SUPPORTED = {".jpg", ".jpeg", ".png", ".pdf"}
VIDEO_SUPPORTED = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUTHORIZATION_SUPPORTED = {".pdf"} | VIDEO_SUPPORTED
DRIVE_DOWNLOAD_ATTEMPTS = 3
DRIVE_CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "video/quicktime": ".mov",
    "video/mp4": ".mp4",
    "video/x-m4v": ".m4v",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}
A4_SIZE = (1654, 2339)  # 200 dpi; good balance between legibility and file size.
OK_CONFIDENCE_THRESHOLD = 0.50
MIN_REVIEW_MATCH = 0.46
MIN_AUTOMATIC_MATCH = 0.50
MAX_OCR_REGION_EDGE = 1600
FAST_OCR_REGION_EDGE = 1200
MAX_PAGE_PROCESSING_EDGE = 2400
TOP_RIGHT_PHOTO_SAFE_BOTTOM = 0.135
VIDEO_TRANSCRIPTION_TIMEOUT = 45
WHISPER_MODEL_NAME = os.environ.get("TERMOS_WHISPER_MODEL", "base")
CAREFUL_WHISPER_MODEL_NAME = os.environ.get("TERMOS_WHISPER_CAREFUL_MODEL", "small")
VIDEO_NAME_TRANSCRIPTION_SECONDS = 14.0
VIDEO_CAREFUL_RETRY_SECONDS = 16.0
MAX_ROSTER_OCR_RECHECKS = 8
VIDEO_AUDIT_FRAME_INTERVAL = 3.0
VIDEO_AUDIT_MAX_FACES = 36
VIDEO_AUDIT_MAX_FRAME_SAMPLES = 54
VIDEO_AUDIT_TRANSCRIPTION_SECONDS: float | None = None
VIDEO_AUDIT_VISUAL_OCR_INTERVAL = 1.0
VIDEO_FOREGROUND_MIN_FACE_HEIGHT = 0.075
VIDEO_FOREGROUND_MIN_FACE_AREA = 0.0035
FACE_DETECTION_MODEL = APP_DIR / "models" / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNITION_MODEL = APP_DIR / "models" / "face_recognition_sface_2021dec.onnx"
FACE_STRONG_SIMILARITY = 0.50
FACE_REPEATED_SIMILARITY = 0.42
FACE_MIN_MARGIN = 0.10
VERIFY_NAMED_VIDEO_FILES = os.environ.get("TERMOS_VERIFY_NAMED_VIDEOS", "").strip() == "1"
OCR_COMMON_FIRST_NAME_LIMIT = 0
OCR_COMMON_SURNAME_LIMIT = 0
ACCEPTED_FIRST_NAME_DATABASE_LIMIT = 20000
FUZZY_FIRST_NAME_DATABASE_LIMIT = 6000
# O nome do arquivo é uma pista forte e um fallback, mas não substitui a
# leitura da faixa manuscrita. O atalho permanece disponível somente para
# diagnóstico explícito de desempenho.
FAST_TRUST_SOURCE_NAMED_TERMS = os.environ.get("TERMOS_SKIP_NAMED_TERM_OCR", "0").strip() == "1"
MAX_FOCUSED_NAME_REGIONS_WITHOUT_HINT = 1
PALETTE = {
    "bg": "#101010",
    "surface": "#171717",
    "surface_alt": "#202020",
    "surface_soft": "#29251A",
    "border": "#33302A",
    "ink": "#F7F3E8",
    "muted": "#AAA69A",
    "subtle": "#77736A",
    "accent": "#F5C542",
    "accent_hover": "#FFD766",
    "accent_soft": "#3A321A",
    "success": "#8FD694",
    "warning": "#F7B955",
    "danger": "#FF7A70",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .-")
    return text[:120] or "Termo sem nome"


def alpha_key(text: str) -> tuple[str, str]:
    return (normalize(text), text.casefold())


def label_from_photo_path(path: Path) -> str:
    return re.sub(r"[_-]+", " ", path.stem).strip() or path.stem


def person_identity(person: "PersonPhoto") -> tuple[str, str]:
    try:
        image_key = str(person.image_path.resolve())
    except OSError:
        image_key = str(person.image_path)
    return (normalize(person.label), image_key)


@dataclass
class PersonPhoto:
    label: str
    caption: str
    image_path: Path


@dataclass
class DriveTermFile:
    id: str
    path: str
    local_path: Path


@dataclass
class VideoFacePreview:
    timestamp: float
    index: int
    image_path: Path
    kind: str = "rosto"
    label: str = ""
    authorization_path: Path | None = None


@dataclass
class VideoAuditSuggestion:
    pdf_path: Path
    confidence: float
    reason: str


@dataclass
class VideoVisualEvidence:
    timestamp: float
    text: str


@dataclass
class AuthorizationFaceReference:
    person: PersonPhoto
    feature: np.ndarray
    source: str


@dataclass
class VideoAuditBatch:
    video_path: Path
    source_root: Path
    relative_path: Path
    output_dir: Path | None = None
    transcript: str = ""
    visual_evidence: list[VideoVisualEvidence] | None = None
    suggestions: list[VideoAuditSuggestion] | None = None
    face_previews: list[VideoFacePreview] | None = None
    sheet_path: Path | None = None
    copied_paths: list[Path] | None = None
    face_photos: dict[Path, Path] | None = None
    filled_photo_count: int = 0
    error: str = ""


@dataclass
class TermResult:
    source: Path
    page: Image.Image
    ocr_text: str
    model: str
    person: PersonPhoto | None
    confidence: float
    output_name: str
    manual_crop: bool = False
    photo_zoom: float = 1.0
    photo_offset_x: float = 0.0
    photo_offset_y: float = 0.0
    duplicate_note: str = ""
    duplicate_confirmed: bool = False
    original_page: Image.Image | None = None

    @property
    def status(self) -> str:
        if self.person is None:
            return "SEM FOTO"
        return "OK" if self.confidence >= OK_CONFIDENCE_THRESHOLD else "REVISAR"


def name_similarity(first: str, second: str) -> float:
    a, b = normalize(first), normalize(second)
    if not a or not b:
        return 0.0
    direct = SequenceMatcher(None, a, b).ratio()
    sorted_tokens = SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()
    return max(direct, sorted_tokens)


def meaningful_name(value: str) -> bool:
    clean = normalize(value)
    return len(clean) >= 3 and any(ch.isalpha() for ch in clean) and clean not in {
        "sem foto", "sem nome", "termo sem nome"
    }


NON_PERSON_CAPTIONS = {
    "nao quis assinar",
    "nao quer aparecer",
    "nao autorizou",
    "sem nome",
    "sem foto",
}


def load_history() -> list[dict]:
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def duplicate_notes(names: list[str], history: list[dict]) -> dict[int, str]:
    notes: dict[int, str] = {}
    for first_idx, first in enumerate(names):
        if not meaningful_name(first):
            continue
        for second_idx in range(first_idx + 1, len(names)):
            second = names[second_idx]
            if not meaningful_name(second):
                continue
            score = name_similarity(first, second)
            exact = normalize(first) == normalize(second)
            if exact or (min(len(normalize(first)), len(normalize(second))) >= 6 and score >= 0.90):
                notes.setdefault(first_idx, f"Também aparece neste lote: {second} ({score:.0%} parecido)")
                notes.setdefault(second_idx, f"Também aparece neste lote: {first} ({score:.0%} parecido)")

    for idx, name in enumerate(names):
        if idx in notes or not meaningful_name(name):
            continue
        best: tuple[float, dict] | None = None
        for item in history:
            old_name = str(item.get("name", ""))
            if not meaningful_name(old_name):
                continue
            score = name_similarity(name, old_name)
            exact = normalize(name) == normalize(old_name)
            if not exact and (min(len(normalize(name)), len(normalize(old_name))) < 6 or score < 0.90):
                continue
            if best is None or score > best[0]:
                best = (score, item)
        if best:
            score, item = best
            date = str(item.get("generated_at", ""))[:10]
            date_text = f" em {date[8:10]}/{date[5:7]}/{date[:4]}" if len(date) == 10 else ""
            notes[idx] = f"Já consta no histórico{date_text}: {item.get('name', name)} ({score:.0%} parecido)"
    return notes


def record_history(results: list[TermResult]) -> None:
    history = load_history()
    now = datetime.now().isoformat(timespec="seconds")
    for result in results:
        if meaningful_name(result.output_name):
            history.append({
                "name": result.output_name,
                "normalized_name": normalize(result.output_name),
                "generated_at": now,
                "source": result.source.name,
            })
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = HISTORY_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(HISTORY_FILE)


def split_caption(caption: str) -> list[str]:
    clean = re.sub(r"\s+", " ", caption).strip()
    if not clean or normalize(clean) in NON_PERSON_CAPTIONS:
        return []
    # Parenthetical aliases/descriptions stay with the name before them.
    base = re.sub(r"\s*\([^)]*\)\s*", "", clean).strip()
    parts = re.split(r"\s*[,;/&]\s*|\s+[Ee]\s+", base)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [clean]
    return parts


def find_chat_text(folder: Path) -> Path:
    files = list(folder.rglob("*.txt"))
    if not files:
        raise ValueError("O ZIP não contém o arquivo de texto da conversa.")
    return files[0]


def read_text_flex(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    return path.read_text(errors="replace")


def _whatsapp_captioned_media(
    zip_path: Path,
    extract_dir: Path,
    supported: set[str],
) -> list[tuple[str, Path]]:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    chat = read_text_flex(find_chat_text(extract_dir))
    lines = chat.splitlines()
    suffixes = "|".join(re.escape(suffix.lstrip(".")) for suffix in sorted(supported))
    attachment = re.compile(
        rf"([\w .()\-]+\.(?:{suffixes}))\s+\((?:arquivo anexado|file attached)\)(.*)$",
        re.IGNORECASE,
    )
    timestamp = re.compile(r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}.*?\s-\s")
    found: list[tuple[str, Path]] = []

    for idx, line in enumerate(lines):
        match = attachment.search(line.replace("\u200e", ""))
        if not match:
            continue
        media_name = Path(match.group(1).strip()).name
        caption_lines: list[str] = []
        inline_caption = match.group(2).strip(" \t-:·")
        if inline_caption:
            caption_lines.append(inline_caption)
        j = idx + 1
        while j < len(lines) and not timestamp.match(lines[j]):
            if lines[j].strip():
                caption_lines.append(lines[j].strip())
            j += 1
        caption = " ".join(caption_lines).strip()
        media_matches = list(extract_dir.rglob(media_name))
        if not caption or not media_matches:
            continue
        found.append((caption, media_matches[0]))
    return found


def parse_whatsapp_export(zip_path: Path, extract_dir: Path) -> list[PersonPhoto]:
    people: list[PersonPhoto] = []
    for caption, media_path in _whatsapp_captioned_media(
        zip_path,
        extract_dir,
        {".jpg", ".jpeg", ".png", ".heic"},
    ):
        for label in split_caption(caption):
            people.append(PersonPhoto(label=label, caption=caption, image_path=media_path))
    if not people:
        raise ValueError("Não encontrei fotografias com legenda no ZIP exportado.")
    return people


def export_whatsapp_videos(zip_path: Path, extract_dir: Path, destination: Path) -> list[Path]:
    """Exporta vídeos anexados usando a própria legenda como nome do arquivo."""
    media = _whatsapp_captioned_media(zip_path, extract_dir, VIDEO_SUPPORTED)
    if not media:
        raise ValueError("Não encontrei vídeos com legenda no ZIP exportado.")
    destination.mkdir(parents=True, exist_ok=True)
    used = {path for path in destination.iterdir() if path.is_file()}
    exported: list[Path] = []
    for caption, source in media:
        clean_caption = re.sub(r"\s+", " ", caption).strip()
        target = unique_path(
            destination / f"{safe_filename(clean_caption)}{source.suffix.lower()}",
            used,
        )
        shutil.copy2(source, target)
        exported.append(target)
    return exported


def people_from_folder(folder: Path) -> list[PersonPhoto]:
    people: list[PersonPhoto] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            label = label_from_photo_path(path)
            people.append(PersonPhoto(label=label, caption=label, image_path=path))
    return people


def video_files_from_folder(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and "autorizacao de imagem" not in {normalize(part) for part in path.parts}
        and (
            path.suffix.lower() in VIDEO_SUPPORTED
            or (not path.suffix and bool(downloaded_file_suffix(path, "", VIDEO_SUPPORTED)))
        )
    )


def speech_engine_name() -> str:
    try:
        import faster_whisper  # noqa: F401
        return f"Whisper local ({WHISPER_MODEL_NAME})"
    except ImportError:
        pass
    if sys.platform != "darwin":
        return ""
    try:
        import Foundation  # noqa: F401
        import Speech  # noqa: F401
        import objc  # noqa: F401
    except ImportError:
        return ""
    return "Transcrição local do macOS"


@lru_cache(maxsize=2)
def _whisper_model(model_name: str = WHISPER_MODEL_NAME):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _audio_for_whisper(path: Path, audio_path: Path, max_seconds: float | None = None) -> Path:
    try:
        import av
    except ImportError:
        return path

    container = av.open(str(path))
    try:
        audio_streams = list(container.streams.audio)
        if not audio_streams:
            return path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.unlink(missing_ok=True)
        stream = audio_streams[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        with wave.open(str(audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            for frame in container.decode(stream):
                if max_seconds and frame.time is not None and float(frame.time) > max_seconds:
                    break
                for resampled in resampler.resample(frame):
                    output.writeframes(resampled.to_ndarray().tobytes())
            for resampled in resampler.resample(None):
                output.writeframes(resampled.to_ndarray().tobytes())
        if audio_path.exists() and audio_path.stat().st_size > 44:
            return audio_path
        return path
    finally:
        container.close()


def _whisper_roster_prompt(roster_names: Iterable[str] | None) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for raw in roster_names or []:
        name = _canonical_person_name(raw)
        key = normalize(name)
        if not key or key in seen or not _valid_name_candidate(name):
            continue
        seen.add(key)
        names.append(name)
    if names:
        return (
            "Entrevista em português. A pessoa começa dizendo o próprio nome. "
            "Transcreva literalmente o nome falado e a autorização."
        )
    return (
        "A pessoa diz o próprio nome brasileiro no começo do vídeo. "
        "Transcreva literalmente o nome dito e a autorização de uso de imagem."
    )


def _faster_whisper_transcribe_video(
    path: Path,
    max_seconds: float | None = None,
    *,
    careful: bool = False,
    roster_names: Iterable[str] | None = None,
    name_prompt: bool = True,
) -> str:
    guided = bool(roster_names)
    model_name = CAREFUL_WHISPER_MODEL_NAME if careful or guided else WHISPER_MODEL_NAME
    model = _whisper_model(model_name)
    with tempfile.TemporaryDirectory(prefix="termos_whisper_audio_") as folder:
        media_path = _audio_for_whisper(path, Path(folder) / f"{safe_filename(path.stem)}.wav", max_seconds=max_seconds)
        roster_prompt = _whisper_roster_prompt(roster_names) if name_prompt else ""
        hotwords = ", ".join(
            dict.fromkeys(
                _canonical_person_name(name)
                for name in roster_names or []
                if _valid_name_candidate(_canonical_person_name(name))
            )
        )
        options = {
            "language": "pt",
            "beam_size": 5 if careful or guided else 1,
            "best_of": 5 if careful or guided else 1,
            "temperature": 0.0,
            # Os nomes estão logo no começo e alguns vídeos têm pausas curtas.
            # Sem VAD, não perdemos sílabas que pareciam silêncio no modelo base.
            "vad_filter": False if careful or guided else True,
            "condition_on_previous_text": False,
        }
        if roster_prompt:
            options["initial_prompt"] = roster_prompt
        if hotwords:
            # O faster-whisper soma o prompt inicial e as hotwords ao mesmo
            # limite de 448 tokens. Repetir a lista de nomes nos dois campos
            # pode deixar zero tokens para o áudio e abortar a transcrição.
            options["hotwords"] = hotwords[:1200]
        segments, _info = model.transcribe(str(media_path), **options)
    return re.sub(r"\s+", " ", " ".join(segment.text.strip() for segment in segments)).strip()


def _speech_usage_description_available() -> bool:
    try:
        import Foundation
    except ImportError:
        return False
    try:
        bundle = Foundation.NSBundle.mainBundle()
        return bool(bundle.objectForInfoDictionaryKey_("NSSpeechRecognitionUsageDescription"))
    except Exception:
        return False


def _speech_authorized(timeout: float = 10.0) -> bool:
    import Speech

    try:
        status = int(Speech.SFSpeechRecognizer.authorizationStatus())
    except Exception:
        return False
    if status == 3:
        return True
    if status in {1, 2}:
        return False
    if not _speech_usage_description_available():
        return False

    holder: dict[str, int] = {}
    done = threading.Event()

    def callback(new_status) -> None:
        try:
            holder["status"] = int(new_status)
        finally:
            done.set()

    Speech.SFSpeechRecognizer.requestAuthorization_(callback)
    done.wait(timeout)
    return holder.get("status") == 3


def _audio_for_speech(video_path: Path, audio_path: Path, timeout: float = 90.0) -> Path:
    try:
        import AVFoundation
        import Foundation
        import objc
    except ImportError:
        return video_path

    with objc.autorelease_pool():
        url = Foundation.NSURL.fileURLWithPath_(str(video_path))
        asset = AVFoundation.AVURLAsset.URLAssetWithURL_options_(url, None)
        preset = getattr(AVFoundation, "AVAssetExportPresetAppleM4A", "com.apple.m4a-audio")
        session = AVFoundation.AVAssetExportSession.exportSessionWithAsset_presetName_(asset, preset)
        if session is None:
            return video_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.unlink(missing_ok=True)
        session.setOutputURL_(Foundation.NSURL.fileURLWithPath_(str(audio_path)))
        session.setOutputFileType_(getattr(AVFoundation, "AVFileTypeAppleM4A", "com.apple.m4a-audio"))
        done = threading.Event()
        session.exportAsynchronouslyWithCompletionHandler_(lambda: done.set())
        if not done.wait(timeout):
            try:
                session.cancelExport()
            except Exception:
                pass
            return video_path
        completed = int(getattr(AVFoundation, "AVAssetExportSessionStatusCompleted", 3))
        try:
            status = int(session.status())
        except Exception:
            status = -1
        return audio_path if status == completed and audio_path.exists() and audio_path.stat().st_size else video_path


def _speech_transcribe_video(path: Path, timeout: float = VIDEO_TRANSCRIPTION_TIMEOUT) -> str:
    import Foundation
    import Speech
    import objc

    if not _speech_authorized():
        return ""

    with objc.autorelease_pool():
        locale = Foundation.NSLocale.localeWithLocaleIdentifier_("pt_BR")
        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if recognizer is None or not bool(recognizer.isAvailable()):
            return ""
        supports_local = bool(
            recognizer.supportsOnDeviceRecognition()
            if hasattr(recognizer, "supportsOnDeviceRecognition")
            else False
        )
        if not supports_local:
            return ""
        with tempfile.TemporaryDirectory(prefix="termos_video_audio_") as folder:
            speech_path = _audio_for_speech(path, Path(folder) / f"{safe_filename(path.stem)}.m4a")
            url = Foundation.NSURL.fileURLWithPath_(str(speech_path))
            request = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
            request.setShouldReportPartialResults_(False)
            if hasattr(request, "setRequiresOnDeviceRecognition_"):
                request.setRequiresOnDeviceRecognition_(True)

            state: dict[str, str] = {"text": ""}
            done = threading.Event()

            def handler(result, error) -> None:
                if result is not None:
                    try:
                        state["text"] = str(result.bestTranscription().formattedString())
                    except Exception:
                        pass
                if error is not None or (result is not None and bool(result.isFinal())):
                    done.set()

            task = recognizer.recognitionTaskWithRequest_resultHandler_(request, handler)
            if not done.wait(timeout):
                try:
                    task.cancel()
                except Exception:
                    pass
            return state["text"].strip()


VIDEO_NAME_PREFIXES = (
    r"meu\s+nome\s+completo\s+(?:e|eh|é)",
    r"meu\s+nome\s+(?:e|eh|é)",
    r"me\s+chamo",
    r"eu\s+me\s+chamo",
    r"eu\s+sou",
    r"sou",
)

VIDEO_NAME_STOPS = (
    "autorizo", "autorizacao", "autorização", "concordo", "aceito", "permito",
    "declaro", "cedo", "gravar", "gravacao", "gravação", "imagem", "video",
    "vídeo", "foto", "fotos", "som", "uso", "minha", "meu cpf", "cpf",
    "voce", "você", "fala", "fale", "residente", "domiciliado", "domiciliada",
    "nascido", "nascida", "tenho", "trabalho",
)

VIDEO_FILLER_WORDS = {
    "meu", "minha", "nome", "e", "eh", "sou", "eu", "me", "chamo", "aqui",
    "estou", "pra", "para", "bom", "boa", "dia", "tarde", "noite",
    "o", "a", "seu", "sua", "qual", "completo", "completa", "por", "favor", "uma", "um",
    "umas", "uns", "pessoa", "pessoas", "parte",
    "todo", "toda", "todos", "todas", "termo", "uso", "imagem", "som",
    "autorizo", "autorizacao", "autorização", "concordo", "aceito", "permito",
    "declaro", "cedo", "direito", "direitos", "cedente", "cessionario",
    "cessionário", "clausula", "cláusula", "lgpd", "dados", "pessoais",
}

HARD_NON_NAME_WORDS = {
    "aparece", "atitude", "brasileiros", "brasileiras", "caia", "caixa",
    "casta", "cedente", "cessao", "cessão", "chira", "com", "cortador",
    "cortadora", "cortadores", "cortadoras", "cotador", "cotadora", "dairro",
    "deus", "direito", "direitos", "eaf", "fazer", "faixa", "lhe", "mais", "manter",
    "mantem", "mantêm", "mexer", "modafrates", "motei", "mudei", "outro",
    "outra", "pagar", "pago", "remundo", "sem", "sobre", "voz",
    "desemp", "desidenta", "domiciado", "domiciliario", "domiriliario",
    "endere", "esidente", "foniel", "grostinto", "instumento",
    "mpradora", "mporadora", "paradora", "paradores", "patador",
    "patadora", "patadoras", "patadores", "porador", "poradora",
    "poradoras", "poradores", "potador", "potadora", "potadoras",
    "potadores", "prosonto", "residente", "tesidente",
    "teseup", "tumento", "unb",
}

FORM_INTRO_NOISE_WORDS = {
    "eu", "ou", "pelo", "polo", "pojo", "palo", "presente", "presento",
    "prosente", "prosonte", "presonto", "instrumento", "instrumendo",
    "instrumonto", "instumento", "prosonto", "tumento", "umento", "termo",
}

VIDEO_SPOKEN_NON_NAME_WORDS = {
    "adicia", "aquele", "aquela", "ao", "assaturismo", "automacao",
    "automação", "barbois", "barbóis", "cabri", "conhecimento", "falar",
    "fica", "fui", "ja", "já", "naquele", "naquela", "posso", "sair",
    "so", "só", "volante", "analista", "apresentador", "apresentadora",
    "enfermeiro", "enfermeira", "especialista", "fotografo", "fotógrafo",
    "fotografa", "fotógrafa", "mergulhador", "mergulhadora", "motorista",
    "operador", "operadora", "tecnico", "técnico", "tecnica", "técnica",
}

NON_NAME_FIRST_TOKENS = {
    "atua", "conta", "digital", "ordem", "privado", "producao", "produção",
    "sala", "sede", "sua",
}

KNOWN_FIRST_NAMES = {
    "adalberto", "alex", "alice", "amanda", "ana", "anderson", "andre", "andressa",
    "anita", "antonio", "antonia", "ariana", "camila", "claudia", "claudio", "daniel",
    "daniele", "davi", "david", "eliana", "erick", "gabriel", "gabriela", "elio",
    "helio", "higor", "hugo", "iago", "italo", "jessica", "joao", "joel",
    "jose", "lucas", "luciano", "marcelo", "maria", "matheus", "pedro", "rafael",
    "raimundo", "raymundo", "roger", "rui", "vitoria",
}

COMMON_SURNAME_FALLBACK = {
    "albuquerque", "almeida", "alves", "amaral", "andrade", "araujo",
    "assis", "azevedo", "barbosa", "barreto", "barros", "batista", "brito",
    "campos", "cardoso", "carvalho", "castro", "cavalcante", "coelho",
    "correia", "costa", "cruz", "cunha", "dias", "duarte", "farias",
    "fernandes", "ferreira", "freitas", "gomes", "goncalves", "lima",
    "lopes", "machado", "marques", "martins", "melo", "mendes", "monteiro",
    "moraes", "moreira", "mota", "moura", "nascimento", "nunes", "oliveira",
    "pereira", "pinheiro", "pires", "prates", "ramos", "reis", "ribeiro",
    "rocha", "rodrigues", "santos", "silva", "soares", "sousa", "souza",
    "teixeira", "valente", "vieira",
}


@lru_cache(maxsize=1)
def _name_database() -> tuple[tuple[str, ...], tuple[str, ...]]:
    first_names: list[str] = []
    surnames: list[str] = []

    def add_unique(destination: list[str], seen: set[str], value: str) -> None:
        norm = normalize(value)
        if not norm or norm in seen:
            return
        if any(part.isdigit() for part in norm.split()):
            return
        if len(norm.replace(" ", "")) < 3:
            return
        seen.add(norm)
        destination.append(norm)

    first_seen: set[str] = set()
    surname_seen: set[str] = set()
    try:
        data = json.loads(NAME_DATABASE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    for value in data.get("first_names", []) if isinstance(data, dict) else []:
        # Composite names in public datasets are useful to Vision as words, but
        # the validator needs the first token as the first-name anchor.
        parts = normalize(str(value)).split()
        if parts:
            add_unique(first_names, first_seen, parts[0])
    for value in data.get("surnames", []) if isinstance(data, dict) else []:
        for part in normalize(str(value)).split():
            if part in {"de", "da", "do", "dos", "das"}:
                continue
            add_unique(surnames, surname_seen, part)
    for value in sorted(KNOWN_FIRST_NAMES):
        add_unique(first_names, first_seen, value)
    for value in sorted(COMMON_SURNAME_FALLBACK):
        add_unique(surnames, surname_seen, value)
    return tuple(first_names), tuple(surnames)


@lru_cache(maxsize=1)
def known_first_names() -> frozenset[str]:
    return frozenset(_name_database()[0])


@lru_cache(maxsize=1)
def common_surnames() -> frozenset[str]:
    return frozenset(_name_database()[1])


@lru_cache(maxsize=1)
def curated_first_names() -> frozenset[str]:
    return frozenset(normalize(value) for value in KNOWN_FIRST_NAMES)


@lru_cache(maxsize=1)
def accepted_first_names() -> frozenset[str]:
    first_names = set(_name_database()[0][:ACCEPTED_FIRST_NAME_DATABASE_LIMIT])
    first_names.update(curated_first_names())
    return frozenset(name for name in first_names if name not in NON_NAME_FIRST_TOKENS)


@lru_cache(maxsize=1)
def fuzzy_first_names() -> tuple[str, ...]:
    first_names = list(_name_database()[0][:FUZZY_FIRST_NAME_DATABASE_LIMIT])
    first_names.extend(curated_first_names())
    seen: set[str] = set()
    unique: list[str] = []
    for name in first_names:
        if name and name not in seen and name not in NON_NAME_FIRST_TOKENS:
            seen.add(name)
            unique.append(name)
    return tuple(unique)


@lru_cache(maxsize=1)
def fuzzy_first_name_buckets() -> dict[int, tuple[str, ...]]:
    buckets: dict[int, list[str]] = {}
    for name in fuzzy_first_names():
        buckets.setdefault(len(name), []).append(name)
    return {length: tuple(names) for length, names in buckets.items()}


def _name_database_words(limit_first: int, limit_surname: int) -> tuple[str, ...]:
    first_names, surnames = _name_database()
    words = [name.title() for name in first_names[:limit_first]]
    words.extend(name.title() for name in surnames[:limit_surname])
    return tuple(dict.fromkeys(words))


def _surname_support_score(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    surnames = common_surnames()
    hits = sum(1 for token in tokens[1:] if token in surnames)
    distinctive_hits = sum(1 for token in tokens[1:] if len(token) >= 5 and token in surnames)
    return min(0.12, hits * 0.035 + distinctive_hits * 0.025)


OCR_NAME_TOKEN_CORRECTIONS = {
    "alues": "Alves",
    "dousa": "Souza",
    "feirra": "Ferreira",
    "ligor": "Higor",
    "nunen": "Nunes",
    "nune": "Nunes",
    "nuneit": "Nunes",
    "nuneiti": "Nunes",
    "nuner": "Nunes",
    "nunert": "Nunes",
    "nunes": "Nunes",
    "nuns": "Nunes",
    "peates": "Prates",
    "pentes": "Prates",
    "pratt": "Prates",
    "pratts": "Prates",
    "pratcs": "Prates",
    "seus": "Jesus",
    "une": "Nunes",
}

VIDEO_NON_NAME_STARTS = {
    "uma", "um", "umas", "uns", "das", "dos", "pessoa", "pessoas", "parte",
    "essencia", "equipe", "profissional", "morador", "moradora",
    "todo", "toda", "todos", "todas", "termo", "uso", "imagem", "direitos",
    "cedente", "cessionario", "cessionário", "clausula", "cláusula",
    "pelo", "polo", "pojo", "palo", "presente", "presento", "prosente",
    "instrumento", "instumento",
}


def _raw_starts_like_form_intro(text: str) -> bool:
    tokens = normalize(text).split()
    return bool(tokens and tokens[0] in FORM_INTRO_NOISE_WORDS)


def _has_person_name_shape(tokens: list[str], *, allow_single: bool = False) -> bool:
    if not tokens:
        return False
    first = tokens[0]
    if first not in accepted_first_names():
        return False
    if len(tokens) == 1:
        if not allow_single or len(first) < 4:
            return False
        return first not in common_surnames() or first in curated_first_names()
    surnames = common_surnames()
    corrected = {normalize(value) for value in OCR_NAME_TOKEN_CORRECTIONS.values()}
    support_tokens = [
        token for token in tokens[1:]
        if token in surnames or token in known_first_names() or token in corrected
    ]
    connectors = {"de", "da", "do", "dos", "das"}
    meaningful_tail = [token for token in tokens[1:] if token not in connectors]
    if not meaningful_tail:
        return False
    if support_tokens:
        return True
    # The 49k-name base is large enough that a completely unsupported tail is
    # more often a transcript/contract fragment than a real full name.
    return False


def _clean_video_name(raw: str) -> str:
    text = re.sub(r"[_|]+", " ", raw)
    stop_pattern = r"\b(?:" + "|".join(re.escape(word) for word in VIDEO_NAME_STOPS) + r")\b"
    text = re.split(stop_pattern, text, maxsplit=1, flags=re.I)[0]
    text = re.split(r"[\n\r.,;:!?]", text, maxsplit=1)[0]
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-Za-zÀ-ÿ' ]+", " ", text)
    words: list[str] = []
    for word in re.sub(r"\s+", " ", text).strip().split():
        cleaned = normalize(word)
        if cleaned in VIDEO_FILLER_WORDS:
            continue
        if len(cleaned) >= 2 or cleaned in {"de", "da", "do"}:
            words.append(word)
    return re.sub(r"\s+", " ", " ".join(words)).strip(" .,-:")


def _valid_video_name(candidate: str, *, allow_single: bool = True) -> bool:
    raw_tokens = normalize(candidate).split()
    if any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in raw_tokens):
        return False
    tokens = _name_tokens(candidate)
    if not (1 <= len(tokens) <= 7):
        return False
    if not allow_single and len(tokens) < 2:
        return False
    if len(" ".join(tokens)) < 3:
        return False
    first = tokens[0] if tokens else ""
    if first in VIDEO_NON_NAME_STARTS:
        return False
    if first not in accepted_first_names():
        return False
    if all(token in VIDEO_FILLER_WORDS or token in NAME_BLOCKLIST for token in tokens):
        return False
    if len(tokens) == 1 and (len(tokens[0]) < 4 or tokens[0] not in accepted_first_names()):
        return False
    if not _has_person_name_shape(tokens, allow_single=allow_single):
        return False
    return True


def _valid_spoken_video_name(candidate: str, *, allow_single: bool = True) -> bool:
    if _raw_starts_like_form_intro(candidate):
        return False
    raw_tokens = normalize(candidate).split()
    if any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in raw_tokens):
        return False
    tokens = _name_tokens(candidate)
    if not (1 <= len(tokens) <= 5):
        return False
    if not allow_single and len(tokens) < 2:
        return False
    first = tokens[0] if tokens else ""
    if first not in accepted_first_names():
        return False
    if not _has_person_name_shape(tokens, allow_single=allow_single):
        return False
    return _valid_video_name(candidate, allow_single=allow_single)


def _valid_explicit_spoken_name(candidate: str) -> bool:
    """Validate a name captured immediately after an explicit self-introduction."""
    raw_tokens = normalize(candidate).split()
    if any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in raw_tokens):
        return False
    tokens = _name_tokens(candidate)
    if not (1 <= len(tokens) <= 5) or tokens[0] not in accepted_first_names():
        return False
    if len(tokens) == 1:
        return len(tokens[0]) >= 4
    # Explicit speech may contain a rare Brazilian surname absent from the
    # database. Requiring alphabetic tail tokens still rejects sentence scraps.
    return all(len(token) >= 3 for token in tokens[1:])


def _valid_file_person_name(candidate: str, *, allow_single: bool = False) -> bool:
    if _raw_starts_like_form_intro(candidate):
        return False
    raw_tokens = normalize(candidate).split()
    if any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in raw_tokens):
        return False
    tokens = _name_tokens(candidate)
    if not tokens:
        return False
    if len(tokens) == 1 and (not allow_single or len(tokens[0]) < 4):
        return False
    if len(tokens) > 7:
        return False
    if len(" ".join(tokens)) < 3:
        return False
    if not _has_person_name_shape(tokens, allow_single=allow_single):
        return False
    return True


def _split_compound_spoken_surnames(tokens: list[str]) -> list[str]:
    """Separate surnames that Whisper occasionally joins into one word."""
    surnames = common_surnames()
    expanded: list[str] = []
    for token in tokens:
        if token in surnames or len(token) < 8:
            expanded.append(token)
            continue
        split_parts: tuple[str, str] | None = None
        for index in range(3, len(token) - 2):
            left, right = token[:index], token[index:]
            if left in surnames and right in surnames:
                split_parts = (left, right)
                break
        if split_parts:
            expanded.extend(split_parts)
        else:
            expanded.append(token)
    return expanded


def _repair_explicit_spoken_name(candidate: str, introductory_firsts: Iterable[str]) -> str:
    """Reconcile a short introduction with the later explicit full-name answer."""
    tokens = normalize(candidate).split()
    while tokens and tokens[0] in {"por", "favor"}:
        tokens.pop(0)
    if not tokens:
        return ""
    tokens = _split_compound_spoken_surnames(tokens)
    if tokens[0] not in accepted_first_names():
        for first in introductory_firsts:
            first = normalize(first)
            if (
                first in accepted_first_names()
                and len(first) >= 4
                and tokens[0][:3] == first[:3]
                and SequenceMatcher(None, tokens[0], first).ratio() >= 0.56
            ):
                tokens[0] = first
                break
    repaired = " ".join(tokens).title()
    return repaired if _valid_explicit_spoken_name(repaired) else ""


def name_from_video_transcript(transcript: str) -> str:
    if not transcript.strip():
        return ""

    introductory_candidates: list[str] = []
    prefix_pattern = r"(?:^|[\n\r.,;:!?]\s*)(?:eu\s+sou|sou|eu\s*,?)\s+(.{3,70})"
    for match in re.finditer(prefix_pattern, transcript, flags=re.I):
        introductory_candidates.append(_clean_video_name(match.group(1)))
    before_consent = re.match(
        r"^\s*(?:e\s+o\s+|e\s+a\s+|o\s+|a\s+)?(.{3,90}?)\s+\b(?:autoriza|autorizo|concordo|aceito|permito|declaro)\b",
        transcript,
        flags=re.I,
    )
    if before_consent:
        introductory_candidates.append(_clean_video_name(before_consent.group(1)))
    introductory_firsts = [
        _name_tokens(candidate)[0]
        for candidate in introductory_candidates
        if _valid_explicit_spoken_name(candidate) and _name_tokens(candidate)
    ]

    explicit_patterns = [
        r"\b(?:fale|diga|fala|faz|diz)?\s*(?:o\s+)?(?:seu\s+)?nome\s+(?:todo|completo)(?:\s*[,;:-]?\s*por\s+favor)?\s*(?:\?|:|-|,|\.)*\s+(.{3,90})",
        r"\b(?:qual\s+(?:e|eh|é)\s+)?(?:o\s+)?(?:seu\s+)?nome(?:\s+completo)?\s*(?:\?|:|-)?\s+(.{3,90})",
        r"\bmeu\s+nome(?:\s+completo)?\s*(?:e|eh|é|:|-)?\s+(.{3,90})",
        r"\b(?:eu\s+)?me\s+chamo\s+(.{3,90})",
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, transcript, flags=re.I):
            candidate = _clean_video_name(match.group(1))
            if _valid_explicit_spoken_name(candidate):
                return candidate.title()
            repaired = _repair_explicit_spoken_name(candidate, introductory_firsts)
            if repaired:
                return repaired

    for candidate in introductory_candidates:
        if _valid_explicit_spoken_name(candidate):
            return candidate.title()
    return ""


def rough_video_name_candidates(transcript: str) -> list[str]:
    if not transcript.strip():
        return []
    fragments: list[str] = []
    patterns = [
        r"\b(?:fale|diga|fala|faz|diz)?\s*(?:o\s+)?(?:seu\s+)?nome\s+(?:todo|completo)(?:\s*[,;:-]?\s*por\s+favor)?\s*(?:\?|:|-|,|\.)*\s+(.{3,90})",
        r"\b(?:qual\s+(?:e|eh|é)\s+)?(?:o\s+)?(?:seu\s+)?nome(?:\s+completo)?\s*(?:\?|:|-)?\s+(.{3,90})",
        r"\bmeu\s+nome(?:\s+completo)?\s*(?:e|eh|é|:|-)?\s+(.{3,90})",
        r"\b(?:eu\s+)?me\s+chamo\s+(.{3,90})",
        r"(?:^|[\n\r.,;:!?]\s*)(?:eu\s+sou|sou)\s+(.{3,70})",
        r"(?:^|[\n\r.,;:!?]\s*)eu\s+(.{3,70})",
        r"^\s*(?:e\s+o\s+|e\s+a\s+|o\s+|a\s+)?(.{3,90}?)\s+\b(?:autoriza|autorizo|concordo|aceito|permito|declaro)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, transcript, flags=re.I):
            raw = re.split(r"\b(?:autoriza|autorizo|concordo|aceito|permito|declaro|imagem|voz|audio|áudio)\b", match.group(1), maxsplit=1, flags=re.I)[0]
            raw = re.sub(r"\d+", " ", raw)
            raw = re.sub(r"[^A-Za-zÀ-ÿ' ]+", " ", raw)
            words: list[str] = []
            for word in re.sub(r"\s+", " ", raw).strip().split():
                token = normalize(word)
                if not token or token in VIDEO_FILLER_WORDS or token in FORM_INTRO_NOISE_WORDS:
                    continue
                if len(token) >= 3 or token in {"de", "da", "do", "dos", "das"}:
                    words.append(word)
            cleaned = re.sub(r"\s+", " ", " ".join(words)).strip(" .,-:")
            tokens = normalize(cleaned).split()
            if 1 <= len(tokens) <= 6 and len(" ".join(tokens)) >= 3:
                fragments.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        key = normalize(fragment)
        if key and key not in seen:
            seen.add(key)
            unique.append(fragment)
    return unique[:6]


def _roster_first_names(roster_names: Iterable[str]) -> list[str]:
    return [
        first for first in (_first_name_token(name) for name in roster_names)
        if first
    ]


def _spoken_fragment_roster_score(
    roster_name: str,
    fragment: str,
    roster_firsts: list[str],
) -> float:
    roster_tokens = _name_tokens(_canonical_person_name(roster_name))
    fragment_tokens = _name_tokens(_canonical_person_name(fragment))
    if not roster_tokens or not fragment_tokens:
        return 0.0

    first_score = _spoken_first_name_similarity(roster_tokens[0], fragment_tokens[0])
    if first_score < 0.78:
        return 0.0
    if not _fuzzy_unique_in_pool(roster_tokens[0], roster_firsts, threshold=0.82):
        return 0.0

    remaining_scores = [
        max(
            (_token_similarity(roster_token, fragment_token) for fragment_token in fragment_tokens[1:]),
            default=0.0,
        )
        for roster_token in roster_tokens[1:]
    ]
    strong_tail = sum(score >= 0.82 for score in remaining_scores)
    soft_tail = sum(score >= 0.68 for score in remaining_scores)
    exact_tail = len(set(roster_tokens[1:]) & set(fragment_tokens[1:]))

    if len(roster_tokens) == 1:
        return 0.84 if first_score >= 0.92 else 0.76
    if strong_tail >= 2 or exact_tail >= 2:
        return min(0.98, 0.86 + 0.04 * min(3, strong_tail) + 0.04 * first_score)
    if strong_tail >= 1 or exact_tail >= 1:
        return min(0.93, 0.78 + 0.09 * first_score + 0.04 * min(2, soft_tail))
    if first_score >= 0.86:
        # Em um lote pequeno, um primeiro nome único é evidência útil. Ele não
        # é suficiente quando há duas pessoas com o mesmo primeiro nome.
        return 0.76
    if first_score >= 0.84 and soft_tail >= 1:
        return 0.72
    return 0.0


def _spoken_first_name_similarity(expected: str, heard: str) -> float:
    score = _token_similarity(expected, heard)
    first = normalize(expected).replace("y", "i")
    second = normalize(heard).replace("y", "i")
    first = re.sub(r"c(?=[ei])", "s", first)
    second = re.sub(r"c(?=[ei])", "s", second)
    score = max(score, SequenceMatcher(None, first, second).ratio())
    if len(first) >= 5 and len(second) >= 5 and first[-4:] == second[-4:]:
        score = max(score, 0.86)
    return score


def match_transcript_to_roster(
    transcript: str,
    roster_names: Iterable[str],
) -> tuple[str, float, float, str]:
    roster = [name for name in roster_names if _valid_name_candidate(name)]
    fragments = rough_video_name_candidates(transcript)
    direct = name_from_video_transcript(transcript)
    if direct:
        fragments.insert(0, direct)
    if not roster or not fragments:
        return "", 0.0, 0.0, direct

    roster_firsts = _roster_first_names(roster)
    ranked: list[tuple[float, str, str]] = []
    for roster_name in roster:
        best_score = 0.0
        best_fragment = ""
        for fragment in fragments:
            score = _spoken_fragment_roster_score(roster_name, fragment, roster_firsts)
            if score > best_score:
                best_score = score
                best_fragment = fragment
        ranked.append((best_score, roster_name, best_fragment))
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name, best_fragment = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < 0.70 or (best_score < 0.82 and best_score - second_score < 0.10):
        return "", best_score, second_score, best_fragment or direct
    return best_name, best_score, second_score, best_fragment or direct


def generic_video_file_label(label: str) -> bool:
    norm = normalize(label)
    if not norm:
        return True
    if re.match(r"^(img|vid|mov|video|wa|pxl|dsc|scan|digitalizacao|digitalizacao em)\s*\d+", norm):
        return True
    return bool(re.fullmatch(r"\d{6,}(?:\s+\d+)?", norm))


def unidentified_person_label(label: str) -> bool:
    norm = normalize(label)
    if not norm:
        return True
    if norm.startswith("nome nao lido"):
        return True
    if generic_video_file_label(label):
        return True
    return False


def transcribe_video_label(path: Path, roster_names: Iterable[str] | None = None) -> tuple[str, str, str]:
    transcript = ""
    warnings: list[str] = []
    file_label = label_from_photo_path(path)
    cleaned_file_label = _canonical_person_name(file_label)
    file_name_is_person = (not generic_video_file_label(file_label)) and _valid_file_person_name(cleaned_file_label)
    if file_name_is_person and not VERIFY_NAMED_VIDEO_FILES:
        return cleaned_file_label or file_label, "", "Nome usado do arquivo; transcrição pulada para agilizar"
    roster = list(roster_names or [])
    try:
        transcript = _faster_whisper_transcribe_video(
            path,
            max_seconds=VIDEO_NAME_TRANSCRIPTION_SECONDS,
            careful=bool(roster),
            roster_names=roster,
        )
    except ImportError:
        pass
    except Exception as exc:
        warnings.append(f"Whisper local falhou: {exc}")

    if not transcript and speech_engine_name():
        try:
            transcript = _speech_transcribe_video(path)
        except Exception as exc:
            warnings.append(f"Transcrição local do macOS falhou: {exc}")
    if not transcript and not warnings:
        warnings.append("Transcrição local indisponível")

    spoken_name = name_from_video_transcript(transcript) if transcript else ""
    roster_name = ""
    roster_score = 0.0
    if transcript and roster:
        roster_name, roster_score, _second_score, roster_fragment = match_transcript_to_roster(transcript, roster)
        if roster_name:
            spoken_name = roster_name
            warnings.append(
                f"Nome falado associado pela lista do lote ({roster_score:.0%})"
                + (f": {roster_fragment}" if roster_fragment else "")
            )
        elif generic_video_file_label(file_label):
            # Uma segunda passada curta recupera nomes que perderam uma sílaba
            # na primeira leitura, sem encarecer os vídeos já identificados.
            try:
                retry = _faster_whisper_transcribe_video(
                    path,
                    max_seconds=VIDEO_CAREFUL_RETRY_SECONDS,
                    careful=True,
                    roster_names=roster,
                )
                retry_name, retry_score, _retry_second, retry_fragment = match_transcript_to_roster(retry, roster)
                if retry_name:
                    transcript = retry
                    spoken_name = retry_name
                    warnings.append(
                        f"Nome falado recuperado na confirmação ({retry_score:.0%})"
                        + (f": {retry_fragment}" if retry_fragment else "")
                    )
            except Exception:
                pass
        if not spoken_name and generic_video_file_label(file_label):
            # A lista pode ajudar ou enviesar o Whisper. Quando ela não produz
            # um nome, uma última escuta curta e neutra decide o caso.
            try:
                neutral = _faster_whisper_transcribe_video(
                    path,
                    max_seconds=max(VIDEO_CAREFUL_RETRY_SECONDS, 24.0),
                    careful=True,
                    roster_names=None,
                )
                neutral_name = name_from_video_transcript(neutral)
                neutral_roster, neutral_score, _neutral_second, neutral_fragment = match_transcript_to_roster(neutral, roster)
                if neutral_roster:
                    spoken_name = neutral_roster
                    transcript = neutral
                    warnings.append(
                        f"Nome falado recuperado sem viés ({neutral_score:.0%})"
                        + (f": {neutral_fragment}" if neutral_fragment else "")
                    )
                elif neutral_name:
                    spoken_name = neutral_name
                    transcript = neutral
                    warnings.append("Nome falado recuperado na escuta neutra")
            except Exception:
                pass
    if transcript and not spoken_name and not roster:
        try:
            careful_transcript = _faster_whisper_transcribe_video(path, max_seconds=VIDEO_CAREFUL_RETRY_SECONDS, careful=True)
            careful_name = name_from_video_transcript(careful_transcript)
            if careful_name:
                spoken_name = careful_name
                transcript = f"{transcript} {careful_transcript}".strip()
        except Exception:
            pass
    if file_name_is_person:
        if (
            spoken_name
            and len(_name_tokens(spoken_name)) > len(_name_tokens(cleaned_file_label))
            and _name_extends_or_matches(cleaned_file_label, spoken_name)
        ):
            name = spoken_name
            warnings.append("Nome falado confirmou e completou o nome do arquivo")
        else:
            name = cleaned_file_label or file_label
            if spoken_name and not _name_extends_or_matches(spoken_name, cleaned_file_label) and not _name_extends_or_matches(cleaned_file_label, spoken_name):
                warnings.append(f"Nome falado divergente ignorado: {spoken_name}")
            elif transcript and not spoken_name:
                warnings.append("Nome falado não identificado; usei o nome do arquivo")
    elif spoken_name:
        name = spoken_name
    else:
        name = f"Nome não lido ({file_label})"
        if transcript:
            warnings.append("Nome falado não identificado; deixei fora do casamento automático")
    return name, transcript, " · ".join(warnings)


def extract_video_thumbnail(video_path: Path, destination: Path, used_paths: set[Path]) -> Path:
    qlmanage = shutil.which("qlmanage")
    if not qlmanage:
        raise RuntimeError("Não encontrei o gerador de miniaturas do macOS.")

    scratch = destination / "_miniaturas"
    scratch.mkdir(parents=True, exist_ok=True)
    expected = scratch / f"{video_path.name}.png"
    expected.unlink(missing_ok=True)
    before = {path.resolve() for path in scratch.glob("*") if path.is_file()}
    completed = subprocess.run(
        [qlmanage, "-t", "-s", "1200", "-o", str(scratch), str(video_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    after = [path for path in scratch.glob("*") if path.is_file() and path.resolve() not in before]
    if expected.exists():
        after.insert(0, expected)
    if completed.returncode != 0 and not after:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or "Não consegui gerar a miniatura do vídeo.")

    thumbnail = after[0] if after else expected
    image = ImageOps.exif_transpose(Image.open(thumbnail)).convert("RGB")
    output = unique_path(destination / f"{safe_filename(video_path.stem)}.jpg", used_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "JPEG", quality=92)
    return output


def people_from_video_folder(
    folder: Path,
    destination: Path,
    progress_callback=None,
    roster_names: Iterable[str] | None = None,
) -> tuple[list[PersonPhoto], list[str]]:
    videos = video_files_from_folder(folder)
    people: list[PersonPhoto] = []
    failures: list[str] = []
    used_paths: set[Path] = set()
    started = time.monotonic()
    for index, video in enumerate(videos, start=1):
        try:
            label, transcript, warning = transcribe_video_label(video, roster_names=roster_names)
            thumbnail = extract_video_thumbnail(video, destination, used_paths)
            caption_parts = [f"Vídeo: {video.name}"]
            if transcript:
                caption_parts.append(f"Transcrição: {transcript}")
            if warning:
                caption_parts.append(warning)
            people.append(PersonPhoto(label=label, caption=" · ".join(caption_parts), image_path=thumbnail))
        except Exception as exc:
            failures.append(f"{video.name}: {exc}")
        if progress_callback:
            elapsed = time.monotonic() - started
            eta = (elapsed / index) * (len(videos) - index) if index else None
            progress_callback(index, len(videos), video.name, eta)
    return people, failures


def _image_nsdata(image: Image.Image):
    import Foundation

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    payload = buffer.getvalue()
    return Foundation.NSData.dataWithBytes_length_(payload, len(payload))


def _vision_face_observations(image: Image.Image) -> list[tuple[float, float, float, float]]:
    try:
        import Vision
        import objc

        with objc.autorelease_pool():
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(_image_nsdata(image), {})
            request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
            success, _error = handler.performRequests_error_([request], None)
            if not success:
                return []
            observations: list[tuple[float, float, float, float]] = []
            for observation in request.results() or []:
                box = observation.boundingBox()
                observations.append((
                    float(box.origin.x),
                    float(box.origin.y),
                    float(box.size.width),
                    float(box.size.height),
                ))
            return observations
    except Exception:
        return []


def _yunet_face_observations(image: Image.Image) -> list[tuple[float, float, float, float]]:
    detector, _recognizer = _sface_models()
    if detector is None or cv2 is None:
        return []
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    try:
        detector.setInputSize((width, height))
        _status, faces = detector.detect(bgr)
    except Exception:
        return []
    if faces is None:
        return []
    observations: list[tuple[float, float, float, float]] = []
    for face in faces:
        x, y, w, h = (float(value) for value in face[:4])
        if w <= 0 or h <= 0:
            continue
        observations.append((x / width, 1.0 - (y + h) / height, w / width, h / height))
    return observations


def _face_observation_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_left, first_bottom, first_width, first_height = first
    second_left, second_bottom, second_width, second_height = second
    left = max(first_left, second_left)
    right = min(first_left + first_width, second_left + second_width)
    bottom = max(first_bottom, second_bottom)
    top = min(first_bottom + first_height, second_bottom + second_height)
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    if intersection <= 0:
        return 0.0
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


@lru_cache(maxsize=1)
def _opencv_face_cascade():
    if cv2 is None:
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


@lru_cache(maxsize=1)
def _opencv_eye_cascade():
    if cv2 is None:
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


def _opencv_face_observations(image: Image.Image) -> list[tuple[float, float, float, float]]:
    cascade = _opencv_face_cascade()
    eye_cascade = _opencv_eye_cascade()
    if cascade is None or eye_cascade is None:
        return []
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    minimum = max(28, round(min(image.size) * 0.055))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.06,
        minNeighbors=4,
        minSize=(minimum, minimum),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    width, height = image.size
    verified: list[tuple[float, float, float, float]] = []
    for x, y, w, h in faces:
        upper_face = gray[y:y + max(1, round(h * 0.68)), x:x + w]
        eyes = eye_cascade.detectMultiScale(
            upper_face,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(max(8, round(w * 0.12)), max(8, round(h * 0.10))),
        )
        if len(eyes) < 1:
            continue
        verified.append((
            float(x) / width,
            1.0 - float(y + h) / height,
            float(w) / width,
            float(h) / height,
        ))
    return verified


def _face_observations(image: Image.Image) -> list[tuple[float, float, float, float]]:
    observations: list[tuple[float, float, float, float]] = []
    for observation in _vision_face_observations(image) + _yunet_face_observations(image):
        if not any(_face_observation_iou(observation, existing) >= 0.55 for existing in observations):
            observations.append(observation)
    return observations or _opencv_face_observations(image)


def _crop_face_observation(
    image: Image.Image,
    observation: tuple[float, float, float, float],
) -> Image.Image | None:
    x, y, w, h = observation
    width, height = image.size
    # Mostra cabeça, ombros e parte do contexto para a confirmação visual.
    left = int((x - w * 0.58) * width)
    right = int((x + w * 1.58) * width)
    top = int((1 - y - h * 1.50) * height)
    bottom = int((1 - y + h * 1.20) * height)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right - left < 45 or bottom - top < 45:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def _face_observation_area(observation: tuple[float, float, float, float]) -> float:
    return observation[2] * observation[3]


def _is_foreground_face_observation(observation: tuple[float, float, float, float]) -> bool:
    _x, _y, width, height = observation
    area = width * height
    return height >= VIDEO_FOREGROUND_MIN_FACE_HEIGHT and area >= VIDEO_FOREGROUND_MIN_FACE_AREA


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, second = divmod(total, 60)
    return f"{minutes:02d}:{second:02d}"


def _append_video_face_previews(
    image: Image.Image,
    timestamp: float,
    destination: Path,
    previews: list[VideoFacePreview],
    max_faces: int,
) -> bool:
    found = False
    detected = sorted(_face_observations(image), key=_face_observation_area, reverse=True)
    largest_area = _face_observation_area(detected[0]) if detected else 0.0
    observations = [
        observation for observation in detected
        if _is_foreground_face_observation(observation)
        and _face_observation_area(observation) >= largest_area * 0.28
    ]
    for face_index, observation in enumerate(observations, start=1):
        crop = _crop_face_observation(image, observation)
        if crop is None:
            continue
        crop = ImageOps.fit(crop, (220, 220), Image.Resampling.LANCZOS)
        output = destination / f"rosto_{len(previews) + 1:03d}_{format_timestamp(timestamp).replace(':', '-')}_f{face_index}.jpg"
        crop.save(output, "JPEG", quality=92)
        previews.append(VideoFacePreview(timestamp=timestamp, index=face_index, image_path=output, kind="rosto"))
        found = True
        if len(previews) >= max_faces:
            break
    return found


def _video_sample_times(duration: float, interval: float) -> list[float]:
    interval = max(interval, 0.25)
    if duration > 0:
        regular = [index * interval for index in range(int(duration / interval) + 1)]
        # Nomes falados e cartelas costumam aparecer nos primeiros segundos.
        opening = [0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        times = sorted({min(duration, value) for value in regular + opening if value <= duration})
        if len(times) <= VIDEO_AUDIT_MAX_FRAME_SAMPLES:
            return times
        opening_times = [value for value in times if value <= min(duration, 16.0)]
        remaining = [value for value in times if value > 16.0]
        slots = max(0, VIDEO_AUDIT_MAX_FRAME_SAMPLES - len(opening_times))
        if slots and remaining:
            step = max(1.0, len(remaining) / slots)
            sampled = [remaining[min(len(remaining) - 1, int(index * step))] for index in range(slots)]
        else:
            sampled = []
        return sorted(set((opening_times + sampled)[:VIDEO_AUDIT_MAX_FRAME_SAMPLES]))
    return [index * interval for index in range(VIDEO_AUDIT_MAX_FRAME_SAMPLES)]


def extract_video_face_previews(
    video_path: Path,
    destination: Path,
    interval: float = VIDEO_AUDIT_FRAME_INTERVAL,
    max_faces: int = VIDEO_AUDIT_MAX_FACES,
    progress_callback=None,
    visual_text_callback=None,
    visual_name_hints: Iterable[str] | None = None,
) -> list[VideoFacePreview]:
    destination.mkdir(parents=True, exist_ok=True)
    previews: list[VideoFacePreview] = []
    stable_visual_name_hints = tuple(visual_name_hints or ())
    last_visual_ocr_at = -VIDEO_AUDIT_VISUAL_OCR_INTERVAL

    def inspect_frame(image: Image.Image, timestamp: float) -> bool:
        nonlocal last_visual_ocr_at
        found = _append_video_face_previews(image, timestamp, destination, previews, max_faces)
        if (
            found
            and visual_text_callback
            and timestamp - last_visual_ocr_at >= VIDEO_AUDIT_VISUAL_OCR_INTERVAL
        ):
            last_visual_ocr_at = timestamp
            try:
                text = _vision_region_text(
                    image,
                    stable_visual_name_hints,
                    fast=True,
                    alternatives=2,
                )
            except Exception:
                text = ""
            if text.strip():
                visual_text_callback(timestamp, text)
        return found

    if cv2 is not None:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if capture.isOpened():
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
                if not duration:
                    duration = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                for timestamp in _video_sample_times(duration, interval):
                    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ok, frame_bgr = capture.read()
                    if not ok or frame_bgr is None:
                        continue
                    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
                    image.thumbnail((960, 960), Image.Resampling.LANCZOS)
                    inspect_frame(image, timestamp)
                    if len(previews) >= max_faces:
                        return previews
                    if progress_callback:
                        progress_callback(min(timestamp, duration) if duration else timestamp, duration, len(previews))
                return previews
        finally:
            capture.release()

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("O componente de leitura de vídeo não está instalado. Abra o app pelo .command para atualizar.") from exc

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        misses = 0
        for requested_time in _video_sample_times(duration, interval):
            try:
                if stream.time_base:
                    offset = int(requested_time / float(stream.time_base))
                    container.seek(offset, any_frame=False, backward=True, stream=stream)
                else:
                    container.seek(int(requested_time * av.time_base), any_frame=False, backward=True)
            except Exception:
                container.seek(int(requested_time * av.time_base), any_frame=True, backward=True)
            image = None
            timestamp = requested_time
            for frame in container.decode(stream):
                timestamp = float(frame.time or requested_time)
                if timestamp + 0.05 < requested_time:
                    continue
                scale = min(1.0, 960.0 / max(frame.width, frame.height))
                if scale < 1.0:
                    frame = frame.reformat(width=max(1, int(frame.width * scale)), height=max(1, int(frame.height * scale)))
                image = frame.to_image().convert("RGB")
                break
            if image is None:
                misses += 1
                if misses >= 5 and not duration:
                    break
                continue
            misses = 0
            image.thumbnail((960, 960), Image.Resampling.LANCZOS)
            inspect_frame(image, timestamp)
            if len(previews) >= max_faces:
                return previews
            if progress_callback:
                progress_callback(min(timestamp, duration), duration, len(previews))
            if duration and timestamp >= duration:
                break
    finally:
        container.close()
    return previews


def video_face_contact_sheet(previews: list[VideoFacePreview], destination: Path, columns: int = 5) -> Path | None:
    if not previews:
        return None
    tile_w, tile_h = 220, 246
    rows = (len(previews) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    for idx, preview in enumerate(previews):
        image = Image.open(preview.image_path).convert("RGB")
        tile = Image.new("RGB", (tile_w, tile_h), "#111111")
        tile.paste(image, (0, 0))
        # Use PIL's default font to avoid external font dependencies.
        from PIL import ImageDraw

        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 220, tile_w, tile_h), fill="#111111")
        face_label = preview.label or "Não identificado"
        draw.text(
            (8, 225),
            f"{format_timestamp(preview.timestamp)} · {safe_filename(face_label)[:27]}",
            fill="#F7F3E8",
        )
        sheet.paste(tile, ((idx % columns) * tile_w, (idx // columns) * tile_h))
    output = destination / "rostos_detectados.jpg"
    sheet.save(output, "JPEG", quality=92)
    return output


@lru_cache(maxsize=1)
def _sface_models():
    if cv2 is None or not FACE_DETECTION_MODEL.exists() or not FACE_RECOGNITION_MODEL.exists():
        return None, None
    try:
        detector = cv2.FaceDetectorYN_create(
            str(FACE_DETECTION_MODEL), "", (320, 320), 0.75, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF_create(str(FACE_RECOGNITION_MODEL), "")
        return detector, recognizer
    except Exception:
        return None, None


def _sface_feature(image: Image.Image) -> np.ndarray | None:
    detector, recognizer = _sface_models()
    if detector is None or recognizer is None:
        return None
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    if min(width, height) < 48:
        return None
    try:
        detector.setInputSize((width, height))
        _result, faces = detector.detect(bgr)
        if faces is None or not len(faces):
            return None
        face = max(faces, key=lambda row: float(row[2] * row[3]))
        aligned = recognizer.alignCrop(bgr, face)
        return recognizer.feature(aligned)
    except Exception:
        return None


def _sface_similarity(first: np.ndarray, second: np.ndarray) -> float:
    _detector, recognizer = _sface_models()
    if recognizer is None:
        return 0.0
    try:
        return float(recognizer.match(first, second, cv2.FaceRecognizerSF_FR_COSINE))
    except Exception:
        return 0.0


def _sface_feature_from_path(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as source:
            return _sface_feature(source.convert("RGB"))
    except Exception:
        return None


def pdf_name_from_path(path: Path) -> str:
    name = path.stem
    name = re.sub(
        r"\s*[-_]\s*Termo\s+de\s+Autoriza[cç][aã]o(?:[ _-]*SEMFOTO)?$",
        "",
        name,
        flags=re.I,
    )
    name = re.sub(r"[ _-]+SEMFOTO$", "", name, flags=re.I)
    cleaned = _canonical_person_name(name)
    if _valid_file_person_name(cleaned, allow_single=True):
        return cleaned
    return re.sub(r"\s+", " ", name).strip()


def _name_after_authorization_anchor(text: str) -> str:
    if not text.strip():
        return ""
    flattened = re.sub(r"\s+", " ", text)
    anchors = [
        r"Pelo\s+presente\s+instrumento\s*,?\s*eu\s*[,;:]?\s*",
        r"Pelo\s+presente\s+termo\s+particular\s+de\s+responsabilidade\s*[:;,]?\s*",
    ]
    stops = (
        r"(?=[\s,;:-]+(?:portador\s*\(?a\)?|inscrit[oa]|CPF(?:/MF)?|RG\b|residente\b|"
        r"estado\s+civil|profiss[aã]o|n[ºo°]\b))"
    )
    for anchor in anchors:
        match = re.search(anchor + r"(.{2,120}?)" + stops, flattened, flags=re.I)
        if not match:
            continue
        candidate = _canonical_person_name(match.group(1))
        tokens = _name_tokens(candidate)
        rare_but_structured = (
            2 <= len(tokens) <= 7
            and all(len(token) >= 3 for token in tokens)
            and not any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in normalize(candidate).split())
        )
        if _valid_file_person_name(candidate, allow_single=True) or rare_but_structured:
            return candidate
    return ""


def authorization_name_from_pdf(path: Path) -> tuple[str, str]:
    """Return a reliable name and its source without reading the whole PDF visually."""
    try:
        document = fitz.open(path)
        try:
            text = "\n".join(document[index].get_text("text") for index in range(min(2, document.page_count)))
        finally:
            document.close()
    except Exception:
        return "", ""
    name = _name_after_authorization_anchor(text)
    return (name, "texto do PDF") if name else ("", "")


def authorization_name_from_path(path: Path) -> tuple[str, str]:
    file_name = pdf_name_from_path(path)
    file_valid = _valid_file_person_name(file_name, allow_single=True)
    suffix = path.suffix.lower()
    if not suffix:
        suffix = downloaded_file_suffix(path, "", AUTHORIZATION_SUPPORTED)
    if suffix == ".pdf":
        embedded_name, embedded_source = authorization_name_from_pdf(path)
        if embedded_name:
            if not file_valid or generic_video_file_label(file_name):
                return embedded_name, embedded_source
            if _name_extends_or_matches(embedded_name, file_name) or _name_extends_or_matches(file_name, embedded_name):
                longer = embedded_name if len(_name_tokens(embedded_name)) > len(_name_tokens(file_name)) else file_name
                return longer, f"nome do arquivo + {embedded_source}"
        if file_valid:
            return file_name, "nome do arquivo"
        return "", ""
    if file_valid:
        return file_name, "nome do arquivo"
    try:
        spoken_name, _transcript, _warning = transcribe_video_label(path, roster_names=None)
    except Exception:
        spoken_name = ""
    if _valid_video_name(spoken_name, allow_single=True):
        return spoken_name, "nome falado na autorização"
    return "", ""


def authorization_files_from_folder(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in AUTHORIZATION_SUPPORTED
            or (not path.suffix and bool(downloaded_file_suffix(path, "", AUTHORIZATION_SUPPORTED)))
        )
        and "autorizacao de imagem" not in {normalize(part) for part in path.parts}
    )


def authorization_people_from_folder(folder: Path, progress_callback=None) -> tuple[list[PersonPhoto], list[str]]:
    paths = authorization_files_from_folder(folder)
    people: list[PersonPhoto] = []
    failures: list[str] = []
    seen: set[tuple[str, Path]] = set()
    for index, path in enumerate(paths, start=1):
        name, source = authorization_name_from_path(path)
        if name:
            key = (normalize(name), path.resolve())
            if key not in seen:
                seen.add(key)
                people.append(PersonPhoto(label=name, caption=source, image_path=path))
        else:
            failures.append(f"{path.name}: nome não identificado")
        if progress_callback:
            progress_callback(index, len(paths), path.name)
    return people, failures


def _authorization_pdf_face_crop(path: Path) -> Image.Image | None:
    if "semfoto" in normalize(path.stem):
        return None
    try:
        image = load_source(path)
        width, height = image.size
        region = image.crop((round(width * 0.68), 0, width, round(height * 0.28)))
        # O retrato inserido no PDF costuma ter apenas 25–45 px de rosto. A
        # operação thumbnail nunca amplia e fazia todos esses retratos sumirem
        # da base facial; ampliamos a região antes da detecção.
        scale = min(4.0, max(1.0, 1800.0 / max(region.size)))
        if scale > 1.0:
            region = region.resize(
                (round(region.width * scale), round(region.height * scale)),
                Image.Resampling.LANCZOS,
            )
        detected = sorted(_face_observations(region), key=_face_observation_area, reverse=True)
        if not detected:
            return None
        crop = _crop_face_observation(region, detected[0])
        return ImageOps.fit(crop, (220, 220), Image.Resampling.LANCZOS) if crop is not None else None
    except Exception:
        return None


def build_authorization_face_references(
    people: list[PersonPhoto],
    destination: Path,
    progress_callback=None,
) -> list[AuthorizationFaceReference]:
    """Build local SFace references from PDF portraits and authorization videos."""
    destination.mkdir(parents=True, exist_ok=True)
    references: list[AuthorizationFaceReference] = []
    for index, person in enumerate(people, start=1):
        crops: list[tuple[Image.Image, str]] = []
        suffix = person.image_path.suffix.lower()
        if suffix == ".pdf":
            crop = _authorization_pdf_face_crop(person.image_path)
            if crop is not None:
                crops.append((crop, "retrato do PDF"))
        elif suffix in VIDEO_SUPPORTED:
            media_dir = destination / f"video_{index:03d}"
            try:
                previews = extract_video_face_previews(
                    person.image_path,
                    media_dir,
                    interval=1.0,
                    max_faces=2,
                )
            except Exception:
                previews = []
            for preview in previews:
                try:
                    with Image.open(preview.image_path) as source:
                        crops.append((source.convert("RGB"), "rosto da autorização em vídeo"))
                except Exception:
                    continue
        for crop, source in crops:
            feature = _sface_feature(crop)
            if feature is not None:
                references.append(AuthorizationFaceReference(person=person, feature=feature, source=source))
        if progress_callback:
            progress_callback(index, len(people), len(references))
    return references


def pdf_terms_from_folder(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def video_pdf_suggestions(video_text: str, pdf_paths: list[Path]) -> list[VideoAuditSuggestion]:
    if not video_text.strip() or not pdf_paths:
        return []
    pseudo_people = [
        PersonPhoto(label=pdf_name_from_path(path), caption="", image_path=path)
        for path in pdf_paths
    ]
    token_counts = _person_token_counts(pseudo_people)
    suggestions: list[VideoAuditSuggestion] = []
    normalized_text = normalize(video_text)
    for pdf_path, person in zip(pdf_paths, pseudo_people):
        name = person.label
        name_norm = normalize(name)
        score = free_text_match_score(name, video_text, token_counts)
        if name_norm and name_norm in normalized_text:
            score = max(score, 0.96)
        elif name_norm:
            tokens = [token for token in _name_tokens(name) if len(token) >= 4]
            exact_hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", normalized_text))
            if len(tokens) == 1 and exact_hits:
                score = max(score, 0.62)
            elif exact_hits >= 2:
                score = max(score, min(0.90, 0.58 + exact_hits * 0.11))
        if score >= 0.50:
            reason = "nome ouvido no vídeo" if score >= 0.70 else "nome parecido na transcrição"
            suggestions.append(VideoAuditSuggestion(pdf_path=pdf_path, confidence=score, reason=reason))
    return sorted(suggestions, key=lambda item: (item.confidence, alpha_key(pdf_name_from_path(item.pdf_path))), reverse=True)


def _strict_spoken_fragments(transcript: str) -> list[str]:
    def valid_explicit_evidence(candidate: str) -> bool:
        raw_tokens = normalize(candidate).split()
        if any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in raw_tokens):
            return False
        tokens = _name_tokens(candidate)
        if not (1 <= len(tokens) <= 5):
            return False
        if len(tokens) == 1:
            return tokens[0] in accepted_first_names() and len(tokens[0]) >= 4
        # A construção explícita "eu sou" permite nomes raros, como Kirk,
        # desde que não contenha palavras de frase ou do formulário.
        return all(len(token) >= 3 for token in tokens)

    fragments: list[str] = []
    for pattern in (
        r"\b(?:eu\s+sou|meu\s+nome\s+(?:é|e|eh)|eu\s+me\s+chamo|me\s+chamo)\s+(.{3,80})",
        r"\b(?:qual\s+(?:é|e|eh)\s+(?:o\s+)?seu\s+nome(?:\s+completo)?)[?:,;\s]+(.{3,80})",
    ):
        for match in re.finditer(pattern, transcript, flags=re.I):
            fragment = re.split(
                r"\b(?:e\s+estou|e\s+sou|sou|trabalho|atuo|autorizo|concordo|aceito|para|aqui|hoje)\b",
                match.group(1),
                maxsplit=1,
                flags=re.I,
            )[0]
            fragment = _canonical_person_name(fragment)
            if valid_explicit_evidence(fragment):
                fragments.append(fragment)
    unique: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        key = normalize(fragment)
        if key and key not in seen:
            seen.add(key)
            unique.append(fragment)
    return unique


def _visual_roster_score(
    name: str,
    text: str,
    roster_firsts: list[str],
    token_counts: dict[str, int] | None = None,
) -> float:
    name_tokens = _name_tokens(name)
    text_tokens = [
        token for token in normalize(text).split()
        if len(token) >= 3 and token not in HARD_NON_NAME_WORDS
    ]
    if not name_tokens or not text_tokens:
        return 0.0
    first = name_tokens[0]
    first_unique = _fuzzy_unique_in_pool(first, roster_firsts, threshold=0.82)
    best = 0.0
    for index, token in enumerate(text_tokens):
        first_score = _spoken_first_name_similarity(first, token)
        if first_score < 0.82:
            continue
        window = text_tokens[index:index + min(7, len(name_tokens) + 2)]
        tails = [
            max((_token_similarity(expected, actual) for actual in window[1:]), default=0.0)
            for expected in name_tokens[1:]
        ]
        strong_tail = sum(score >= 0.82 for score in tails)
        if len(name_tokens) == 1:
            score = 0.78 if first_score >= 0.92 else 0.70
        elif strong_tail >= 2:
            score = min(0.99, 0.91 + 0.02 * strong_tail)
        elif strong_tail == 1:
            score = min(0.94, 0.82 + 0.08 * first_score)
        else:
            # Em um lote pequeno, um primeiro nome exato e único é evidência
            # suficiente para sugerir a pessoa, mesmo quando a tarja omite ou
            # erra os sobrenomes. Nomes repetidos, como João, não usam este atalho.
            score = 0.80 if first_unique and first_score >= 0.92 else 0.0
        best = max(best, score)
    # Algumas tarjas perdem o primeiro nome, mas preservam um sobrenome raro.
    # Ele só decide quando aparece em uma única pessoa da base inteira.
    counts = token_counts or {}
    for token in name_tokens[1:]:
        if len(token) < 4 or counts.get(token, 0) != 1:
            continue
        if any(_token_similarity(token, actual) >= 0.93 for actual in text_tokens):
            best = max(best, 0.84)
    return best


def video_authorization_suggestions(
    transcript: str,
    visual_evidence: list[VideoVisualEvidence],
    authorization_people: list[PersonPhoto],
) -> list[VideoAuditSuggestion]:
    """Match only evidence extracted independently from the authorization roster."""
    if not authorization_people:
        return []
    roster = [person.label for person in authorization_people]
    roster_firsts = _roster_first_names(roster)
    token_counts = _person_token_counts(authorization_people)
    spoken_fragments = _strict_spoken_fragments(transcript)
    evidence: dict[Path, tuple[float, set[str]]] = {}

    def record(path: Path, score: float, reason: str) -> None:
        previous_score, reasons = evidence.get(path, (0.0, set()))
        if score >= 0.70:
            reasons.add(reason)
            evidence[path] = (max(previous_score, score), reasons)

    for fragment in spoken_fragments:
        ranked = sorted(
            (
                (_spoken_fragment_roster_score(person.label, fragment, roster_firsts), person)
                for person in authorization_people
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_person = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score >= 0.70 and (best_score >= 0.82 or best_score - second_score >= 0.10):
            record(best_person.image_path, best_score, f"apresentação falada: {fragment}")

    for item in visual_evidence:
        ranked = sorted(
            (
                (_visual_roster_score(person.label, item.text, roster_firsts, token_counts), person)
                for person in authorization_people
            ),
            key=lambda candidate: candidate[0],
            reverse=True,
        )
        best_score, best_person = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score >= 0.76 and (best_score >= 0.88 or best_score - second_score >= 0.10):
            record(best_person.image_path, best_score, f"nome junto ao rosto em {format_timestamp(item.timestamp)}")

    suggestions: list[VideoAuditSuggestion] = []
    for person in authorization_people:
        score, reasons = evidence.get(person.image_path, (0.0, set()))
        if not reasons:
            continue
        if len(reasons) >= 2 and any(reason.startswith("apresentação falada") for reason in reasons):
            score = max(score, 0.98)
        suggestions.append(
            VideoAuditSuggestion(
                pdf_path=person.image_path,
                confidence=score,
                reason=" + ".join(sorted(reasons)),
            )
        )
    return sorted(suggestions, key=lambda item: (-item.confidence, alpha_key(item.pdf_path.name)))


def merge_video_authorization_suggestions(
    *groups: Iterable[VideoAuditSuggestion],
) -> list[VideoAuditSuggestion]:
    """Combine independent evidence without duplicating an authorization."""
    merged: dict[Path, tuple[float, set[str]]] = {}
    for group in groups:
        for suggestion in group:
            path = suggestion.pdf_path
            score, reasons = merged.get(path, (0.0, set()))
            reasons.update(part.strip() for part in suggestion.reason.split(" + ") if part.strip())
            merged[path] = (max(score, suggestion.confidence), reasons)
    return sorted(
        (
            VideoAuditSuggestion(path, score, " + ".join(sorted(reasons)))
            for path, (score, reasons) in merged.items()
        ),
        key=lambda item: (-item.confidence, alpha_key(item.pdf_path.name)),
    )


def video_face_identity_suggestions(
    previews: list[VideoFacePreview],
    references: list[AuthorizationFaceReference],
) -> list[VideoAuditSuggestion]:
    """Match repeated foreground faces against local authorization portraits."""
    if not previews or not references:
        return []

    observations: dict[Path, list[tuple[float, float, float, VideoFacePreview]]] = {}
    people_by_path = {reference.person.image_path: reference.person for reference in references}
    for preview in previews:
        feature = _sface_feature_from_path(preview.image_path)
        if feature is None:
            continue
        scores_by_path: dict[Path, float] = {}
        for reference in references:
            path = reference.person.image_path
            score = _sface_similarity(feature, reference.feature)
            scores_by_path[path] = max(scores_by_path.get(path, -1.0), score)
        ranked = sorted(
            ((score, path) for path, score in scores_by_path.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked:
            continue
        best_score, best_path = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        if best_score >= FACE_REPEATED_SIMILARITY and margin >= FACE_MIN_MARGIN:
            observations.setdefault(best_path, []).append((best_score, margin, preview.timestamp, preview))

    suggestions: list[VideoAuditSuggestion] = []
    for path, records in observations.items():
        records.sort(key=lambda record: record[0], reverse=True)
        strong = records[0][0] >= FACE_STRONG_SIMILARITY
        repeated = len(records) >= 2 and sum(record[0] for record in records[:2]) / 2 >= 0.46
        if not strong and not repeated:
            continue
        best_score = records[0][0]
        if best_score >= 0.60:
            confidence = 0.96
        elif best_score >= 0.55:
            confidence = 0.92
        elif strong:
            confidence = 0.86
        else:
            confidence = 0.80
        timestamps = sorted({format_timestamp(record[2]) for record in records})[:3]
        reason = "rosto confirmado em " + ", ".join(timestamps)
        person = people_by_path.get(path)
        if person is not None:
            for _score, _margin, _timestamp, preview in records:
                if not preview.label:
                    preview.label = person.label
                    preview.authorization_path = person.image_path
        suggestions.append(VideoAuditSuggestion(path, confidence, reason))
    return sorted(suggestions, key=lambda item: (-item.confidence, alpha_key(item.pdf_path.name)))


def named_face_references(
    previews: list[VideoFacePreview],
    visual_evidence: list[VideoVisualEvidence],
    authorization_people: list[PersonPhoto],
) -> list[AuthorizationFaceReference]:
    """Turn an unambiguous lower-third into references for later videos."""
    if not previews or not visual_evidence or not authorization_people:
        return []
    roster = [person.label for person in authorization_people]
    roster_firsts = _roster_first_names(roster)
    token_counts = _person_token_counts(authorization_people)
    by_timestamp: dict[int, list[VideoFacePreview]] = {}
    for preview in previews:
        by_timestamp.setdefault(round(preview.timestamp * 10), []).append(preview)

    references: list[AuthorizationFaceReference] = []
    used: set[tuple[Path, Path]] = set()
    for item in visual_evidence:
        ranked = sorted(
            (
                (_visual_roster_score(person.label, item.text, roster_firsts, token_counts), person)
                for person in authorization_people
            ),
            key=lambda candidate: candidate[0],
            reverse=True,
        )
        if not ranked:
            continue
        best_score, best_person = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < 0.80 or (best_score < 0.88 and best_score - second_score < 0.10):
            continue
        nearby = by_timestamp.get(round(item.timestamp * 10), [])
        # A tarja identifica com segurança o rosto somente quando há uma pessoa
        # em primeiro plano naquele instante.
        if len(nearby) != 1:
            continue
        preview = nearby[0]
        key = (best_person.image_path, preview.image_path)
        if key in used:
            continue
        feature = _sface_feature_from_path(preview.image_path)
        if feature is None:
            continue
        preview.label = best_person.label
        preview.authorization_path = best_person.image_path
        used.add(key)
        references.append(
            AuthorizationFaceReference(
                person=best_person,
                feature=feature,
                source=f"nome junto ao rosto em {format_timestamp(item.timestamp)}",
            )
        )
    return references


def production_folder_for_video(video_path: Path, source_root: Path) -> Path:
    parent = video_path.parent
    if re.match(r"^\s*\d*\.?\s*video\s*$", normalize(parent.name)):
        parent = parent.parent
    try:
        parent.relative_to(source_root)
    except ValueError:
        return video_path.parent
    return parent


def authorization_output_folder(video_path: Path, source_root: Path, fallback_root: Path | None = None) -> Path:
    if fallback_root is None:
        return production_folder_for_video(video_path, source_root) / "AUTORIZACAO DE IMAGEM"
    try:
        relative_parent = production_folder_for_video(video_path, source_root).relative_to(source_root)
    except ValueError:
        relative_parent = Path(safe_filename(video_path.parent.name))
    return fallback_root / relative_parent / "AUTORIZACAO DE IMAGEM"


def copy_authorization_suggestions(
    suggestions: Iterable[VideoAuditSuggestion],
    destination: Path,
    face_photos: dict[Path, Path] | None = None,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    used = {path for path in destination.iterdir() if path.is_file()}
    available_faces = face_photos or {}
    for suggestion in suggestions:
        source = suggestion.pdf_path
        face_photo = available_faces.get(source)
        if source.suffix.lower() == ".pdf" and "semfoto" in normalize(source.stem) and face_photo:
            clean_stem = re.sub(r"(?i)[ _-]*SEMFOTO\b", "", source.stem).strip(" _-")
            filled_target = destination / f"{clean_stem or source.stem}.pdf"
            if filled_target.exists():
                copied.append(filled_target)
                continue
            try:
                page = load_source(source)
                models = load_models()
                model = classify_model(model_hint_ocr(page), models)
                composed = place_photo(page, face_photo, models[model]["foto"])
                save_pdf(composed, filled_target)
                used.add(filled_target)
                copied.append(filled_target)
                continue
            except Exception:
                # A recuperação da foto é complementar. Uma falha isolada não
                # impede que a autorização original seja copiada para o lote.
                pass
        existing = destination / source.name
        if existing.exists():
            try:
                if existing.stat().st_size == source.stat().st_size and file_digest(existing) == file_digest(source):
                    copied.append(existing)
                    continue
            except OSError:
                pass
        target = unique_path(existing, used)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def drive_file_id_from_url(url: str) -> str | None:
    patterns = [
        r"/file/d/([^/?#]+)",
        r"[?&]id=([^&#]+)",
        r"/uc\?id=([^&#]+)",
        r"/open\?id=([^&#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def drive_folder_id_from_url(url: str) -> str | None:
    patterns = [
        r"/folders/([^/?#]+)",
        r"[?&]id=([^&#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def canonical_drive_folder_url(url: str) -> str:
    folder_id = drive_folder_id_from_url(url)
    if not folder_id:
        raise ValueError("Cole o link de uma pasta do Google Drive, não o link de um arquivo isolado.")
    return f"https://drive.google.com/drive/folders/{folder_id}"


def drive_folder_links_from_text(text: str) -> list[str]:
    if not text.strip():
        return []
    clean_text = text.replace("\u200b", "").replace("\u200e", "").replace("\ufeff", "")
    # Links copied from chats sometimes arrive with a line break in the middle
    # of the query string, for example "usp=drive_\nfs". Rejoin only Drive URL
    # continuations so separate links still stay separate.
    previous = None
    while previous != clean_text:
        previous = clean_text
        clean_text = re.sub(
            r"(https?://drive\.google\.com/[^\s<>\"]*[=&_%])\s+([A-Za-z0-9_%=&?.-]+)",
            r"\1\2",
            clean_text,
        )
    candidates = re.findall(r"https?://[^\s<>\"]+", clean_text)
    if not candidates:
        candidates = [line.strip() for line in clean_text.splitlines() if line.strip()]
    links: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        url = candidate.strip().rstrip(").,;]")
        if "drive.google.com" not in url:
            continue
        if not drive_folder_id_from_url(url):
            continue
        canonical = canonical_drive_folder_url(url)
        if canonical not in seen:
            seen.add(canonical)
            links.append(canonical)
    return links


def safe_drive_relative_path(path_text: str) -> Path:
    parts: list[str] = []
    for raw in re.split(r"[\\/]+", path_text.strip()):
        if not raw or raw in {".", ".."}:
            continue
        clean = re.sub(r"[\x00-\x1f\\/:*?\"<>|]+", "-", raw).strip(" .")
        parts.append(clean or "arquivo")
    return Path(*parts) if parts else Path("arquivo")


def unique_path(path: Path, used: set[Path]) -> Path:
    candidate = path
    counter = 2
    while candidate in used:
        candidate = path.with_name(f"{path.stem} {counter}{path.suffix}")
        counter += 1
    used.add(candidate)
    return candidate


def file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_term_sources(paths: Iterable[Path]) -> tuple[list[Path], int]:
    unique: list[Path] = []
    seen_paths: set[Path] = set()
    seen_digests: set[tuple[str, int]] = set()
    duplicates = 0
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            duplicates += 1
            continue
        seen_paths.add(resolved)
        try:
            signature = (file_digest(path), path.stat().st_size)
        except OSError:
            unique.append(path)
            continue
        if signature in seen_digests:
            duplicates += 1
            continue
        seen_digests.add(signature)
        unique.append(path)
    return unique, duplicates


def list_public_drive_folder(
    url: str,
    destination: Path,
    skipped_duplicates: list[str] | None = None,
    supported: set[str] | None = None,
) -> list[DriveTermFile]:
    try:
        import gdown  # noqa: F401
        from importlib.metadata import version
    except ImportError as exc:
        raise RuntimeError("O componente do Google Drive não está instalado. Abra o aplicativo pelo arquivo .command.") from exc

    gdown_version = version("gdown")
    numbers = [int(item) for item in re.findall(r"\d+", gdown_version)[:2]]
    if len(numbers) < 2 or tuple(numbers) < (6, 1):
        raise RuntimeError(
            "O componente do Google Drive está desatualizado. Feche o app e abra novamente pelo arquivo "
            "'Abrir Automatizador.command' para instalar a versão nova."
        )

    folder_url = canonical_drive_folder_url(url)
    command = [sys.executable, "-m", "gdown", folder_url, "--folder", "--json", "--quiet"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("A listagem do Google Drive demorou demais. Tente novamente ou use a pasta sincronizada.") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if "--json" in detail or "unrecognized" in detail.lower():
            raise RuntimeError(
                "O componente do Google Drive está desatualizado. Feche o app e abra novamente pelo arquivo "
                "'Abrir Automatizador.command' para instalar a versão nova."
            )
        raise RuntimeError(detail or "Não consegui listar a pasta do Google Drive.")
    try:
        entries = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("O Google Drive devolveu uma listagem inesperada. Tente abrir novamente pelo .command.") from exc

    files: list[DriveTermFile] = []
    accepted = supported or SUPPORTED
    used_paths: set[Path] = set()
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path_text = str(entry.get("path", "")).strip()
        file_url = str(entry.get("url", "")).strip()
        if not path_text:
            continue
        listed_suffix = Path(path_text).suffix.lower()
        # O Google Drive permite remover a extensão visível de um vídeo sem
        # alterar seu tipo real. Esses itens precisam ser baixados primeiro;
        # o Content-Type define a extensão local depois do download.
        if listed_suffix and listed_suffix not in accepted:
            continue
        path_key = normalize(path_text) or path_text.casefold()
        if path_key in seen_paths:
            if skipped_duplicates is not None:
                skipped_duplicates.append(path_text)
            continue
        seen_paths.add(path_key)
        file_id = drive_file_id_from_url(file_url)
        if not file_id:
            continue
        relative = safe_drive_relative_path(path_text)
        local_path = unique_path(destination / relative, used_paths)
        files.append(DriveTermFile(id=file_id, path=path_text, local_path=local_path))
    return files


def download_public_drive_file(file_id: str, destination: Path) -> str:
    """Download one public Drive file without relying on gdown's HTML confirmation parser."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("O componente de conexão com o Google Drive não está instalado.") from exc

    direct_url = "https://drive.usercontent.google.com/download"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with requests.get(
            direct_url,
            params={"id": file_id, "export": "download", "confirm": "t"},
            headers=headers,
            stream=True,
            timeout=(20, 180),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if "text/html" in content_type:
                raise ValueError("o Google devolveu uma página de acesso em vez do arquivo")
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if not partial.exists() or partial.stat().st_size == 0:
            raise ValueError("o arquivo baixado está vazio")
        partial.replace(destination)
        return content_type
    finally:
        if partial.exists():
            partial.unlink(missing_ok=True)


def downloaded_file_suffix(path: Path, content_type: str, accepted: set[str]) -> str:
    suffix = DRIVE_CONTENT_TYPE_SUFFIXES.get(content_type, "")
    if suffix in accepted:
        return suffix
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return ""
    if header.startswith(b"%PDF") and ".pdf" in accepted:
        return ".pdf"
    if header.startswith(b"\xff\xd8\xff") and ".jpg" in accepted:
        return ".jpg"
    if header.startswith(b"\x89PNG\r\n\x1a\n") and ".png" in accepted:
        return ".png"
    if len(header) >= 12 and header[4:8] == b"ftyp" and accepted & VIDEO_SUPPORTED:
        # QuickTime and MP4 share the ISO-BMFF signature. The major brand keeps
        # the local extension faithful even when Drive sends octet-stream.
        if header[8:12].rstrip(b" \x00") == b"qt" and ".mov" in accepted:
            return ".mov"
        return ".mp4" if ".mp4" in accepted else ".mov"
    return ""


def available_download_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem} {counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def download_drive_folder(
    url: str,
    destination: Path,
    progress_callback=None,
    supported: set[str] | None = None,
    kind_label: str = "termos",
) -> tuple[Path, list[str], int]:
    if "drive.google.com" not in url or not drive_folder_id_from_url(url):
        raise ValueError("Cole o link de uma pasta do Google Drive, não o link de um arquivo isolado.")
    destination.mkdir(parents=True, exist_ok=True)
    # O gdown 6.1 lista pastas públicas pelo embeddedfolderview e não fica
    # preso ao antigo limite de 50 itens. O download segue direto pelo endpoint
    # de conteúdo porque ele é mais confiável para JPEGs públicos.
    skipped_duplicates: list[str] = []
    accepted = supported or SUPPORTED
    candidates = list_public_drive_folder(url, destination, skipped_duplicates, accepted)
    if not candidates:
        raise ValueError(
            f"Não encontrei {kind_label} nessa pasta. Confirme o link e o compartilhamento, "
            "ou selecione a pasta pelo Finder."
        )

    failures: list[str] = []
    started = time.monotonic()
    for index, item in enumerate(candidates, start=1):
        last_error: Exception | None = None
        for attempt in range(1, DRIVE_DOWNLOAD_ATTEMPTS + 1):
            try:
                content_type = download_public_drive_file(item.id, item.local_path)
                if item.local_path.suffix.lower() not in accepted:
                    inferred_suffix = downloaded_file_suffix(item.local_path, content_type, accepted)
                    if not inferred_suffix:
                        item.local_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"tipo de arquivo não reconhecido ({content_type or 'sem Content-Type'})"
                        )
                    inferred_path = available_download_path(item.local_path.with_suffix(inferred_suffix))
                    item.local_path.replace(inferred_path)
                    item.local_path = inferred_path
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                item.local_path.unlink(missing_ok=True)
                if attempt < DRIVE_DOWNLOAD_ATTEMPTS:
                    time.sleep(0.6 * attempt)
        if last_error is not None:
            failures.append(f"{Path(item.path).name}: {last_error}")
        if progress_callback:
            elapsed = time.monotonic() - started
            eta = (elapsed / index) * (len(candidates) - index)
            progress_callback(index, len(candidates), Path(item.path).name, eta)

    downloaded = [
        path for path in destination.rglob("*")
        if path.is_file() and path.suffix.lower() in accepted
    ]
    if not downloaded:
        detail = failures[0] if failures else "nenhum arquivo pôde ser baixado"
        raise ValueError(
            f"A pasta foi encontrada, mas o Google não liberou o download de {kind_label}. "
            f"Detalhe: {detail}. Como alternativa, escolha a pasta sincronizada pelo Finder."
        )
    return destination, failures, len(skipped_duplicates)


def load_source(path: Path) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        if not doc.page_count:
            raise ValueError(f"PDF vazio: {path.name}")
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.6, 2.6), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return image
    image = Image.open(path)
    try:
        image.draft("RGB", (MAX_PAGE_PROCESSING_EDGE, MAX_PAGE_PROCESSING_EDGE))
        return ImageOps.exif_transpose(image).convert("RGB")
    finally:
        image.close()


def order_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype("float32")
    result = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    result[0] = pts[np.argmin(sums)]      # top-left
    result[2] = pts[np.argmax(sums)]      # bottom-right
    result[1] = pts[np.argmin(diffs)]     # top-right
    result[3] = pts[np.argmax(diffs)]     # bottom-left
    return result


def best_document_quad(image_bgr: np.ndarray) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, 1500.0 / max(h, w))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale < 1 else image_bgr.copy()
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: list[tuple[float, np.ndarray]] = []
    edge = cv2.Canny(gray, 45, 135)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    masks = [edge]
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    masks.append(bright)

    image_area = float(sw * sh)
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.25:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype("float32")
            else:
                rect = cv2.minAreaRect(contour)
                pts = cv2.boxPoints(rect).astype("float32")
            ordered = order_points(pts)
            widths = [np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])]
            heights = [np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])]
            ratio = max(widths) / max(max(heights), 1.0)
            # Accept portrait pages with some perspective distortion.
            if 0.48 <= ratio <= 0.92:
                border_bonus = area / image_area
                candidates.append((border_bonus, ordered / scale))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def warp_document(image: Image.Image) -> Image.Image:
    if cv2 is None:
        return image
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    quad = best_document_quad(bgr)
    if quad is None:
        return image
    # When the sheet nearly fills the camera frame, contour detection can place
    # one corner slightly inside the paper and clip text near that edge. Snap
    # near-border corners back to the frame before the perspective transform.
    quad[:, 0] = np.where(quad[:, 0] < image.width * 0.12, 0, quad[:, 0])
    quad[:, 0] = np.where(quad[:, 0] > image.width * 0.88, image.width - 1, quad[:, 0])
    quad[:, 1] = np.where(quad[:, 1] < image.height * 0.08, 0, quad[:, 1])
    quad[:, 1] = np.where(quad[:, 1] > image.height * 0.90, image.height - 1, quad[:, 1])
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 300 or height < 500:
        return image
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    warped = cv2.warpPerspective(bgr, matrix, (width, height), borderValue=(255, 255, 255))
    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def simple_trim(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    gray = arr.mean(axis=2)
    row_ratio = (gray > 145).mean(axis=1)
    rows = np.where(row_ratio > 0.52)[0]
    if len(rows) > image.height * 0.35:
        top, bottom = int(rows[0]), int(rows[-1] + 1)
    else:
        top, bottom = 0, image.height
    cropped = image.crop((0, top, image.width, bottom))
    return cropped


def ensure_portrait(image: Image.Image) -> Image.Image:
    if image.width > image.height:
        return image.rotate(90, expand=True)
    return image


def limit_page_processing_size(image: Image.Image) -> Image.Image:
    if max(image.size) <= MAX_PAGE_PROCESSING_EDGE:
        return image
    copy = image.copy()
    copy.thumbnail((MAX_PAGE_PROCESSING_EDGE, MAX_PAGE_PROCESSING_EDGE), Image.Resampling.BICUBIC)
    return copy


def fill_short_false_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    result = values.copy()
    start = None
    for idx, value in enumerate(np.r_[result, True]):
        if not value and start is None:
            start = idx
        elif value and start is not None:
            if idx - start <= max_gap:
                result[start:idx] = True
            start = None
    return result


def trim_external_bands(image: Image.Image) -> Image.Image:
    """Remove dark table/folder bands that remain just outside the white paper."""
    gray = np.asarray(image.convert("L"))

    def longest_true_run(values: np.ndarray) -> tuple[int, int]:
        best = (0, len(values))
        start = None
        best_len = -1
        for idx, value in enumerate(np.r_[values, False]):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                if idx - start > best_len:
                    best, best_len = (start, idx), idx - start
                start = None
        return best

    # Printed/written form lines can make a row briefly dark. Close these small
    # gaps before choosing the page band, otherwise the top identity block can be
    # cropped away on photos with lots of handwriting.
    row_is_paper = (gray > 145).mean(axis=1) > 0.50
    col_is_paper = (gray > 145).mean(axis=0) > 0.50
    row_is_paper = fill_short_false_gaps(row_is_paper, max(12, int(image.height * 0.035)))
    col_is_paper = fill_short_false_gaps(col_is_paper, max(8, int(image.width * 0.025)))
    y0, y1 = longest_true_run(row_is_paper)
    x0, x1 = longest_true_run(col_is_paper)
    if (y1 - y0) < image.height * 0.70:
        y0, y1 = 0, image.height
    else:
        pad_y = int(image.height * 0.006)
        y0, y1 = max(0, y0 - pad_y), min(image.height, y1 + pad_y)
    if (x1 - x0) < image.width * 0.70:
        x0, x1 = 0, image.width
    else:
        pad_x = int(image.width * 0.006)
        x0, x1 = max(0, x0 - pad_x), min(image.width, x1 + pad_x)
    return image.crop((x0, y0, x1, y1))


def normalize_page(image: Image.Image) -> Image.Image:
    image = limit_page_processing_size(image)
    page = ensure_portrait(trim_external_bands(ensure_portrait(warp_document(image))))
    # The supplied terms are A4. Normalizing to the known page ratio removes the
    # remaining camera perspective and ensures the photo box stays on the paper.
    page = page.resize(A4_SIZE, Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(page).enhance(1.035)


def _known_name_hints(name_lines: list[str], custom_names: tuple[str, ...]) -> list[str]:
    if not name_lines or not custom_names:
        return []
    pseudo_people = [
        PersonPhoto(label=name, caption="", image_path=Path(name))
        for name in custom_names
        if name and not unidentified_person_label(name)
    ]
    token_counts = _person_token_counts(pseudo_people)
    hints: list[str] = []
    seen: set[str] = set()
    for line in name_lines:
        scored = sorted(
            (
                (_candidate_match_score(person.label, line, token_counts), person.label)
                for person in pseudo_people
            ),
            reverse=True,
        )
        if not scored:
            continue
        best_score, best_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.82 or (best_score >= 0.58 and best_score - second_score >= 0.18):
            key = normalize(best_name)
            if key not in seen:
                seen.add(key)
                hints.append(best_name)
    return hints


def _decorate_ocr_output(outputs: list[str], custom_names: tuple[str, ...] = ()) -> str:
    name_lines: list[str] = []
    seen_candidates: set[str] = set()

    def add_candidate(raw: str) -> None:
        cleaned = _name_like_fragment(raw) or _clean_person_line(raw)
        key = normalize(cleaned)
        if key and key not in seen_candidates and _valid_name_candidate(cleaned):
            seen_candidates.add(key)
            name_lines.append(cleaned)

    def add_plausible_free_line(raw: str) -> None:
        cleaned = _name_like_fragment(raw) or _clean_person_line(raw)
        tokens = _name_tokens(cleaned)
        if not tokens or tokens[0] not in accepted_first_names():
            return
        if not (1 <= len(tokens) <= 5):
            return
        if any(token in HARD_NON_NAME_WORDS for token in normalize(cleaned).split()):
            return
        key = normalize(cleaned)
        if key and key not in seen_candidates:
            seen_candidates.add(key)
            name_lines.append(cleaned)

    for output in outputs:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        focused_output = len(lines) <= 18 or any(
            "pelo presente instrumento" in normalize(line)
            for line in lines[:4]
        )
        for idx, line in enumerate(lines):
            norm_line = normalize(line)
            if focused_output:
                add_plausible_free_line(line)
                continue
            if not any(
                marker in norm_line
                for marker in ("instrumento", "instrumendo", "instumento", "tumento", "umento", "responsabilidade")
            ):
                continue
            for end in range(idx, min(len(lines), idx + 4)):
                window = " ".join(lines[idx:end + 1])
                add_candidate(window)
                following_norm = normalize(lines[end])
                if any(stop in following_norm for stop in ("cpf", "portador", "porador", "patador", "potador", "residente", "cedente")):
                    break
        if focused_output:
            limit = min(len(lines), 7)
            for start in range(limit):
                for end in range(start, min(limit, start + 4)):
                    window = " ".join(lines[start:end + 1])
                    add_candidate(window)
                    add_plausible_free_line(window)
    name_lines.extend(_known_name_hints(name_lines, custom_names))
    return "\n".join(outputs) + "\nNOME_CANDIDATO\n" + "\n".join(name_lines)


@lru_cache(maxsize=12)
def _custom_ocr_words(custom_names: tuple[str, ...]) -> tuple[str, ...]:
    words: list[str] = []
    words.extend(_name_database_words(OCR_COMMON_FIRST_NAME_LIMIT, OCR_COMMON_SURNAME_LIMIT))
    for name in custom_names:
        if not name or normalize(name) in NON_PERSON_CAPTIONS:
            continue
        words.append(name)
        words.extend(word for word in name.split() if len(normalize(word)) >= 3)
    return tuple(list(dict.fromkeys(words))[:3600])


def _ocr_regions(page: Image.Image) -> list[Image.Image]:
    width, height = page.size
    boxes = [
        # Termos com clausulas trazem o cedente numa faixa fixa logo abaixo do
        # titulo. Ler primeiro essa area reduz nomes vindos das clausulas.
        (int(width * 0.015), int(height * 0.050), int(width * 0.930), int(height * 0.175)),
        # O modelo de termo de responsabilidade escreve o nome um pouco mais abaixo.
        (0, int(height * 0.140), width, int(height * 0.345)),
    ]
    return [page.crop(box) for box in boxes]


def _model_hint_regions(page: Image.Image) -> list[Image.Image]:
    width, height = page.size
    return [
        page.crop((0, int(height * 0.285), width, int(height * 0.430))),
    ]


def _color_ink_mask(region: Image.Image) -> Image.Image | None:
    arr = np.asarray(region.convert("RGB")).astype(np.int16)
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red_ink = ((red - green) > 18) & ((red - blue) > 8) & (red > 80) & (green < 225) & (blue < 225)
    blue_ink = ((blue - red) > 10) & ((blue - green) > 4) & (blue > 70) & (red < 215) & (green < 225)
    mask = red_ink | blue_ink
    if mask.mean() < 0.001:
        return None
    ink = Image.fromarray(np.where(mask, 0, 255).astype("uint8"), "L")
    ink = ink.filter(ImageFilter.MinFilter(3))
    ink = ImageEnhance.Contrast(ink).enhance(1.8)
    return ink.convert("RGB")


def _detected_term_title_top(page: Image.Image) -> float | None:
    """Locate the printed title so the name strip follows shifted scans."""
    gray = np.asarray(page.convert("L"))
    height, width = gray.shape
    if height < 100 or width < 100:
        return None
    top_limit = max(1, int(height * 0.32))
    left, right = int(width * 0.20), int(width * 0.80)
    band = gray[:top_limit, left:right]
    if not band.size:
        return None
    row_density = (band < 165).mean(axis=1)
    active = row_density > 0.018
    minimum_rows = max(3, int(round(height * 0.003)))
    clusters: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= minimum_rows:
                clusters.append((start, index))
            start = None
    if start is not None and len(active) - start >= minimum_rows:
        clusters.append((start, len(active)))

    for cluster_top, cluster_bottom in clusters:
        normalized_top = cluster_top / height
        normalized_height = (cluster_bottom - cluster_top) / height
        peak_density = float(row_density[cluster_top:cluster_bottom].max(initial=0.0))
        if (
            0.025 <= normalized_top <= 0.235
            and 0.003 <= normalized_height <= 0.025
            and peak_density >= 0.07
        ):
            return normalized_top
    return None


def _adaptive_name_strip(page: Image.Image, *, guided: bool = False) -> tuple[float, float, float, float] | None:
    title_top = _detected_term_title_top(page)
    if title_top is None or abs(title_top - 0.095) < 0.025:
        return None
    vertical_margin = 0.065 if guided else 0.055
    return (
        0.160 if guided else 0.205,
        max(0.0, title_top - vertical_margin),
        0.930 if guided else 0.895,
        min(0.32, title_top + vertical_margin),
    )


def _focused_name_region_boxes(page: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = page.size
    legacy_box = (
        # Termo com clausulas: somente a faixa do nome manuscrito depois de
        # "Pelo presente instrumento, eu". Não varremos cláusulas nem outros
        # campos manuscritos.
        0.205, 0.040, 0.895, 0.135,
    )
    adaptive_box = _adaptive_name_strip(page)
    normalized_boxes = [adaptive_box, legacy_box] if adaptive_box else [legacy_box]
    return [
        (
            int(width * left),
            int(height * top),
            int(width * right),
            int(height * bottom),
        )
        for left, top, right, bottom in normalized_boxes
    ]


def _guided_name_region_boxes(page: Image.Image) -> list[tuple[int, int, int, int]]:
    width, height = page.size
    # Todas as caixas continuam restritas à linha imediatamente após a âncora.
    # As pequenas variações compensam fotos inclinadas sem reler o contrato.
    normalized_boxes: list[tuple[float, float, float, float]] = [
        (0.160, 0.020, 0.820, 0.150),
        (0.205, 0.035, 0.895, 0.140),
        (0.180, 0.065, 0.780, 0.125),
    ]
    adaptive_box = _adaptive_name_strip(page, guided=True)
    if adaptive_box:
        normalized_boxes.insert(0, adaptive_box)
    return [
        (int(width * left), int(height * top), int(width * right), int(height * bottom))
        for left, top, right, bottom in normalized_boxes
    ]


def guided_name_raw_ocr(page: Image.Image, custom_names: Iterable[str]) -> str:
    hints = tuple(dict.fromkeys(name for name in custom_names if name))
    outputs: list[str] = []
    seen: set[str] = set()
    for box in _guided_name_region_boxes(page):
        region = page.crop(box)
        variants = [region]
        ink = _color_ink_mask(region)
        if ink is not None:
            variants.append(ink)
        for variant in variants:
            try:
                text = (
                    _vision_region_text(variant, hints, fast=True, alternatives=3)
                    if sys.platform == "darwin"
                    else ""
                )
            except Exception:
                text = ""
            key = normalize(text)
            if key and key not in seen:
                seen.add(key)
                outputs.append(text)
    return "\n".join(outputs)


def _focused_name_regions(page: Image.Image) -> Iterable[tuple[int, Image.Image]]:
    for field_index, box in enumerate(_focused_name_region_boxes(page)):
        region = page.crop(box)
        yield field_index, region


def focused_name_ocr(
    page: Image.Image,
    custom_names: list[str] | tuple[str, ...] | None = None,
    *,
    max_regions: int | None = None,
) -> str:
    custom_names_tuple = tuple(dict.fromkeys(custom_names or []))
    outputs: list[str] = []
    for field_index, region in _focused_name_regions(page):
        if max_regions is not None and field_index >= max_regions:
            break
        variants = [region]
        ink = _color_ink_mask(region)
        if ink is not None:
            variants.append(ink)
        for variant in variants:
            try:
                text = _vision_region_text(variant, custom_names_tuple) if sys.platform == "darwin" else ""
            except Exception:
                text = ""
            if text.strip():
                outputs.append(text)
                decorated = _decorate_ocr_output(outputs, custom_names_tuple)
                if extract_name_candidate(decorated):
                    return decorated
    return _decorate_ocr_output(outputs, custom_names_tuple) if outputs else ""


def enrich_name_field_from_raw_ocr(ocr_text: str) -> str:
    _, after_marker = _marker_sections(ocr_text)
    if after_marker.strip():
        return ocr_text
    lines: list[str] = []
    for raw in ocr_text.splitlines():
        cleaned = _name_like_fragment(raw) or _clean_person_line(raw)
        if _valid_name_candidate(cleaned):
            lines.append(cleaned)
    if not lines:
        return ocr_text
    ranked = _rank_name_candidates(lines)
    if not ranked:
        return ocr_text
    return f"{ocr_text}\nNOME_CANDIDATO\n" + "\n".join(ranked[:5])


def _prepare_ocr_region(region: Image.Image, max_edge: int = MAX_OCR_REGION_EDGE) -> Image.Image:
    crop = ImageOps.autocontrast(ImageOps.grayscale(region))
    crop = ImageEnhance.Contrast(crop).enhance(1.45)
    crop = ImageEnhance.Sharpness(crop).enhance(1.20)
    min_width = 720
    min_height = 420
    if crop.width < min_width or crop.height < min_height:
        padded = Image.new("L", (max(crop.width, min_width), max(crop.height, min_height)), "white")
        padded.paste(crop, ((padded.width - crop.width) // 2, (padded.height - crop.height) // 2))
        crop = padded
    if crop.height and crop.width / crop.height > 2.4:
        padded_height = max(crop.height, round(crop.width / 2.1))
        padded = Image.new("L", (crop.width, padded_height), "white")
        padded.paste(crop, (0, (padded_height - crop.height) // 2))
        crop = padded
    if max(crop.size) > max_edge:
        crop.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return crop


def _vision_region_text(
    region: Image.Image,
    custom_names: tuple[str, ...],
    *,
    fast: bool = True,
    alternatives: int = 1,
) -> str:
    import Foundation
    import Vision
    import objc

    def make_request():
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        request.setRecognitionLanguages_(["pt-BR", "en-US"])
        request.setMinimumTextHeight_(0.006)
        custom_words = _custom_ocr_words(custom_names)
        if custom_words:
            # PyObjC nem sempre converte uma lista Python de forma compatível
            # com o NSArray<NSString *> esperado pelo Vision. A conversão
            # explícita evita que o macOS descarte silenciosamente as pistas.
            native_words = [Foundation.NSString.stringWithString_(word) for word in custom_words]
            request.setCustomWords_(Foundation.NSArray.arrayWithArray_(native_words))
        return request

    def run_handler(handler) -> str:
        request = make_request()
        success, error = handler.performRequests_error_([request], None)
        if not success:
            raise RuntimeError(f"A leitura nativa do macOS falhou: {error}")
        observations = list(request.results() or [])
        observations.sort(
            key=lambda item: (
                -float(item.boundingBox().origin.y),
                float(item.boundingBox().origin.x),
            )
        )
        lines: list[str] = []
        seen_lines: set[str] = set()
        for observation in observations:
            candidates = observation.topCandidates_(max(1, alternatives))
            for candidate in candidates:
                line = str(candidate.string()).strip()
                key = normalize(line)
                if key and key not in seen_lines:
                    seen_lines.add(key)
                    lines.append(line)
        return "\n".join(lines)

    last_error: Exception | None = None
    edges = (FAST_OCR_REGION_EDGE,) if fast else (MAX_OCR_REGION_EDGE, 1400, 1200, 1024, 768, 512)
    for edge in edges:
        crop = _prepare_ocr_region(region, edge)
        with objc.autorelease_pool():
            buffer = io.BytesIO()
            crop.convert("RGB").save(buffer, format="PNG")
            payload = buffer.getvalue()
            data = Foundation.NSData.dataWithBytes_length_(payload, len(payload))
            data_handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
            try:
                return run_handler(data_handler)
            except Exception as data_error:
                last_error = data_error
                with tempfile.TemporaryDirectory(prefix="termos_vision_") as folder:
                    image_path = Path(folder) / "ocr_region.png"
                    crop.convert("RGB").save(image_path, "PNG")
                    url = Foundation.NSURL.fileURLWithPath_(str(image_path))
                    file_handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
                    try:
                        return run_handler(file_handler)
                    except Exception as file_error:
                        last_error = file_error
                        continue
    if last_error:
        raise last_error
    return ""


def _vision_ocr(page: Image.Image, custom_names: tuple[str, ...]) -> list[str]:
    regions = _ocr_regions(page)
    outputs: list[str] = []
    last_error: Exception | None = None
    for region in regions:
        try:
            text = _vision_region_text(region, custom_names)
        except Exception as exc:
            last_error = exc
            continue
        if text.strip():
            outputs.append(text)
            decorated = _decorate_ocr_output(outputs, custom_names)
            if extract_name_candidate(decorated):
                break
    if not outputs and last_error:
        raise last_error
    return outputs


def _tesseract_ocr(page: Image.Image) -> list[str]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return []
    language_check = subprocess.run([tesseract, "--list-langs"], text=True, capture_output=True, check=False)
    languages = set(language_check.stdout.splitlines())
    language = "por+eng" if "por" in languages and "eng" in languages else ("por" if "por" in languages else "eng")
    regions = _ocr_regions(page)
    outputs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="termos_ocr_") as folder:
        for idx, region in enumerate(regions):
            crop = _prepare_ocr_region(region)
            tmp_path = Path(folder) / f"ocr_{idx}.png"
            crop.save(tmp_path)
            proc = subprocess.run(
                [tesseract, str(tmp_path), "stdout", "--psm", "6", "-l", language],
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
            )
            if proc.stdout.strip():
                outputs.append(proc.stdout.strip())
    return outputs


def ocr_engine_name() -> str:
    if sys.platform == "darwin":
        try:
            import Foundation  # noqa: F401
            import Vision  # noqa: F401
            import objc  # noqa: F401
            return "Leitura nativa do macOS"
        except ImportError:
            return ""
    return "Tesseract" if shutil.which("tesseract") else ""


def run_ocr(page: Image.Image, custom_names: list[str] | None = None) -> str:
    custom_names_tuple = tuple(dict.fromkeys(custom_names or []))
    outputs: list[str] = []
    vision_error: Exception | None = None
    if sys.platform == "darwin":
        try:
            outputs = _vision_ocr(page, custom_names_tuple)
        except ImportError:
            outputs = []
        except Exception as exc:
            vision_error = exc
            outputs = []
    if not outputs:
        outputs = _tesseract_ocr(page)
    if not outputs:
        if vision_error:
            raise RuntimeError(
                "O leitor automático de texto do macOS falhou e não encontrei outro leitor local disponível. "
                f"Detalhe: {vision_error}"
            ) from vision_error
        raise RuntimeError(
            "O leitor automático de texto não está disponível. Abra novamente pela versão atualizada do aplicativo para concluir a instalação."
        )
    return _decorate_ocr_output(outputs, custom_names_tuple)


def model_hint_ocr(page: Image.Image, custom_names: list[str] | tuple[str, ...] | None = None) -> str:
    custom_names_tuple = tuple(dict.fromkeys(custom_names or []))
    outputs: list[str] = []
    for region in _model_hint_regions(page):
        try:
            text = _vision_region_text(region, custom_names_tuple) if sys.platform == "darwin" else ""
        except Exception:
            text = ""
        if text.strip():
            outputs.append(text)
    return "\n".join(outputs)


def load_models() -> dict[str, dict]:
    return json.loads((APP_DIR / "modelos.json").read_text(encoding="utf-8"))


def classify_model(ocr_text: str, models: dict[str, dict]) -> str:
    norm = normalize(ocr_text)
    best_name = "Termo com clausulas" if "Termo com clausulas" in models else next(iter(models))
    best_hits = 0
    for name, config in models.items():
        hits = sum(normalize(marker) in norm for marker in config.get("marcadores", []))
        if hits > best_hits:
            best_name, best_hits = name, hits
    return best_name


NAME_BLOCKLIST = {
    "autorizo", "autorizacao", "autorizacao", "imagem", "termo", "responsabilidade",
    "cpf", "profissao", "profissional", "estado", "civil", "inscrito", "inscrita",
    "residente", "domiciliado", "domiciliada", "avenida", "rua", "bairro", "cidade",
    "salvador", "bahia", "telefone", "contato", "assinatura", "agencia",
    "comunicacao", "marketing", "publicitarias", "campanhas", "presente",
    "particular", "uso", "som", "cedula", "identidade", "portador", "portadora",
    "img", "vid", "mov", "video", "wa", "dsc", "pxl", "scan", "digitalizacao",
    "nome", "lido", "nao", "arquivo", "pelo", "instrumento", "cedente",
    "cessionario", "entidade", "administradora", "faixa", "clausula",
    "cedo", "cedido", "cedida",
    "todo", "toda", "todos", "todas", "declaro", "direito", "direitos",
    "lgpd", "dados", "pessoais", "objeto", "gratuidade", "vigencia",
    "revogacao", "disposicoes", "finais", "protecao", "brasileiro",
    "brasileira", "nacional", "internacional", "internet", "streaming",
    "aplicativos", "moveis", "publicacoes", "oficiais", "jpg", "jpeg",
    "png", "pdf",
    "aparece", "atitude", "brasileiros", "brasileiras", "caia", "caixa",
    "casta", "chira", "com", "cortador", "cortadora", "cortadores",
    "cortadoras", "cotador", "cotadora", "deus", "fazer", "lhe", "mais", "manter",
    "mantem", "mexer", "modafrates", "motei", "mudei", "outro", "outra",
    "pagar", "pago", "remundo", "sem", "sobre",
    "desidenta", "domiciado", "domiciliario", "domiriliario", "sresidente",
    "coronel", "ente", "cente", "cep", "idade", "temo",
    "doravante", "doravane", "doravant",
    "pojo", "polo", "palo", "presento", "prosente", "prosonte", "presonto",
    "instrumendo", "instrumonto", "instumento", "prosonto", "tumento", "umento",
    "deu", "eua", "meu", "nunoitesema", "seu", "seup", "teseup", "torno", "unb",
}


NAME_FIELD_START_PATTERNS = [
    r"pelo\s+presente\s+instrumento\s*,?\s*eu\s*,?",
    r"(?:instrumento|umento)\s*,?\s*eu\s*,?",
    r"pelo\s+presente\s+termo\s+particular\s+de\s+responsabilidade\s*:?",
    r"responsabilidade\s*:?",
]

NAME_FIELD_STOP_PATTERN = (
    r"\b(?:cpf|cpf/mf|portador|portadora|portado|porta|porador|poradora|"
    r"poradoras|poradores|patador|patadora|patadoras|patadores|"
    r"cortador|cortadora|cortadoras|cortadores|cotador|cotadora|"
    r"potador|potadora|potadoras|potadores|portadoras|portadores|"
    r"residente|sresidente|tesidente|desidenta|domiciliado|domiciliada|"
    r"domiciado|domiciliario|domiriliario|"
    r"bairro|cep|cidade|estado|cedente|autorizo|cedo|clausula|cláusula|"
    r"entidade|cessionario|cessionário)\b"
)


def _marker_sections(ocr_text: str) -> tuple[str, str]:
    if "NOME_CANDIDATO" not in ocr_text:
        return ocr_text, ""
    before, after = ocr_text.split("NOME_CANDIDATO", 1)
    return before, after


def _clean_person_line(raw: str) -> str:
    raw = re.sub(r"[_|]+", " ", raw)
    raw = _strip_form_intro_noise(raw)
    raw = re.sub(r"^\s*(?:pelo\s+presente\s+)?(?:instrumento|umento|termo).*?\beu\b\s*,?\s*", "", raw, flags=re.I)
    raw = re.sub(r"^\s*pelo\s+presente\s+termo\s+particular\s+de\s+responsabilidade\s*:?\s*", "", raw, flags=re.I)
    raw = re.sub(r"^\s*(?:eu|cedente)\s*,?\s*", "", raw, flags=re.I)
    raw = re.sub(
        r"^\s*(?:(?:pelo|polo|pojo|palo|presente|presento|prosente|prosonte|presonto|"
        r"instrumento|instrumendo|instrumonto|umento|termo|eu|ou)\b\s*)+",
        "",
        raw,
        flags=re.I,
    )
    raw = re.split(NAME_FIELD_STOP_PATTERN, raw, maxsplit=1, flags=re.I)[0]
    raw = re.sub(r"\d+", " ", raw)
    raw = re.sub(
        r"^\s*(?:(?:pelo|polo|pojo|palo|presente|presento|prosente|prosonte|presonto|"
        r"instrumento|instrumendo|instrumonto|umento|termo|eu|ou)\b\s*,?\s*)+",
        "",
        raw,
        flags=re.I,
    )
    raw = re.sub(r"[^A-Za-zÀ-ÿ' ]+", " ", raw)
    raw = re.sub(r"\b[A-Za-zÀ-ÿ]\b", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" .,-:")
    raw = _correct_fuzzy_first_name(raw)
    return _polish_person_name(raw)


def _polish_person_name(raw: str) -> str:
    words: list[str] = []
    for word in re.sub(r"\s+", " ", raw).strip().split():
        token = normalize(word)
        corrected = OCR_NAME_TOKEN_CORRECTIONS.get(token)
        if corrected:
            words.append(corrected)
            continue
        if token in {"de", "da", "do", "dos", "das"}:
            if words:
                words.append(word.lower())
            continue
        if len(token) < 3:
            continue
        if (
            token in NAME_BLOCKLIST
            or token in HARD_NON_NAME_WORDS
            or token in VIDEO_SPOKEN_NON_NAME_WORDS
            or token in FORM_INTRO_NOISE_WORDS
        ):
            continue
        words.append(word)
    while words and normalize(words[-1]) in {"de", "da", "do", "dos", "das"}:
        words.pop()
    return re.sub(r"\s+", " ", " ".join(words)).strip(" .,-:")


def _canonical_person_name(raw: str) -> str:
    return _polish_person_name(_clean_person_line(raw))


def _name_like_fragment(raw: str) -> str:
    stop_match = re.search(NAME_FIELD_STOP_PATTERN, raw, flags=re.I)
    if stop_match:
        raw = raw[:stop_match.start()]
    for match in re.finditer(r"[A-Za-zÀ-ÿ']+", raw):
        fragment = raw[match.start():]
        cleaned = _clean_person_line(fragment)
        if _valid_name_candidate(cleaned):
            return cleaned
    return ""


def _strip_form_intro_noise(raw: str) -> str:
    words = list(re.finditer(r"[A-Za-zÀ-ÿ]+", raw))
    if not words:
        return raw
    tokens = [normalize(match.group(0)) for match in words]

    def similar(token: str, expected: str, threshold: float = 0.72) -> bool:
        return token == expected or SequenceMatcher(None, token, expected).ratio() >= threshold

    for idx, token in enumerate(tokens[:12]):
        intro_like = (
            token in {
                "instrumento", "instrumendo", "instrumonto", "instumento",
                "prosonto", "tumento", "umento", "termo",
            }
            or similar(token, "instrumento")
            or similar(token, "termo", 0.80)
        )
        if not intro_like:
            continue
        before = tokens[max(0, idx - 5):idx]
        has_context = any(
            part in FORM_INTRO_NOISE_WORDS
            or similar(part, "presente")
            or similar(part, "pelo", 0.75)
            for part in before
        )
        if not has_context and token not in {"termo"} and idx != 0:
            continue
        cut_idx = idx
        for next_idx in range(idx + 1, min(len(tokens), idx + 4)):
            if tokens[next_idx] in {"eu", "ou"}:
                cut_idx = next_idx
                break
        return raw[words[cut_idx].end():]
    return raw


def _name_tokens(text: str) -> list[str]:
    return [
        token for token in normalize(text).split()
        if len(token) >= 3 and any(ch.isalpha() for ch in token) and token not in NAME_BLOCKLIST
    ]


def _correct_fuzzy_first_name(text: str) -> str:
    parts = re.sub(r"\s+", " ", text).strip().split()
    if not parts:
        return text
    first = normalize(parts[0])
    corrected_first = _correct_fuzzy_first_token(first)
    if corrected_first and corrected_first != first:
        parts[0] = corrected_first.title()
        return " ".join(parts)
    return text


@lru_cache(maxsize=12000)
def _correct_fuzzy_first_token(first: str) -> str:
    if not first or first in accepted_first_names():
        return first
    corrected = normalize(OCR_NAME_TOKEN_CORRECTIONS.get(first, ""))
    if corrected and corrected in accepted_first_names():
        return corrected
    if first in HARD_NON_NAME_WORDS or first in VIDEO_SPOKEN_NON_NAME_WORDS or first in FORM_INTRO_NOISE_WORDS:
        return first
    if len(first) < 4:
        return first
    buckets = fuzzy_first_name_buckets()
    candidates: list[str] = []
    for length in range(len(first) - 3, len(first) + 4):
        candidates.extend(buckets.get(length, ()))
    if not candidates:
        return first
    scored = sorted(
        ((SequenceMatcher(None, first, known).ratio(), known) for known in candidates),
        reverse=True,
    )
    best_score, best_name = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    threshold = 0.86 if len(first) <= 5 else 0.80
    if best_score >= threshold and best_score - second_score >= 0.12:
        return best_name
    return first


def _person_token_counts(people: list[PersonPhoto]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for person in people:
        if not person_is_identified(person):
            continue
        for token in set(_name_tokens(person.label)):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _first_name_token(name: str) -> str:
    tokens = _name_tokens(name)
    return tokens[0] if tokens else ""


def person_is_identified(person: PersonPhoto) -> bool:
    if unidentified_person_label(person.label):
        return False
    norm = normalize(person.label)
    if any(token in HARD_NON_NAME_WORDS for token in norm.split()):
        return False
    first = norm.split()[0] if norm.split() else ""
    if first in VIDEO_NON_NAME_STARTS:
        return False
    tokens = _name_tokens(person.label)
    if person.caption.startswith("Vídeo:") and not _valid_video_name(person.label):
        return False
    if len(tokens) == 1 and re.search(r"\b(?:img|vid|mov|video|wa|dsc|pxl|scan|jpg|jpeg|png|pdf)\b", norm):
        return False
    return bool(tokens)


def matchable_people(people: list[PersonPhoto]) -> list[PersonPhoto]:
    return [person for person in people if person_is_identified(person)]


def source_name_candidate(path: Path) -> str:
    label = label_from_photo_path(path)
    raw_tokens = normalize(label).split()
    if any(
        token in HARD_NON_NAME_WORDS
        or token in VIDEO_SPOKEN_NON_NAME_WORDS
        or token in FORM_INTRO_NOISE_WORDS
        for token in raw_tokens
    ):
        return ""
    cleaned_label = _canonical_person_name(label)
    norm = normalize(cleaned_label)
    if not norm:
        return ""
    if generic_video_file_label(label):
        return ""
    if re.match(
        r"^(?:digitalizacao|digitalizacao em|captura|scan|img|vid|mov|video|wa|pxl|dsc|"
        r"termo|termos|documento|arquivo|whatsapp)\b",
        norm,
    ):
        return ""
    tokens = _name_tokens(cleaned_label)
    if not (1 <= len(tokens) <= 7):
        return ""
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return ""
    if not any(token not in NAME_BLOCKLIST for token in tokens):
        return ""
    if not _valid_file_person_name(cleaned_label, allow_single=True):
        return ""
    return re.sub(r"\s+", " ", cleaned_label).strip()


def _valid_name_candidate(line: str) -> bool:
    raw_tokens = normalize(line).split()
    if any(token in HARD_NON_NAME_WORDS or token in VIDEO_SPOKEN_NON_NAME_WORDS for token in raw_tokens):
        return False
    if any(token in FORM_INTRO_NOISE_WORDS for token in raw_tokens[:4]):
        return False
    line = _canonical_person_name(line)
    tokens = _name_tokens(line)
    if not (1 <= len(tokens) <= 7):
        return False
    if len(" ".join(tokens)) < 3:
        return False
    if len(tokens) == 1:
        if len(tokens[0]) < 4:
            return False
        if tokens[0] not in curated_first_names():
            return False
        if tokens[0] in common_surnames() and tokens[0] not in curated_first_names():
            return False
    if tokens and tokens[0] not in accepted_first_names():
        return False
    if len(tokens) >= 2:
        corrected_tokens = {normalize(value) for value in OCR_NAME_TOKEN_CORRECTIONS.values()}
        supported_tail = [
            token for token in tokens[1:]
            if token in common_surnames()
            or token in known_first_names()
            or token in corrected_tokens
        ]
        if not supported_tail:
            return False
        suspicious_unknowns = [
            token for token in tokens[1:]
            if token not in common_surnames()
            and token not in known_first_names()
            and token not in corrected_tokens
            and len(token) >= 5
        ]
        if len(suspicious_unknowns) >= max(2, len(tokens) - 1):
            # Let rare surnames pass when the first name is strong, but reject
            # long OCR phrases whose remaining words are all unknown.
            return False
    return True


def _rank_name_candidates(lines: Iterable[str]) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for raw in lines:
        line = _clean_person_line(raw)
        if not _valid_name_candidate(line):
            continue
        tokens = _name_tokens(line)
        letters = [ch for ch in line if ch.isalpha()]
        uppercase_ratio = sum(ch.isupper() for ch in letters) / max(1, len(letters))
        surname_bonus = round(_surname_support_score(tokens) * 100)
        first_bonus = 10 if tokens and tokens[0] in accepted_first_names() else 0
        score = (
            len(tokens) * 18
            + len(" ".join(tokens))
            + (12 if uppercase_ratio >= 0.65 else 0)
            + first_bonus
            + surname_bonus
        )
        candidates.append((score, line))
    if not candidates:
        return []
    candidates.sort(reverse=True)
    unique: list[str] = []
    seen: set[str] = set()
    for _, candidate in candidates:
        key = normalize(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _line_window_name_candidates(ocr_text: str) -> list[str]:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    windows: list[str] = []
    for idx, line in enumerate(lines):
        norm_line = normalize(line)
        if not any(
            re.search(pattern, norm_line, flags=re.I)
            for pattern in NAME_FIELD_START_PATTERNS
        ):
            continue
        for end in range(idx, min(len(lines), idx + 3)):
            windows.append(" ".join(lines[idx:end + 1]))
    return _rank_name_candidates(windows)


def _inline_name_candidates(ocr_text: str) -> list[str]:
    flattened = re.sub(r"\s+", " ", ocr_text)
    windows: list[str] = []
    for pattern in NAME_FIELD_START_PATTERNS:
        for match in re.finditer(pattern, flattened, flags=re.I):
            fragment = flattened[match.start(): match.start() + 220]
            windows.append(fragment)
    return _rank_name_candidates(windows)


def extracted_name_candidates(ocr_text: str) -> list[str]:
    before_marker, after_marker = _marker_sections(ocr_text)
    marker_candidates = _rank_name_candidates(after_marker.splitlines())
    # O nome do termo só deve vir de uma leitura focada do campo manuscrito
    # ou de uma pista explícita de nome do arquivo. Textos brutos do contrato
    # geraram falsos positivos como cláusulas, perguntas e frases de vídeo.
    return marker_candidates


def _token_similarity(first: str, second: str) -> float:
    if first == second:
        return 1.0
    if len(first) < 3 or len(second) < 3:
        return 0.0
    score = SequenceMatcher(None, first, second).ratio()
    shorter, longer = sorted((first, second), key=len)
    if len(shorter) >= 5 and longer.startswith(shorter):
        score = max(score, 0.84)
    return score


def _fragment_token_similarity(expected: str, observed: str) -> float:
    """Compare cursive fragments while tolerating one clipped edge letter."""
    best = _token_similarity(expected, observed)
    if len(expected) >= 5 and len(observed) >= 3:
        variants = {
            expected[1:], expected[:-1],
            observed[1:], observed[:-1],
        }
        for first in (expected, expected[1:], expected[:-1]):
            for second in (observed, observed[1:], observed[:-1]):
                if len(first) >= 3 and len(second) >= 3:
                    best = max(best, SequenceMatcher(None, first, second).ratio())
    return best


@lru_cache(maxsize=512)
def _bridged_first_name_similarity(expected: str, observed: str) -> float:
    """Join complementary speech/OCR fragments through a plausible first name."""
    expected = normalize(expected)
    observed = normalize(observed)
    direct = _fragment_token_similarity(expected, observed)
    if len(expected) < 3 or len(observed) < 3:
        return direct
    best = direct
    for candidate in fuzzy_first_names():
        if abs(len(candidate) - len(expected)) > 2 or abs(len(candidate) - len(observed)) > 3:
            continue
        spoken_score = _token_similarity(expected, candidate)
        if spoken_score < 0.78:
            continue
        written_score = _fragment_token_similarity(candidate, observed)
        if written_score < 0.72:
            continue
        best = max(best, min(spoken_score, written_score))
    return best


def _candidate_match_score(
    label: str,
    candidate: str,
    token_counts: dict[str, int] | None = None,
    *,
    source_hint: bool = False,
) -> float:
    label_tokens = _name_tokens(_canonical_person_name(label))
    candidate_tokens = _name_tokens(_canonical_person_name(candidate))
    if not label_tokens or not candidate_tokens:
        return 0.0

    label_norm = " ".join(label_tokens)
    candidate_norm = " ".join(candidate_tokens)
    if label_norm == candidate_norm:
        return 1.0
    if len(label_tokens) == 1:
        token = label_tokens[0]
        first_similarity = _token_similarity(token, candidate_tokens[0]) if candidate_tokens else 0.0
        if token not in candidate_tokens:
            if first_similarity >= 0.78:
                return 0.70 if not (token_counts and token_counts.get(token, 0) > 1) else 0.54
            return 0.0
        ambiguous = bool(token_counts and token_counts.get(token, 0) > 1)
        if candidate_tokens and candidate_tokens[0] == token:
            return 0.58 if ambiguous else (0.84 if len(candidate_tokens) == 1 else 0.86)
        if first_similarity >= 0.86:
            return 0.66 if not ambiguous else 0.50
        return 0.46 if not ambiguous else 0.34
    if label_norm in candidate_norm or candidate_norm in label_norm:
        bonus = _surname_support_score(label_tokens) + _surname_support_score(candidate_tokens)
        return min(1.0, (0.94 if min(len(label_tokens), len(candidate_tokens)) >= 2 else 0.86) + bonus)

    token_scores = [
        max((_token_similarity(token, candidate_token) for candidate_token in candidate_tokens), default=0.0)
        for token in label_tokens
    ]
    reverse_scores = [
        max((_token_similarity(candidate_token, token) for token in label_tokens), default=0.0)
        for candidate_token in candidate_tokens
    ]
    exact_hits = sum(1 for token in label_tokens if token in candidate_tokens)
    fuzzy_hits = sum(1 for score in token_scores if score >= 0.84)
    coverage = sum(token_scores) / len(label_tokens)
    candidate_coverage = sum(reverse_scores) / len(candidate_tokens)
    whole_similarity = max(
        SequenceMatcher(None, label_norm, candidate_norm).ratio(),
        SequenceMatcher(None, " ".join(sorted(label_tokens)), " ".join(sorted(candidate_tokens))).ratio(),
    )

    first = label_tokens[0]
    unique_first = bool(token_counts and token_counts.get(first, 0) == 1)
    if unique_first and first in candidate_tokens:
        non_first_scores = [
            max((_token_similarity(token, candidate_token) for candidate_token in candidate_tokens[1:]), default=0.0)
            for token in label_tokens[1:]
        ]
        fuzzy_non_first_hits = sum(1 for score in non_first_scores if score >= 0.82)
        last_token_match = (
            len(label_tokens) >= 2
            and len(candidate_tokens) >= 2
            and _token_similarity(label_tokens[-1], candidate_tokens[-1]) >= 0.82
        )
        # Primeiro nome único ajuda, mas não pode atropelar sobrenome diferente.
        # Com sobrenomes dos dois lados, exigimos ao menos um sobrenome compatível
        # em nomes curtos, ou o último/mais de um token compatível em nomes longos.
        if len(label_tokens) >= 2 and len(candidate_tokens) >= 2:
            if exact_hits >= 2 or fuzzy_non_first_hits >= 2 or (fuzzy_non_first_hits >= 1 and (last_token_match or min(len(label_tokens), len(candidate_tokens)) <= 2)):
                return 0.88 if source_hint else 0.82
            # Primeiro nome exato sem sobrenome compatível é uma pista fraca.
            # A promoção por exclusão acontece depois, quando também sabemos
            # se esse primeiro nome é único entre os termos.
            if exact_hits == 1 and label_tokens[0] == candidate_tokens[0]:
                return 0.44 if source_hint else 0.40
            return 0.0
        if exact_hits >= 1 or candidate_coverage >= 0.62:
            return 0.88 if source_hint else 0.82
        return 0.76 if source_hint else 0.70
    if unique_first and token_scores and token_scores[0] >= 0.78:
        # Cursive OCR often changes the first letter (Higor -> Ligor) while
        # preserving the rest of the name. Accept this only for a first-name
        # position and with another supporting token, to avoid surname-only hits.
        non_first_scores = [
            max((_token_similarity(token, candidate_token) for candidate_token in candidate_tokens[1:]), default=0.0)
            for token in label_tokens[1:]
        ]
        fuzzy_non_first_hits = sum(1 for score in non_first_scores if score >= 0.82)
        if len(label_tokens) >= 2 and len(candidate_tokens) >= 2:
            if exact_hits >= 1 or fuzzy_non_first_hits >= 1:
                return 0.78 if source_hint else 0.72
            return 0.0
        if exact_hits >= 1 or fuzzy_hits >= 1:
            return 0.78 if source_hint else 0.72

    surname_bonus = max(_surname_support_score(label_tokens), _surname_support_score(candidate_tokens))
    if fuzzy_hits >= min(2, len(label_tokens)) or exact_hits >= 2:
        return min(1.0, 0.58 * coverage + 0.28 * candidate_coverage + 0.16 * whole_similarity + surname_bonus)
    if source_hint and exact_hits == 1 and whole_similarity >= 0.62:
        return min(0.76, 0.48 + whole_similarity * 0.20 + surname_bonus)
    return 0.0


def match_score(label: str, ocr_text: str, token_counts: dict[str, int] | None = None) -> float:
    label_tokens = _name_tokens(label)
    if not label_tokens:
        return 0.0
    candidates = extracted_name_candidates(ocr_text)
    if not candidates:
        return 0.0
    search_areas = candidates
    best = 0.0
    for area in search_areas:
        area_tokens = _name_tokens(area)
        if not area_tokens:
            continue
        direct_candidate_score = _candidate_match_score(label, area, token_counts)
        if direct_candidate_score:
            best = max(best, direct_candidate_score)
        area_norm = " ".join(area_tokens)
        label_norm = " ".join(label_tokens)
        exact_hits = sum(1 for token in label_tokens if token in area_tokens)
        token_scores = [
            max((_token_similarity(token, area_token) for area_token in area_tokens), default=0.0)
            for token in label_tokens
        ]
        fuzzy_hits = sum(1 for score in token_scores if score >= 0.84)
        soft_hits = sum(1 for score in token_scores if score >= 0.74)
        coverage = sum(token_scores) / len(label_tokens)
        if len(label_tokens) >= 2:
            first_anchor = token_scores[0] >= 0.78
            if fuzzy_hits < 2 and not first_anchor:
                # A single shared surname is not enough for an automatic match.
                best = max(best, min(0.42, coverage * 0.62))
                continue
        if len(label_tokens) == 1:
            token = label_tokens[0]
            if token in area_tokens:
                # Shortened photo names are accepted only as exact words in the
                # read name area. This avoids Amanda/Angela/Iara being guessed
                # when the handwritten name is clearly something else.
                ambiguous = bool(token_counts and token_counts.get(token, 0) > 1)
                if len(area_tokens) == 1:
                    position_bonus = 0.08 if area_tokens and token == area_tokens[0] else 0.0
                    best = max(best, 0.82 + position_bonus)
                else:
                    # If the term reads "Gabriela Silva" and the batch has both
                    # "Gabriela" and "Gabriela Silva Souza", keep the short name
                    # as a fallback so the more specific photo can win first.
                    best = max(best, 0.58 if ambiguous else 0.66)
            elif len(token) >= 5 and token_scores and token_scores[0] >= 0.88:
                best = max(best, 0.60)
            continue
        similarity = SequenceMatcher(None, label_norm, area_norm).ratio()
        sorted_similarity = SequenceMatcher(
            None,
            " ".join(sorted(label_tokens)),
            " ".join(sorted(area_tokens)),
        ).ratio()
        whole_similarity = max(similarity, sorted_similarity)
        rare_hits = []
        loose_rare_hits: list[tuple[str, str, float]] = []
        if token_counts:
            rare_hits = [
                token for token in label_tokens
                if len(token) >= 5 and token_counts.get(token, 0) == 1 and token in area_tokens
            ]
            loose_rare_hits = [
                (token, area_token, _token_similarity(token, area_token))
                for token in label_tokens
                for area_token in area_tokens
                if (
                    len(token) >= 6
                    and len(area_token) >= 6
                    and token_counts.get(token, 0) == 1
                    and token not in area_tokens
                    and _token_similarity(token, area_token) >= 0.62
                )
            ]
            loose_rare_hits.extend(
                (token, area_token, _token_similarity(token, area_token))
                for token in label_tokens
                for area_token in area_tokens
                if (
                    5 <= len(token) <= 6
                    and 4 <= len(area_token) <= 6
                    and token_counts.get(token, 0) == 1
                    and token not in area_tokens
                    and token[0] == area_token[0]
                    and token[-1] == area_token[-1]
                    and _token_similarity(token, area_token) >= 0.66
                )
            )
        score = 0.0
        whole_threshold = 0.82 if not exact_hits and fuzzy_hits < min(2, len(label_tokens)) else 0.72
        if fuzzy_hits >= min(2, len(label_tokens)) or (exact_hits and coverage >= 0.66) or whole_similarity >= whole_threshold:
            anchor_scores = [token_scores[0], token_scores[-1]] if len(token_scores) > 1 else token_scores
            anchor_bonus = 0.08 if max(anchor_scores, default=0.0) >= 0.84 else 0.0
            score = min(1.0, 0.64 * coverage + 0.30 * whole_similarity + anchor_bonus)
        elif soft_hits >= 2 and coverage >= 0.62:
            score = min(0.74, 0.66 * coverage + 0.22 * whole_similarity)
        if rare_hits:
            # If Vision only reads a distinctive surname, keep it as a reviewable
            # suggestion instead of dropping the term straight into Sem foto.
            score = max(score, 0.64 if len(area_tokens) <= 4 else 0.58)
        if loose_rare_hits:
            # Cursive OCR often distorts one or two letters in surnames
            # (Portilho -> Bortinhe). A unique, long token is still a useful
            # anchor, especially when the OCR only kept that fragment.
            best_loose = max(item[2] for item in loose_rare_hits)
            if len(area_tokens) == 1:
                score = max(score, min(0.58, 0.46 + best_loose * 0.14))
            elif best_loose >= 0.68:
                score = max(score, 0.50)
        if len(area_tokens) >= 2:
            covered_area_tokens = sum(
                1 for area_token in area_tokens
                if max((_token_similarity(area_token, label_token) for label_token in label_tokens), default=0.0) >= 0.88
            )
            if covered_area_tokens >= min(2, len(area_tokens)):
                score = max(score, min(0.90, 0.70 + covered_area_tokens * 0.04))
        if len(label_tokens) >= 2 and len(area_tokens) >= 2:
            label_non_first = label_tokens[1:]
            area_non_first = area_tokens[1:]
            surname_scores = [
                max((_token_similarity(label_token, area_token) for area_token in area_non_first), default=0.0)
                for label_token in label_non_first
            ]
            strong_surname_hits = sum(1 for item in surname_scores if item >= 0.82)
            first_close = token_scores[0] >= 0.78
            if first_close and strong_surname_hits >= 1:
                score = max(score, 0.76)
            elif first_close and any(item >= 0.72 for item in surname_scores) and (
                label_tokens[-1] in area_tokens or area_tokens[-1] in label_tokens
            ):
                score = max(score, 0.70)
        best = max(best, score)
    return best


def free_text_match_score(label: str, text: str, token_counts: dict[str, int] | None = None) -> float:
    label_tokens = _name_tokens(label)
    text_tokens = _name_tokens(text)
    if not label_tokens or not text_tokens:
        return 0.0
    token_scores = [
        max((_token_similarity(token, text_token) for text_token in text_tokens), default=0.0)
        for token in label_tokens
    ]
    fuzzy_hits = sum(1 for score in token_scores if score >= 0.84)
    coverage = sum(token_scores) / len(label_tokens)
    label_norm = " ".join(label_tokens)
    text_norm = " ".join(text_tokens)
    phrase_bonus = 0.10 if label_norm in text_norm else 0.0
    if len(label_tokens) >= 2 and fuzzy_hits < 2:
        return min(0.42, coverage * 0.62)
    if len(label_tokens) == 1:
        token = label_tokens[0]
        if token in text_tokens:
            ambiguous = bool(token_counts and token_counts.get(token, 0) > 1)
            return 0.58 if ambiguous else 0.66
        return 0.0
    return min(1.0, 0.72 * coverage + phrase_bonus)


def rough_transcript_roster_score(label: str, transcript: str, token_counts: dict[str, int] | None = None) -> float:
    label_tokens = _name_tokens(label)
    if not label_tokens:
        return 0.0
    fragments = rough_video_name_candidates(transcript)
    best = 0.0
    for fragment in fragments:
        fragment_tokens = _name_tokens(fragment)
        if not fragment_tokens:
            continue
        token_scores = [
            max((_token_similarity(token, fragment_token) for fragment_token in fragment_tokens), default=0.0)
            for token in label_tokens
        ]
        fuzzy_hits = sum(1 for score in token_scores if score >= 0.82)
        soft_hits = sum(1 for score in token_scores if score >= 0.68)
        exact_hits = sum(1 for token in label_tokens if token in fragment_tokens)
        coverage = sum(token_scores) / len(label_tokens)
        whole_similarity = max(
            SequenceMatcher(None, " ".join(label_tokens), " ".join(fragment_tokens)).ratio(),
            SequenceMatcher(None, " ".join(sorted(label_tokens)), " ".join(sorted(fragment_tokens))).ratio(),
        )
        score = 0.0
        if len(label_tokens) == 1:
            if exact_hits or (token_scores and token_scores[0] >= 0.84):
                score = 0.62
        elif fuzzy_hits >= 2:
            score = min(0.82, 0.54 + 0.20 * coverage + 0.12 * whole_similarity)
        elif exact_hits >= 1 and soft_hits >= 2:
            score = min(0.74, 0.50 + 0.20 * coverage + 0.08 * whole_similarity)
        elif whole_similarity >= 0.66 and soft_hits >= min(2, len(label_tokens)):
            score = min(0.68, 0.42 + 0.22 * whole_similarity + 0.06 * coverage)
        if token_counts and score:
            rare_soft_hits = sum(
                1 for token, token_score in zip(label_tokens, token_scores)
                if len(token) >= 5 and token_counts.get(token, 0) == 1 and token_score >= 0.68
            )
            if rare_soft_hits:
                score = min(0.86, score + min(0.10, rare_soft_hits * 0.05))
        best = max(best, score)
    return best


def _caption_transcript(person: PersonPhoto) -> str:
    for part in person.caption.split(" · "):
        if part.startswith("Transcrição:"):
            return part.split(":", 1)[1].strip()
    return ""


def _term_roster_names(results: list[TermResult]) -> list[str]:
    names: list[str] = []
    for result in results:
        source_name = source_name_candidate(result.source)
        if source_name and _valid_name_candidate(source_name):
            names.append(source_name)
            continue
        for name in extracted_name_candidates(result.ocr_text):
            if _valid_name_candidate(name):
                names.append(name.title())
        if _valid_name_candidate(result.output_name) and source_name_candidate(Path(result.output_name or "")):
            names.append(result.output_name)
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = normalize(name)
        if key and key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def relabel_unidentified_people_from_terms(people: list[PersonPhoto], results: list[TermResult]) -> int:
    roster = _term_roster_names(results)
    if not roster:
        return 0
    pseudo_people = [PersonPhoto(label=name, caption="", image_path=Path(name)) for name in roster]
    token_counts = _person_token_counts(pseudo_people)
    changed = 0
    for person in people:
        if person_is_identified(person):
            continue
        transcript = _caption_transcript(person)
        if not transcript:
            continue
        scored: list[tuple[float, str]] = []
        normalized_transcript = normalize(transcript)
        for name in roster:
            score = free_text_match_score(name, transcript, token_counts)
            score = max(score, rough_transcript_roster_score(name, transcript, token_counts))
            if normalize(name) in normalized_transcript:
                score = max(score, 0.96)
            scored.append((score, name))
        scored.sort(reverse=True)
        if not scored:
            continue
        best_score, best_name = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.70 or (best_score >= 0.58 and best_score - second_score >= 0.16):
            person.label = best_name
            person.caption = f"{person.caption} · Nome recuperado pelos termos: {best_name}"
            changed += 1
    return changed


def enhance_weak_terms_with_roster(results: list[TermResult], people: list[PersonPhoto]) -> int:
    roster = ocr_custom_names(people)
    if not roster or MAX_ROSTER_OCR_RECHECKS <= 0:
        return 0
    candidates = [
        result for result in results
        if not source_name_candidate(result.source) and not extract_name_candidate(result.ocr_text)
    ][:MAX_ROSTER_OCR_RECHECKS]
    distinct_people = _distinct_matching_people(people)
    readings: list[tuple[TermResult, str]] = []
    for result in candidates:
        decorated = focused_name_ocr(result.page, roster, max_regions=1)
        raw_before_marker, _raw_candidates = _marker_sections(decorated)
        raw_guided = guided_name_raw_ocr(result.page, roster)
        evidence = raw_before_marker.strip()
        if raw_guided:
            evidence = f"{evidence}\nLEITURA_GUIADA_BRUTA\n{raw_guided}".strip()
        if evidence:
            result.ocr_text = f"LEITURA_FOCADA_DO_NOME_GUIADA\n{evidence}"
            readings.append((result, evidence))

    if not readings or not distinct_people:
        return 0

    score_maps = [
        {normalize(name): score for score, name in _person_fragment_support_scores(evidence, distinct_people)}
        for _result, evidence in readings
    ]
    scores: list[list[float]] = []
    weights: list[list[float]] = []
    for person in distinct_people:
        person_key = normalize(person.label)
        row = [score_map.get(person_key, 0.0) for score_map in score_maps]
        scores.append(row)
        weights.append([score if score >= 0.70 else 0.0 for score in row] + [0.0] * len(distinct_people))

    recovered = 0
    for person_index, reading_index in enumerate(_hungarian_maximize(weights)):
        if reading_index < 0 or reading_index >= len(readings):
            continue
        score = scores[person_index][reading_index]
        row_competitor = max(
            (value for index, value in enumerate(scores[person_index]) if index != reading_index),
            default=0.0,
        )
        column_competitor = max(
            (
                scores[other_person_index][reading_index]
                for other_person_index in range(len(distinct_people))
                if other_person_index != person_index
            ),
            default=0.0,
        )
        if score < 0.70 or (score < 0.84 and score - max(row_competitor, column_competitor) < 0.08):
            continue
        result, evidence = readings[reading_index]
        guided_name = distinct_people[person_index].label
        result.ocr_text = f"LEITURA_FOCADA_DO_NOME_GUIADA\n{evidence}\nNOME_CANDIDATO\n{guided_name}"
        result.output_name = guided_name
        recovered += 1
    return recovered


def people_without_assigned_term(results: list[TermResult], people: list[PersonPhoto]) -> list[PersonPhoto]:
    assigned_keys = {
        _matching_person_key(result.person)
        for result in results
        if result.person is not None and _matching_person_key(result.person)
    }
    return [
        person for person in matchable_people(people)
        if _matching_person_key(person) not in assigned_keys
    ]


def _person_fragment_support_scores(raw_text: str, people: list[PersonPhoto]) -> list[tuple[float, str]]:
    noise = (
        HARD_NON_NAME_WORDS
        | VIDEO_SPOKEN_NON_NAME_WORDS
        | FORM_INTRO_NOISE_WORDS
        | VIDEO_FILLER_WORDS
        | NAME_BLOCKLIST
        | {"leitura", "focada", "guiada", "bruta", "candidato"}
    )
    observed = [
        token for token in normalize(raw_text).split()
        if len(token) >= 3 and token not in noise and not token.isdigit()
    ]
    if not observed:
        return []
    distinct_people = _distinct_matching_people(people)
    firsts = [_first_name_token(person.label) for person in distinct_people]
    ranked: list[tuple[float, str]] = []
    for person in distinct_people:
        tokens = _name_tokens(person.label)
        if not tokens:
            continue
        direct_scores = [
            max((_fragment_token_similarity(token, fragment) for fragment in observed), default=0.0)
            for token in tokens
        ]
        direct_first = direct_scores[0]
        bridged_first = max(
            direct_first,
            max((_bridged_first_name_similarity(tokens[0], fragment) for fragment in observed), default=0.0),
        )
        strong_non_first = sum(score >= 0.82 for score in direct_scores[1:])
        soft_non_first = sum(score >= 0.68 for score in direct_scores[1:])
        strong = strong_non_first + int(direct_first >= 0.82)
        unique_first = _fuzzy_unique_in_pool(tokens[0], firsts, threshold=0.82)
        score = 0.0
        if strong >= 2 and direct_first >= 0.68:
            score = min(0.94, 0.68 + 0.08 * strong + 0.08 * direct_first)
        elif unique_first and direct_first >= 0.84:
            # Um primeiro nome claramente lido e único já é evidência real em
            # lotes pequenos; a exclusão apenas confirma, não inventa a leitura.
            score = min(0.84, 0.70 + 0.14 * direct_first + 0.02 * soft_non_first)
        elif unique_first and bridged_first >= 0.64 and strong_non_first >= 1 and soft_non_first >= 1:
            # O Vision às vezes perde uma letra em cada ponta do primeiro nome
            # ("Higor" vira "Gor"), mas preserva um sobrenome longo e único.
            # A exclusão entre si torna essa combinação segura em lotes pequenos.
            score = min(0.82, 0.60 + 0.10 * strong_non_first + 0.06 * soft_non_first + 0.06 * bridged_first)
        elif unique_first and bridged_first >= 0.82 and direct_first >= 0.52:
            # Algumas canetas atravessam o título ou a margem do recorte e o
            # Vision conserva só o miolo do primeiro nome ("Higo" -> "gor").
            # Exigimos simultaneamente um fragmento direto reconhecível, uma
            # reconstrução forte pelo banco brasileiro e unicidade no lote.
            # Assim "gor" recupera o único Higo, mas "Avisio" não vira Elio.
            score = min(0.79, 0.66 + 0.10 * bridged_first + 0.05 * direct_first)
        ranked.append((score, person.label))
    ranked.sort(reverse=True)
    return ranked


def _unique_person_supported_by_name_fragments(raw_text: str, people: list[PersonPhoto]) -> str:
    ranked = _person_fragment_support_scores(raw_text, people)
    if not ranked:
        return ""
    best_score, best_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < 0.70 or best_score - second_score < 0.10:
        return ""
    return best_name


def extract_name_candidate(ocr_text: str) -> str:
    candidates = extracted_name_candidates(ocr_text)
    return candidates[0].title() if candidates else ""


def _name_extends_or_matches(shorter_name: str, longer_name: str, token_counts: dict[str, int] | None = None) -> bool:
    shorter = _name_tokens(_canonical_person_name(shorter_name))
    longer = _name_tokens(_canonical_person_name(longer_name))
    if not shorter or not longer or len(longer) < len(shorter):
        return False
    if " ".join(shorter) in " ".join(longer):
        return True
    matched = [
        max((_token_similarity(token, other) for other in longer), default=0.0) >= 0.88
        for token in shorter
    ]
    hits = sum(1 for item in matched if item)
    if len(shorter) == 1:
        return hits == 1
    if len(shorter) == 2:
        return hits == 2
    return hits >= max(2, len(shorter) - 1) and (matched[0] or matched[-1])


def _names_conflict(first_name: str, second_name: str, token_counts: dict[str, int] | None = None) -> bool:
    first = _name_tokens(_canonical_person_name(first_name))
    second = _name_tokens(_canonical_person_name(second_name))
    if len(first) < 2 or len(second) < 2:
        return False
    if _name_extends_or_matches(first_name, second_name, token_counts) or _name_extends_or_matches(second_name, first_name, token_counts):
        return False
    first_set = set(first)
    second_set = set(second)
    exact_shared = first_set & second_set
    if len(exact_shared) >= 2:
        return False
    if first[0] == second[0] and exact_shared:
        return False
    return True


def _safe_supported_name_extension(base_name: str, candidate_name: str) -> bool:
    base_tokens = _name_tokens(_canonical_person_name(base_name))
    candidate_tokens = _name_tokens(_canonical_person_name(candidate_name))
    if not base_tokens or len(candidate_tokens) <= len(base_tokens):
        return False
    supported = common_surnames() | known_first_names() | {
        normalize(value) for value in OCR_NAME_TOKEN_CORRECTIONS.values()
    }
    if not all(token in supported for token in candidate_tokens):
        return False
    candidate_index = 0
    for base_token in base_tokens:
        while (
            candidate_index < len(candidate_tokens)
            and _token_similarity(base_token, candidate_tokens[candidate_index]) < 0.86
        ):
            candidate_index += 1
        if candidate_index >= len(candidate_tokens):
            return False
        candidate_index += 1
    return _token_similarity(base_tokens[0], candidate_tokens[0]) >= 0.86


def _first_names_compatible(first_name: str, second_name: str, threshold: float = 0.84) -> bool:
    first = _first_name_token(_canonical_person_name(first_name))
    second = _first_name_token(_canonical_person_name(second_name))
    return bool(first and second and _token_similarity(first, second) >= threshold)


def _fuzzy_unique_in_pool(token: str, pool: list[str], threshold: float = 0.84) -> bool:
    if not token:
        return False
    return sum(1 for item in pool if _token_similarity(token, item) >= threshold) == 1


def _best_non_first_match(first_name: str, second_name: str) -> float:
    first_tokens = _name_tokens(_canonical_person_name(first_name))[1:]
    second_tokens = _name_tokens(_canonical_person_name(second_name))[1:]
    if not first_tokens or not second_tokens:
        return 0.0
    return max(
        (
            _token_similarity(first_token, second_token)
            for first_token in first_tokens
            for second_token in second_tokens
        ),
        default=0.0,
    )


def _exclusive_first_name_evidence_score(
    result: TermResult,
    person: PersonPhoto,
    term_firsts: list[str],
    person_firsts: list[str],
    token_counts: dict[str, int] | None = None,
) -> float:
    person_first = _first_name_token(person.label)
    if not person_first or not _fuzzy_unique_in_pool(person_first, person_firsts):
        return 0.0

    evidence_names = [
        name for name in (extract_name_candidate(result.ocr_text), source_name_candidate(result.source))
        if name and _valid_name_candidate(name)
    ]
    best = 0.0
    for name in evidence_names:
        term_first = _first_name_token(name)
        if not term_first or not _fuzzy_unique_in_pool(term_first, term_firsts):
            continue
        first_score = _token_similarity(term_first, person_first)
        if first_score < 0.84:
            continue

        direct = _candidate_match_score(person.label, name, token_counts, source_hint=True)
        non_first_score = _best_non_first_match(person.label, name)
        transcript_only_person = (
            person.caption.startswith("Vídeo:")
            and "Transcrição:" in person.caption
            and "Nome usado do arquivo" not in person.caption
            and "Nome falado confirmou" not in person.caption
            and "Nome falado associado pela lista do lote" not in person.caption
            and "Nome falado recuperado na confirmação" not in person.caption
            and "Nome recuperado pelos termos" not in person.caption
        )
        if transcript_only_person and non_first_score < 0.82 and direct < MIN_REVIEW_MATCH:
            continue
        score = 0.76 if first_score < 0.92 else 0.82
        if non_first_score >= 0.88:
            score = 0.90
        elif non_first_score >= 0.82:
            score = 0.86
        if direct:
            score = max(score, direct)
        best = max(best, min(0.92, score))
    return best


def _preferred_result_evidence_name(result: TermResult) -> str:
    source_name = source_name_candidate(result.source)
    if source_name and _valid_name_candidate(source_name):
        return source_name
    ocr_name = extract_name_candidate(result.ocr_text)
    if ocr_name and _valid_name_candidate(ocr_name):
        return ocr_name
    return ""


def _exclusive_surname_evidence_score(
    result: TermResult,
    person: PersonPhoto,
    term_surnames: list[str],
    person_surnames: list[str],
) -> float:
    evidence_name = _preferred_result_evidence_name(result)
    term_tokens = _name_tokens(evidence_name)
    person_tokens = _name_tokens(person.label)
    if len(term_tokens) < 2 or len(person_tokens) < 2:
        return 0.0

    term_surname = term_tokens[-1]
    person_surname = person_tokens[-1]
    if _token_similarity(term_surname, person_surname) < 0.94:
        return 0.0
    if not _fuzzy_unique_in_pool(term_surname, term_surnames, threshold=0.92):
        return 0.0
    if not _fuzzy_unique_in_pool(person_surname, person_surnames, threshold=0.92):
        return 0.0

    first_score = _token_similarity(term_tokens[0], person_tokens[0])
    if first_score < 0.62:
        return 0.0
    if first_score >= 0.84:
        return 0.84
    # Um sobrenome raro e exclusivo permite resolver pequenos erros no primeiro
    # nome (por exemplo, Rangel/Rafael), mas não uma pessoa sem relação nominal.
    return 0.72


@lru_cache(maxsize=2048)
def _orthographic_first_name_key(token: str) -> str:
    token = normalize(token)
    if token.startswith("w"):
        token = "v" + token[1:]
    return token.replace("y", "i")


def _first_name_variant_similarity(first: str, second: str) -> float:
    direct = _token_similarity(first, second)
    first_key = _orthographic_first_name_key(first)
    second_key = _orthographic_first_name_key(second)
    if not first_key or not second_key:
        return direct
    return max(direct, SequenceMatcher(None, first_key, second_key).ratio())


def _tail_name_similarity(first_tokens: list[str], second_tokens: list[str]) -> float:
    if len(first_tokens) < 2 or len(second_tokens) < 2:
        return 0.0
    return max(
        (
            _token_similarity(first_token, second_token)
            for first_token in first_tokens[1:]
            for second_token in second_tokens[1:]
        ),
        default=0.0,
    )


def _exclusive_compound_name_evidence_score(
    result: TermResult,
    person: PersonPhoto,
    results: list[TermResult],
    people: list[PersonPhoto],
) -> float:
    """Resolve a spelling variant only when first name and surname agree globally."""
    term_tokens = _name_tokens(_preferred_result_evidence_name(result))
    person_tokens = _name_tokens(person.label)
    if len(term_tokens) < 2 or len(person_tokens) < 2:
        return 0.0

    direct_first = _token_similarity(term_tokens[0], person_tokens[0])
    variant_first = _first_name_variant_similarity(term_tokens[0], person_tokens[0])
    tail_score = _tail_name_similarity(term_tokens, person_tokens)
    if direct_first < 0.70 or variant_first < 0.94 or tail_score < 0.94:
        return 0.0

    compatible_people = []
    for other_person in people:
        other_tokens = _name_tokens(other_person.label)
        if other_tokens and _first_name_variant_similarity(term_tokens[0], other_tokens[0]) >= 0.94:
            compatible_people.append(other_person)
    if len(compatible_people) != 1 or _matching_person_key(compatible_people[0]) != _matching_person_key(person):
        return 0.0

    surname_supported_results = []
    for other_result in results:
        other_tokens = _name_tokens(_preferred_result_evidence_name(other_result))
        if not other_tokens or _first_name_variant_similarity(person_tokens[0], other_tokens[0]) < 0.94:
            continue
        if _tail_name_similarity(person_tokens, other_tokens) >= 0.94:
            surname_supported_results.append(other_result)
    if len(surname_supported_results) != 1 or surname_supported_results[0] is not result:
        return 0.0

    return 0.90 if direct_first >= 0.76 else 0.86


def result_output_name(
    result: TermResult,
    person: PersonPhoto | None = None,
    people: list[PersonPhoto] | None = None,
    token_counts: dict[str, int] | None = None,
) -> str:
    source_name = source_name_candidate(result.source)
    ocr_name = extract_name_candidate(result.ocr_text)
    source_tokens = _name_tokens(source_name)
    ocr_tokens = _name_tokens(_canonical_person_name(ocr_name))
    source_ocr_compatible = (
        _name_extends_or_matches(source_name, ocr_name, token_counts)
        or (
            _first_names_compatible(source_name, ocr_name, threshold=0.86)
            and bool(source_tokens)
            and any(_token_similarity(source_tokens[-1], token) >= 0.86 for token in ocr_tokens)
        )
    )
    if (
        source_name
        and ocr_name
        and source_ocr_compatible
        and len(ocr_tokens) > len(source_tokens)
        and _safe_supported_name_extension(source_name, ocr_name)
        and not any(token in HARD_NON_NAME_WORDS or token in FORM_INTRO_NOISE_WORDS for token in normalize(ocr_name).split())
    ):
        return ocr_name.title()
    if person:
        source_score = _candidate_match_score(person.label, source_name, token_counts, source_hint=True) if source_name else 0.0
        ocr_score = _candidate_match_score(person.label, ocr_name, token_counts) if ocr_name else 0.0
        if (
            ocr_name
            and len(_name_tokens(ocr_name)) > max(len(source_tokens), len(_name_tokens(person.label)))
            and _first_names_compatible(ocr_name, person.label)
            and _best_non_first_match(ocr_name, person.label) >= 0.82
            and (not source_name or _safe_supported_name_extension(source_name, ocr_name))
            and not any(token in HARD_NON_NAME_WORDS or token in FORM_INTRO_NOISE_WORDS for token in normalize(ocr_name).split())
            and (not source_name or not _names_conflict(source_name, ocr_name, token_counts))
        ):
            return ocr_name.title()
        if (
            ocr_name
            and ocr_score >= 0.82
            and len(_name_tokens(person.label)) >= 2
            and len(_name_tokens(ocr_name)) >= max(2, len(_name_tokens(person.label)))
            and (not source_name or len(ocr_tokens) <= len(source_tokens) or _safe_supported_name_extension(source_name, ocr_name))
            and (not source_name or ocr_score >= source_score + 0.08 or len(_name_tokens(ocr_name)) > len(_name_tokens(source_name)))
            and (not source_name or not _names_conflict(source_name, ocr_name, token_counts))
        ):
            return ocr_name.title()
        if (
            source_name
            and (
                source_score >= MIN_AUTOMATIC_MATCH
                or (len(source_tokens) >= 2 and _first_names_compatible(source_name, person.label))
            )
        ):
            return source_name
        if (
            ocr_name
            and ocr_score >= 0.82
            and len(_name_tokens(person.label)) >= 2
            and len(_name_tokens(ocr_name)) >= len(_name_tokens(person.label))
            and (not source_name or not _names_conflict(source_name, ocr_name, token_counts))
        ):
            return ocr_name.title()
        return person.label
    if len(source_tokens) >= 2:
        return source_name
    if source_name:
        return source_name
    return ocr_name.title() if ocr_name else result.source.stem


def _confidence_from_scores(best_score: float, second_score: float) -> float:
    gap = max(0.0, best_score - second_score)
    if best_score >= MIN_AUTOMATIC_MATCH:
        if second_score >= MIN_REVIEW_MATCH:
            if gap < 0.03:
                return min(0.49, best_score)
            if gap < 0.08:
                return min(best_score, 0.50 + gap)
        return min(1.0, best_score)
    return min(1.0, best_score * min(1.0, 0.78 + gap))


def rank_people_for_ocr(
    people: list[PersonPhoto],
    ocr_text: str,
    token_counts: dict[str, int] | None = None,
) -> list[tuple[float, PersonPhoto]]:
    people = matchable_people(people)
    return sorted(
        ((match_score(person.label, ocr_text, token_counts), person) for person in people),
        key=lambda item: item[0],
        reverse=True,
    )


def result_person_score(result: TermResult, person: PersonPhoto, token_counts: dict[str, int] | None = None) -> float:
    ocr_score = match_score(person.label, result.ocr_text, token_counts)
    source_name = source_name_candidate(result.source)
    source_score = _candidate_match_score(person.label, source_name, token_counts, source_hint=True) if source_name else 0.0
    ocr_name = extract_name_candidate(result.ocr_text)
    ocr_name_score = _candidate_match_score(person.label, ocr_name, token_counts) if ocr_name else 0.0
    ocr_supports_person = bool(ocr_name_score >= MIN_AUTOMATIC_MATCH)
    if (
        source_name
        and _source_name_blocks_person(source_name, person, token_counts)
        and not (
            ocr_name
            and ocr_name_score >= MIN_AUTOMATIC_MATCH
            and (
                _name_extends_or_matches(source_name, ocr_name, token_counts)
                or _name_extends_or_matches(person.label, ocr_name, token_counts)
            )
        )
    ):
        return min(max(ocr_score, source_score), MIN_REVIEW_MATCH - 0.01)
    if (
        source_name
        and ocr_name
        and _names_conflict(source_name, ocr_name, token_counts)
        and _source_name_blocks_person(source_name, person, token_counts)
    ):
        return min(max(ocr_score, source_score), MIN_REVIEW_MATCH - 0.01)
    if source_name and not ocr_supports_person and _source_name_blocks_person(source_name, person, token_counts):
        return min(max(ocr_score, source_score), MIN_REVIEW_MATCH - 0.01)
    if ocr_name and ocr_name_score:
        ocr_tokens = _name_tokens(ocr_name)
        person_tokens = _name_tokens(person.label)
        distinctive_hits = sum(
            1 for token in person_tokens
            if len(token) >= 5 and max((_token_similarity(token, other) for other in ocr_tokens), default=0.0) >= 0.88
        )
        specificity_bonus = min(0.08, distinctive_hits * 0.04 + max(0, len(person_tokens) - 2) * 0.01)
        ocr_score = max(ocr_score, min(1.0, ocr_name_score + specificity_bonus))
    if source_score >= MIN_AUTOMATIC_MATCH and ocr_score >= MIN_REVIEW_MATCH:
        if ocr_name and len(_name_tokens(ocr_name)) > len(_name_tokens(source_name)):
            return max(ocr_score, min(0.92, source_score))
        return max(source_score, min(1.0, ocr_score + 0.06))
    return max(ocr_score, source_score)


def rank_people_for_result(
    people: list[PersonPhoto],
    result: TermResult,
    token_counts: dict[str, int] | None = None,
) -> list[tuple[float, PersonPhoto]]:
    people = matchable_people(people)
    return sorted(
        ((result_person_score(result, person, token_counts), person) for person in people),
        key=lambda item: item[0],
        reverse=True,
    )


def suggest_person(people: list[PersonPhoto], ocr_text: str) -> tuple[PersonPhoto | None, float]:
    people = matchable_people(people)
    if not people or not ocr_text:
        return None, 0.0
    token_counts = _person_token_counts(people)
    ranked = rank_people_for_ocr(people, ocr_text, token_counts)
    best_score, best = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    confidence = _confidence_from_scores(best_score, second)
    if best_score < MIN_REVIEW_MATCH:
        return None, confidence
    return best, confidence


def suggest_person_for_result(people: list[PersonPhoto], result: TermResult) -> tuple[PersonPhoto | None, float]:
    people = matchable_people(people)
    if not people:
        return None, 0.0
    token_counts = _person_token_counts(people)
    ranked = rank_people_for_result(people, result, token_counts)
    best_score, best = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    confidence = _confidence_from_scores(best_score, second)
    if best_score < MIN_REVIEW_MATCH:
        return None, confidence
    return best, confidence


def _source_name_blocks_person(source_name: str, person: PersonPhoto, token_counts: dict[str, int] | None = None) -> bool:
    source_tokens = _name_tokens(source_name)
    if len(source_tokens) < 2:
        return False
    if _name_extends_or_matches(person.label, source_name, token_counts) or _name_extends_or_matches(source_name, person.label, token_counts):
        return False
    person_tokens = _name_tokens(person.label)
    if len(person_tokens) >= 2:
        exact_shared = set(source_tokens) & set(person_tokens)
        first_match = _token_similarity(source_tokens[0], person_tokens[0]) >= 0.86
        if not first_match:
            return True
        if len(exact_shared) < 2 and not (
            source_tokens[-1] in person_tokens
            and source_tokens[0] in person_tokens
        ):
            return True
    score = _candidate_match_score(person.label, source_name, token_counts, source_hint=True)
    return score < MIN_REVIEW_MATCH


def _ocr_specificity_priority(result: TermResult, person: PersonPhoto) -> float:
    ocr_name = extract_name_candidate(result.ocr_text)
    ocr_tokens = _name_tokens(ocr_name)
    person_tokens = _name_tokens(person.label)
    if not ocr_tokens or not person_tokens:
        return 0.0
    matched = 0
    distinctive = 0
    for token in person_tokens:
        if max((_token_similarity(token, other) for other in ocr_tokens), default=0.0) >= 0.88:
            matched += 1
            if len(token) >= 5:
                distinctive += 1
    last_name_anchor = 0.0
    if len(ocr_tokens) >= 2 and max((_token_similarity(token, ocr_tokens[-1]) for token in person_tokens), default=0.0) >= 0.88:
        last_name_anchor = 0.18
    return matched * 0.03 + distinctive * 0.05 + last_name_anchor + min(0.03, max(0, len(person_tokens) - 2) * 0.01)


def _matching_person_key(person: PersonPhoto) -> str:
    return " ".join(_name_tokens(_canonical_person_name(person.label)))


def _person_evidence_priority(person: PersonPhoto) -> tuple[int, int, str]:
    caption = person.caption
    priority = 0
    if "Nome usado do arquivo" in caption:
        priority = 5
    elif "Nome falado recuperado na confirmação" in caption:
        priority = 4
    elif "Nome falado associado pela lista do lote" in caption:
        priority = 3
    elif "Nome recuperado pelos termos" in caption:
        priority = 2
    elif "Transcrição:" in caption:
        priority = 1
    return (priority, len(_name_tokens(person.label)), normalize(person.label))


def _distinct_matching_people(people: list[PersonPhoto]) -> list[PersonPhoto]:
    grouped: dict[str, PersonPhoto] = {}
    for person in matchable_people(people):
        key = _matching_person_key(person)
        if not key:
            continue
        current = grouped.get(key)
        if current is None or _person_evidence_priority(person) > _person_evidence_priority(current):
            grouped[key] = person
    return sorted(grouped.values(), key=lambda person: alpha_key(person.label))


def _hungarian_maximize(weights: list[list[float]]) -> list[int]:
    """Return the selected column for each row in a rectangular weight matrix."""
    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if column_count < row_count or any(len(row) != column_count for row in weights):
        raise ValueError("A matriz de associação precisa ter colunas suficientes e tamanho regular.")

    max_weight = max((max(row, default=0.0) for row in weights), default=0.0)
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    selected_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row_index in range(1, row_count + 1):
        selected_row[0] = row_index
        column = 0
        min_cost = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            active_row = selected_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                cost = max_weight - weights[active_row - 1][candidate_column - 1]
                reduced = cost - u[active_row] - v[candidate_column]
                if reduced < min_cost[candidate_column]:
                    min_cost[candidate_column] = reduced
                    previous_column[candidate_column] = column
                if min_cost[candidate_column] < delta:
                    delta = min_cost[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    u[selected_row[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    min_cost[candidate_column] -= delta
            column = next_column
            if selected_row[column] == 0:
                break
        while True:
            previous = previous_column[column]
            selected_row[column] = selected_row[previous]
            column = previous
            if column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if selected_row[column]:
            assignment[selected_row[column] - 1] = column - 1
    return assignment


def _assignment_score_context(
    results: list[TermResult],
    people: list[PersonPhoto],
) -> tuple[
    list[PersonPhoto],
    dict[str, int],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    distinct_people = _distinct_matching_people(people)
    token_counts = _person_token_counts(distinct_people) if distinct_people else {}
    term_firsts: list[str] = []
    term_surnames: list[str] = []
    for result in results:
        evidence_tokens = _name_tokens(_preferred_result_evidence_name(result))
        if evidence_tokens:
            term_firsts.append(evidence_tokens[0])
        if len(evidence_tokens) >= 2:
            term_surnames.append(evidence_tokens[-1])

    person_firsts: list[str] = []
    person_surnames: list[str] = []
    for person in distinct_people:
        person_tokens = _name_tokens(person.label)
        if person_tokens:
            person_firsts.append(person_tokens[0])
        if len(person_tokens) >= 2:
            person_surnames.append(person_tokens[-1])
    return (
        distinct_people,
        token_counts,
        term_firsts,
        term_surnames,
        person_firsts,
        person_surnames,
    )


def _assignment_pair_scores(
    result: TermResult,
    person: PersonPhoto,
    token_counts: dict[str, int],
    term_firsts: list[str],
    term_surnames: list[str],
    person_firsts: list[str],
    person_surnames: list[str],
    results: list[TermResult],
    people: list[PersonPhoto],
) -> tuple[float, float, float, float, float]:
    direct = result_person_score(result, person, token_counts)
    first_exclusion = _exclusive_first_name_evidence_score(
        result,
        person,
        term_firsts,
        person_firsts,
        token_counts,
    )
    surname_exclusion = _exclusive_surname_evidence_score(
        result,
        person,
        term_surnames,
        person_surnames,
    )
    compound_exclusion = _exclusive_compound_name_evidence_score(
        result,
        person,
        results,
        people,
    )
    after_first = max(direct, first_exclusion)
    final = min(1.0, max(after_first, surname_exclusion, compound_exclusion))
    return direct, after_first, surname_exclusion, compound_exclusion, final


def _assignment_stage_summary(
    stage: str,
    scores: list[list[float]],
    results: list[TermResult],
    people: list[PersonPhoto],
) -> dict:
    weights: list[list[float]] = []
    for person, row in zip(people, scores):
        real_weights = []
        for result, score in zip(results, row):
            if score >= MIN_REVIEW_MATCH:
                tie_break = min(0.0009, _ocr_specificity_priority(result, person) * 0.001)
                real_weights.append(score + tie_break)
            else:
                real_weights.append(0.0)
        weights.append(real_weights + [0.0] * len(people))

    selected_rows: list[dict] = []
    ok = 0
    review = 0
    for person_index, result_index in enumerate(_hungarian_maximize(weights)):
        if result_index < 0 or result_index >= len(results):
            continue
        score = scores[person_index][result_index]
        if score < MIN_REVIEW_MATCH:
            continue
        row_competitor = max(
            (value for index, value in enumerate(scores[person_index]) if index != result_index),
            default=0.0,
        )
        column_competitor = max(
            (
                scores[other_person_index][result_index]
                for other_person_index in range(len(people))
                if other_person_index != person_index
            ),
            default=0.0,
        )
        competitor = max(row_competitor, column_competitor)
        confidence = _confidence_from_scores(score, competitor)
        if confidence >= OK_CONFIDENCE_THRESHOLD:
            ok += 1
        else:
            review += 1
        selected_rows.append({
            "person": people[person_index].label,
            "term": results[result_index].source.name,
            "term_name": _preferred_result_evidence_name(results[result_index]),
            "score": round(score, 4),
            "competitor": round(competitor, 4),
            "confidence": round(confidence, 4),
        })
    return {
        "stage": stage,
        "ok": ok,
        "review": review,
        "unmatched": max(0, len(people) - ok - review),
        "selected": selected_rows,
    }


def assignment_score_audit(results: list[TermResult], people: list[PersonPhoto]) -> dict:
    """Explain every matching stage without changing the selected results."""
    (
        distinct_people,
        token_counts,
        term_firsts,
        term_surnames,
        person_firsts,
        person_surnames,
    ) = _assignment_score_context(results, people)
    if not distinct_people:
        return {"stages": [], "people": []}

    direct_matrix: list[list[float]] = []
    first_matrix: list[list[float]] = []
    final_matrix: list[list[float]] = []
    component_rows: list[list[dict]] = []
    for person in distinct_people:
        direct_row: list[float] = []
        first_row: list[float] = []
        final_row: list[float] = []
        components: list[dict] = []
        for result in results:
            direct, after_first, surname, compound, final = _assignment_pair_scores(
                result,
                person,
                token_counts,
                term_firsts,
                term_surnames,
                person_firsts,
                person_surnames,
                results,
                distinct_people,
            )
            direct_row.append(direct)
            first_row.append(after_first)
            final_row.append(final)
            components.append({
                "term": result.source.name,
                "term_name": _preferred_result_evidence_name(result),
                "direct": round(direct, 4),
                "after_first_name_exclusion": round(after_first, 4),
                "surname_exclusion": round(surname, 4),
                "compound_exclusion": round(compound, 4),
                "final": round(final, 4),
            })
        direct_matrix.append(direct_row)
        first_matrix.append(first_row)
        final_matrix.append(final_row)
        component_rows.append(components)

    assigned_by_person = {
        _matching_person_key(result.person): result
        for result in results
        if result.person is not None and _matching_person_key(result.person)
    }
    people_rows: list[dict] = []
    for person, components in zip(distinct_people, component_rows):
        ranked = sorted(components, key=lambda item: item["final"], reverse=True)
        assigned = assigned_by_person.get(_matching_person_key(person))
        people_rows.append({
            "person": person.label,
            "source_file": person.image_path.name,
            "assigned_term": assigned.source.name if assigned else "",
            "assigned_output_name": assigned.output_name if assigned else "",
            "assigned_confidence": round(assigned.confidence, 4) if assigned else 0.0,
            "top_candidates": ranked[:5],
        })

    return {
        "stages": [
            _assignment_stage_summary("1_direct", direct_matrix, results, distinct_people),
            _assignment_stage_summary("2_first_name_exclusion", first_matrix, results, distinct_people),
            _assignment_stage_summary("3_surname_and_global_exclusion", final_matrix, results, distinct_people),
        ],
        "people": people_rows,
    }


def assign_people_to_results(results: list[TermResult], people: list[PersonPhoto]) -> None:
    all_people = people
    (
        people,
        token_counts,
        term_firsts,
        term_surnames,
        person_firsts,
        person_surnames,
    ) = _assignment_score_context(results, people)

    for result in results:
        result.person = None
        result.confidence = 0.0
        result.output_name = result_output_name(result, people=all_people, token_counts=token_counts)
    if not people:
        return

    scores: list[list[float]] = []
    weights: list[list[float]] = []
    for person in people:
        person_scores: list[float] = []
        person_weights: list[float] = []
        for result in results:
            _direct, _after_first, _surname, _compound, score = _assignment_pair_scores(
                result,
                person,
                token_counts,
                term_firsts,
                term_surnames,
                person_firsts,
                person_surnames,
                results,
                people,
            )
            person_scores.append(score)
            if score >= MIN_REVIEW_MATCH:
                tie_break = min(0.0009, _ocr_specificity_priority(result, person) * 0.001)
                person_weights.append(score + tie_break)
            else:
                person_weights.append(0.0)
        scores.append(person_scores)
        # One private dummy column per person allows every uncertain identity to
        # remain unmatched without stealing a real term from another identity.
        weights.append(person_weights + [0.0] * len(people))

    assignment = _hungarian_maximize(weights)
    for person_index, result_index in enumerate(assignment):
        if result_index < 0 or result_index >= len(results):
            continue
        score = scores[person_index][result_index]
        if score < MIN_REVIEW_MATCH:
            continue
        row_competitor = max(
            (value for index, value in enumerate(scores[person_index]) if index != result_index),
            default=0.0,
        )
        column_competitor = max(
            (
                scores[other_person_index][result_index]
                for other_person_index in range(len(people))
                if other_person_index != person_index
            ),
            default=0.0,
        )
        confidence = _confidence_from_scores(score, max(row_competitor, column_competitor))
        result = results[result_index]
        person = people[person_index]
        result.person = person
        result.confidence = confidence
        result.output_name = result_output_name(result, person, people, token_counts)


def face_count_and_crop(photo: Image.Image, box_ratio: float) -> tuple[Image.Image, int]:
    if cv2 is None or not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return ImageOps.contain(photo, (1200, 1200)), 0
    rgb = np.array(photo.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
    return photo, len(faces)


def manual_photo_crop(
    photo: Image.Image,
    size: tuple[int, int],
    zoom: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Image.Image:
    target_w, target_h = size
    photo = photo.convert("RGB")
    base_scale = max(target_w / photo.width, target_h / photo.height)
    scale = base_scale * max(1.0, zoom)
    resized = photo.resize(
        (max(target_w, round(photo.width * scale)), max(target_h, round(photo.height * scale))),
        Image.Resampling.LANCZOS,
    )
    excess_x = max(0, resized.width - target_w)
    excess_y = max(0, resized.height - target_h)
    left = round(excess_x * (max(-1.0, min(1.0, offset_x)) + 1.0) / 2.0)
    top = round(excess_y * (max(-1.0, min(1.0, offset_y)) + 1.0) / 2.0)
    return resized.crop((left, top, left + target_w, top + target_h))


def safe_photo_rect(page: Image.Image, rect: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(value) for value in rect)
    if y > 0.12 or x < 0.60:
        return x, y, w, h

    right = min(0.985, x + w)
    top = max(0.004, y)
    max_height = max(0.045, TOP_RIGHT_PHOTO_SAFE_BOTTOM - top)
    if h > max_height:
        scale = max_height / h
        w *= scale
        h = max_height
        x = right - w

    # Keep a small right margin and preserve the model's top-right anchoring.
    x = max(0.02, min(x, right - w))
    return x, top, w, h


def place_photo(
    page: Image.Image,
    photo_path: Path,
    rect: Iterable[float],
    manual_crop: bool = False,
    zoom: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Image.Image:
    x, y, w, h = safe_photo_rect(page, rect)
    px, py = int(page.width * x), int(page.height * y)
    pw, ph = int(page.width * w), int(page.height * h)
    photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    if manual_crop:
        fitted = manual_photo_crop(photo, (pw, ph), zoom, offset_x, offset_y)
    else:
        # A composição automática nunca remove partes da foto. O modo de
        # ajuste manual continua disponível quando o usuário quiser preencher
        # toda a caixa com um enquadramento escolhido por ele.
        fitted = ImageOps.contain(photo, (pw, ph), Image.Resampling.LANCZOS)
        contained = Image.new("RGB", (pw, ph), "white")
        contained.paste(fitted, ((pw - fitted.width) // 2, (ph - fitted.height) // 2))
        fitted = contained
    border = max(2, page.width // 600)
    frame = Image.new("RGB", (pw + 2 * border, ph + 2 * border), "white")
    frame.paste(fitted, (border, border))
    result = page.copy()
    result.paste(frame, (px - border, py - border))
    return result


def save_pdf(page: Image.Image, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    page.convert("RGB").save(output, "PDF", resolution=200.0, quality=90, optimize=True)


def ocr_custom_names(people: list[PersonPhoto]) -> list[str]:
    names: list[str] = []
    for person in matchable_people(people):
        label = re.sub(r"\s+", " ", person.label).strip()
        if label and normalize(label) not in NON_PERSON_CAPTIONS:
            names.append(label)
    return list(dict.fromkeys(names))


def analysis_worker_count(total: int) -> int:
    if total <= 1:
        return 1
    cpu_count = os.cpu_count() or 2
    return max(1, min(total, 3 if sys.platform == "darwin" else min(4, cpu_count)))


def analyze_term_source(source: Path, models: dict[str, dict], custom_names: list[str]) -> TermResult:
    raw = load_source(source)
    page = normalize_page(raw)
    source_name = source_name_candidate(source)
    # Do not feed every photo/video name into Vision: it can bias cursive OCR
    # toward the wrong person. Only the term's own useful filename is trusted.
    trusted_ocr_hints = [source_name] if source_name else []
    if FAST_TRUST_SOURCE_NAMED_TERMS and source_name:
        ocr = f"NOME_CANDIDATO\n{source_name}"
        model = classify_model(ocr, models)
        output_name = result_output_name(TermResult(source, page, ocr, model, None, 0.0, source_name), people=[], token_counts={})
        return TermResult(source, page, ocr, model, None, 0.0, output_name, original_page=raw.copy())

    focused = focused_name_ocr(page, trusted_ocr_hints or custom_names, max_regions=MAX_FOCUSED_NAME_REGIONS_WITHOUT_HINT)
    if focused and extract_name_candidate(focused):
        ocr = f"LEITURA_FOCADA_DO_NOME\n{focused}"
    else:
        ocr = focused or ""
        if not ocr.strip():
            ocr = ""
    model = classify_model(ocr, models)
    output_name = source_name or extract_name_candidate(ocr) or source.stem
    return TermResult(source, page, ocr, model, None, 0.0, output_name, original_page=raw.copy())


def fit_term_crop_to_a4(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    fitted = ImageOps.contain(image, A4_SIZE, Image.Resampling.LANCZOS)
    page = Image.new("RGB", A4_SIZE, "white")
    page.paste(fitted, ((A4_SIZE[0] - fitted.width) // 2, (A4_SIZE[1] - fitted.height) // 2))
    return ImageEnhance.Contrast(page).enhance(1.035)


class TermCropEditor(tk.Toplevel):
    def __init__(self, parent: tk.Misc, result: TermResult, on_save) -> None:
        super().__init__(parent)
        self.title("Corrigir recorte do termo")
        self.configure(background=PALETTE["bg"])
        self.result = result
        self.on_save = on_save
        self.original = (result.original_page or result.page).copy().convert("RGB")
        self.left = tk.DoubleVar(value=0.0)
        self.top = tk.DoubleVar(value=0.0)
        self.right = tk.DoubleVar(value=100.0)
        self.bottom = tk.DoubleVar(value=100.0)
        self.rotation = tk.IntVar(value=0)
        self.preview_image: ImageTk.PhotoImage | None = None

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        self.canvas_w = 520
        self.canvas_h = 640
        self.canvas = tk.Canvas(
            frame,
            width=self.canvas_w,
            height=self.canvas_h,
            bg=PALETTE["surface_alt"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
        )
        self.canvas.grid(row=0, column=0, rowspan=12, padx=(0, 14))

        ttk.Label(frame, text="Ajuste as bordas e gire a folha se a foto estiver de lado.").grid(row=0, column=1, sticky="w")
        for row, label, variable in (
            (1, "Esquerda", self.left),
            (3, "Topo", self.top),
            (5, "Direita", self.right),
            (7, "Base", self.bottom),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=1, sticky="w", pady=(10, 0))
            ttk.Scale(frame, from_=0, to=100, variable=variable, command=self._render).grid(row=row + 1, column=1, sticky="ew")

        rotation_row = ttk.Frame(frame)
        rotation_row.grid(row=9, column=1, sticky="ew", pady=(14, 0))
        ttk.Button(rotation_row, text="Girar esquerda", command=lambda: self._rotate(-90)).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(rotation_row, text="Girar direita", command=lambda: self._rotate(90)).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.rotation_label = ttk.Label(frame, text="Rotação: 0°")
        self.rotation_label.grid(row=10, column=1, sticky="w", pady=(6, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=11, column=1, sticky="sew", pady=(18, 0))
        ttk.Button(actions, text="Usar termo inteiro", command=self._reset).pack(side="left")
        ttk.Button(actions, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="Salvar recorte", command=self._save).pack(side="right")
        frame.columnconfigure(1, weight=1, minsize=350)
        self._render()
        self.transient(parent)
        self.grab_set()

    def _working_image(self) -> Image.Image:
        angle = self.rotation.get() % 360
        if not angle:
            return self.original
        return self.original.rotate(-angle, expand=True)

    def _crop_box(self) -> tuple[int, int, int, int]:
        image = self._working_image()
        left = min(self.left.get(), self.right.get() - 2)
        right = max(self.right.get(), self.left.get() + 2)
        top = min(self.top.get(), self.bottom.get() - 2)
        bottom = max(self.bottom.get(), self.top.get() + 2)
        return (
            round(image.width * max(0, left) / 100),
            round(image.height * max(0, top) / 100),
            round(image.width * min(100, right) / 100),
            round(image.height * min(100, bottom) / 100),
        )

    def _render(self, _value=None) -> None:
        image = self._working_image()
        preview = image.copy()
        preview.thumbnail((self.canvas_w - 24, self.canvas_h - 24), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        x0 = (self.canvas_w - preview.width) // 2
        y0 = (self.canvas_h - preview.height) // 2
        self.canvas.create_image(x0, y0, image=self.preview_image, anchor="nw")
        crop = self._crop_box()
        scale_x = preview.width / image.width
        scale_y = preview.height / image.height
        rect = (
            x0 + crop[0] * scale_x,
            y0 + crop[1] * scale_y,
            x0 + crop[2] * scale_x,
            y0 + crop[3] * scale_y,
        )
        self.canvas.create_rectangle(rect, outline=PALETTE["accent"], width=3)
        self.rotation_label.configure(text=f"Rotação: {self.rotation.get() % 360}°")

    def _reset(self) -> None:
        self.left.set(0.0)
        self.top.set(0.0)
        self.right.set(100.0)
        self.bottom.set(100.0)
        self.rotation.set(0)
        self._render()

    def _rotate(self, delta: int) -> None:
        self.rotation.set((self.rotation.get() + delta) % 360)
        self._render()

    def _save(self) -> None:
        crop = self._working_image().crop(self._crop_box())
        self.result.page = fit_term_crop_to_a4(crop)
        self.on_save()
        self.destroy()


class PhotoEditor(tk.Toplevel):
    def __init__(self, parent: tk.Misc, result: TermResult, rect: Iterable[float], on_save) -> None:
        super().__init__(parent)
        if result.person is None:
            raise ValueError("Selecione ou adicione uma foto antes de ajustar o enquadramento.")
        self.title("Mover e recortar foto")
        self.configure(background=PALETTE["bg"])
        self.resizable(False, False)
        self.result = result
        self.on_save = on_save
        self.photo = ImageOps.exif_transpose(Image.open(result.person.image_path)).convert("RGB")
        _, _, rw, rh = rect
        ratio = (A4_SIZE[0] * rw) / (A4_SIZE[1] * rh)
        self.preview_h = 420
        self.preview_w = max(220, round(self.preview_h * ratio))
        self.zoom = tk.DoubleVar(value=result.photo_zoom if result.manual_crop else 1.0)
        self.offset_x = tk.DoubleVar(value=result.photo_offset_x if result.manual_crop else 0.0)
        self.offset_y = tk.DoubleVar(value=result.photo_offset_y if result.manual_crop else 0.0)
        self._drag_origin: tuple[int, int, float, float] | None = None

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            frame,
            width=self.preview_w,
            height=self.preview_h,
            bg=PALETTE["surface_alt"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
        )
        self.canvas.grid(row=0, column=0, rowspan=8, padx=(0, 14))
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag)

        ttk.Label(frame, text="Arraste a imagem ou use os controles.").grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="Zoom").grid(row=1, column=1, sticky="w")
        ttk.Scale(frame, from_=1.0, to=4.0, variable=self.zoom, command=self._render).grid(row=2, column=1, sticky="ew")
        ttk.Label(frame, text="Mover horizontalmente").grid(row=3, column=1, sticky="w", pady=(10, 0))
        ttk.Scale(frame, from_=-1.0, to=1.0, variable=self.offset_x, command=self._render).grid(row=4, column=1, sticky="ew")
        ttk.Label(frame, text="Mover verticalmente").grid(row=5, column=1, sticky="w", pady=(10, 0))
        ttk.Scale(frame, from_=-1.0, to=1.0, variable=self.offset_y, command=self._render).grid(row=6, column=1, sticky="ew")
        actions = ttk.Frame(frame)
        actions.grid(row=7, column=1, sticky="sew", pady=(18, 0))
        ttk.Button(actions, text="Redefinir", command=self._reset).pack(side="left")
        ttk.Button(actions, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="Salvar enquadramento", command=self._save).pack(side="right")
        frame.columnconfigure(1, weight=1, minsize=330)
        self._render()
        self.transient(parent)
        self.grab_set()

    def _render(self, _value=None) -> None:
        preview = manual_photo_crop(
            self.photo,
            (self.preview_w, self.preview_h),
            self.zoom.get(),
            self.offset_x.get(),
            self.offset_y.get(),
        )
        self.tk_preview = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_preview, anchor="nw")

    def _start_drag(self, event) -> None:
        self._drag_origin = (event.x, event.y, self.offset_x.get(), self.offset_y.get())

    def _drag(self, event) -> None:
        if not self._drag_origin:
            return
        x0, y0, ox, oy = self._drag_origin
        self.offset_x.set(max(-1.0, min(1.0, ox - (event.x - x0) * 2 / self.preview_w)))
        self.offset_y.set(max(-1.0, min(1.0, oy - (event.y - y0) * 2 / self.preview_h)))
        self._render()

    def _reset(self) -> None:
        self.zoom.set(1.0)
        self.offset_x.set(0.0)
        self.offset_y.set(0.0)
        self._render()

    def _save(self) -> None:
        self.result.manual_crop = True
        self.result.photo_zoom = self.zoom.get()
        self.result.photo_offset_x = self.offset_x.get()
        self.result.photo_offset_y = self.offset_y.get()
        self.on_save()
        self.destroy()


class PhotoPicker(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        choices: list[tuple[str, PersonPhoto]],
        current_label: str,
        on_pick,
        usage_lookup: dict[tuple[str, str], str] | None = None,
        title: str = "Escolher foto",
        action_text: str = "Usar esta foto",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.configure(background=PALETTE["bg"])
        self.choices = choices
        self.on_pick = on_pick
        self.usage_lookup = usage_lookup or {}
        self.preview_image: ImageTk.PhotoImage | None = None

        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=0, minsize=430)
        frame.columnconfigure(1, weight=1, minsize=420)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Fotos disponíveis", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(frame, text="Prévia", style="PanelTitle.TLabel").grid(row=0, column=1, sticky="w", padx=(14, 0), pady=(0, 8))

        columns = ("nome", "status", "arquivo")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", height=16)
        self.tree.heading("nome", text="Nome")
        self.tree.heading("status", text="Status")
        self.tree.heading("arquivo", text="Arquivo")
        self.tree.column("nome", width=160, anchor="w")
        self.tree.column("status", width=120, anchor="w")
        self.tree.column("arquivo", width=145, anchor="w")
        self.tree.tag_configure("used", foreground=PALETTE["success"])
        self.tree.tag_configure("unused", foreground=PALETTE["warning"])
        for index, (label, person) in enumerate(choices):
            status = self.usage_lookup.get(person_identity(person), "Sem termo")
            tag = "used" if status.startswith("Selecionada") else "unused"
            self.tree.insert("", "end", iid=str(index), values=(label, status, person.image_path.name), tags=(tag,))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=0, sticky="nse")

        preview_box = ttk.Frame(frame, style="Card.TFrame", padding=10)
        preview_box.grid(row=1, column=1, sticky="nsew", padx=(14, 0))
        preview_box.rowconfigure(0, weight=1)
        preview_box.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview_box, width=420, height=420, bg=PALETTE["surface_alt"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.info = ttk.Label(preview_box, text="", style="Hint.TLabel", wraplength=420)
        self.info.grid(row=1, column=0, sticky="w", pady=(8, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(actions, text="Cancelar", command=self.destroy, style="Quiet.TButton").pack(side="right", padx=(8, 0))
        ttk.Button(actions, text=action_text, command=self._choose, style="Secondary.TButton").pack(side="right")

        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._render())
        self.tree.bind("<Double-1>", lambda _event: self._choose())
        if choices:
            selected = next((str(i) for i, (label, _) in enumerate(choices) if label == current_label), "0")
            self.tree.selection_set(selected)
            self.tree.focus(selected)
            self.tree.see(selected)
            self._render()
        self.transient(parent)
        self.grab_set()

    def _selected_choice(self) -> tuple[str, PersonPhoto] | None:
        selection = self.tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self.choices):
            return self.choices[index]
        return None

    def _render(self) -> None:
        self.canvas.delete("all")
        choice = self._selected_choice()
        if not choice:
            return
        label, person = choice
        status = self.usage_lookup.get(person_identity(person), "Sem termo")
        try:
            image = ImageOps.exif_transpose(Image.open(person.image_path)).convert("RGB")
            image.thumbnail((420, 420), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image)
            width = max(self.canvas.winfo_width(), 420)
            height = max(self.canvas.winfo_height(), 420)
            self.canvas.create_image(width // 2, height // 2, image=self.preview_image, anchor="center")
        except Exception:
            self.preview_image = None
            self.canvas.create_text(210, 210, text="Não foi possível abrir a foto", fill=PALETTE["danger"])
        self.info.configure(text=f"{label} · {status} · {person.image_path.name}")

    def _choose(self) -> None:
        choice = self._selected_choice()
        if not choice:
            return
        self.on_pick(*choice)
        self.destroy()


class VideoAuditWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, default_authorization_folder: Path | None = None) -> None:
        super().__init__(parent)
        self.title("Encontrar autorizações nos vídeos")
        self.geometry("1380x900")
        self.minsize(1080, 760)
        self.configure(background=PALETTE["bg"])
        self.video_paths: list[Path] = []
        self.video_root: Path | None = None
        self.video_drive_links: list[str] = []
        self.authorization_folder = (
            default_authorization_folder
            if default_authorization_folder and default_authorization_folder.exists()
            else None
        )
        self.authorization_drive_links: list[str] = []
        self.drive_output_root: Path | None = None
        self.worker_thread: threading.Thread | None = None
        self.work_queue: queue.Queue = queue.Queue()
        self.temp_holder: tempfile.TemporaryDirectory | None = None
        self.batches: list[VideoAuditBatch] = []
        self.tab_views: dict[int, dict] = {}
        self.sheet_images: dict[int, ImageTk.PhotoImage] = {}

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        ttk.Label(root, text="Autorizações necessárias por vídeo", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")

        sources = ttk.Frame(root, style="Card.TFrame", padding=10)
        sources.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        sources.columnconfigure(0, weight=1)
        sources.columnconfigure(1, weight=1)
        self._build_video_source(sources)
        self._build_authorization_source(sources)

        action = ttk.Frame(root)
        action.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        action.columnconfigure(1, weight=1)
        self.audit_button = ttk.Button(action, text="Analisar lote", command=self.start_analysis, style="Accent.TButton")
        self.audit_button.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(action, mode="determinate", maximum=1, value=0)
        self.progress.grid(row=0, column=1, sticky="ew", padx=10)
        self.status = ttk.Label(action, text="Selecione os vídeos e a base de autorizações.", style="Hint.TLabel")
        self.status.grid(row=0, column=2, sticky="e")

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=3, column=0, sticky="nsew")
        self._show_initial_tab()
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_video_source(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="Card.TFrame", padding=(4, 2, 12, 2))
        box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        box.columnconfigure((0, 1), weight=1)
        ttk.Label(box, text="Vídeos finais", style="Editor.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.video_source_status = tk.Label(
            box,
            text="○ Nenhum vídeo selecionado",
            bg=PALETTE["surface"],
            fg=PALETTE["muted"],
            anchor="w",
            justify="left",
            wraplength=560,
        )
        self.video_source_status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 6))
        ttk.Button(box, text="Escolher arquivos", command=self.choose_video_files, style="Secondary.TButton").grid(row=2, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(box, text="Pasta no Finder", command=self.choose_video_folder, style="Quiet.TButton").grid(row=2, column=1, sticky="ew", padx=(4, 0))
        self.video_drive_text = tk.Text(box, height=2, wrap="word", bg=PALETTE["surface_alt"], fg=PALETTE["ink"], insertbackground=PALETTE["ink"], relief="flat")
        self.video_drive_text.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(box, text="Usar link do Drive", command=self.use_video_drive_links, style="Quiet.TButton").grid(row=3, column=1, sticky="nsew", padx=(8, 0), pady=(7, 0))

    def _build_authorization_source(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent, style="Card.TFrame", padding=(12, 2, 4, 2))
        box.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        box.columnconfigure((0, 1), weight=1)
        ttk.Label(box, text="Base de autorizações (PDF ou vídeo)", style="Editor.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        authorization_text = (
            f"✓ Base recebida: {self.authorization_folder}"
            if self.authorization_folder
            else "○ Nenhuma base selecionada"
        )
        self.authorization_source_status = tk.Label(
            box,
            text=authorization_text,
            bg=PALETTE["surface"],
            fg=PALETTE["success"] if self.authorization_folder else PALETTE["muted"],
            anchor="w",
            justify="left",
            wraplength=560,
        )
        self.authorization_source_status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 6))
        ttk.Button(box, text="Pasta no Finder", command=self.choose_authorization_folder, style="Secondary.TButton").grid(row=2, column=0, columnspan=2, sticky="ew")
        self.authorization_drive_text = tk.Text(box, height=2, wrap="word", bg=PALETTE["surface_alt"], fg=PALETTE["ink"], insertbackground=PALETTE["ink"], relief="flat")
        self.authorization_drive_text.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(box, text="Usar link do Drive", command=self.use_authorization_drive_links, style="Quiet.TButton").grid(row=3, column=1, sticky="nsew", padx=(8, 0), pady=(7, 0))

    def _show_initial_tab(self) -> None:
        page = ttk.Frame(self.notebook, padding=20)
        ttk.Label(page, text="Os lotes aparecerão aqui depois da análise.", style="Hint.TLabel").pack(anchor="center", pady=100)
        self.notebook.add(page, text="Lotes")

    def _clear_tabs(self) -> None:
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.tab_views.clear()
        self.sheet_images.clear()

    @staticmethod
    def _set_source_status(label: tk.Label, text: str, ready: bool, error: bool = False) -> None:
        if error:
            symbol, color = "!", PALETTE["danger"]
        elif ready:
            symbol, color = "✓", PALETTE["success"]
        else:
            symbol, color = "○", PALETTE["muted"]
        label.configure(text=f"{symbol} {text}", fg=color)

    def choose_video_files(self) -> None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(VIDEO_SUPPORTED))
        values = filedialog.askopenfilenames(title="Escolher vídeos finais", filetypes=[("Vídeos", patterns), ("Todos", "*.*")])
        if values:
            self.video_paths = [Path(value) for value in values]
            parents = [str(path.parent) for path in self.video_paths]
            self.video_root = Path(os.path.commonpath(parents))
            self.video_drive_links = []
            self._set_source_status(
                self.video_source_status,
                f"Recebidos {len(self.video_paths)} vídeo(s) deste Mac",
                True,
            )

    def choose_video_folder(self) -> None:
        value = filedialog.askdirectory(title="Pasta com vídeos finais")
        if value:
            self.video_root = Path(value)
            self.video_paths = video_files_from_folder(self.video_root)
            self.video_drive_links = []
            self._set_source_status(
                self.video_source_status,
                f"Recebidos {len(self.video_paths)} vídeo(s) da pasta {self.video_root.name}",
                True,
            )

    def use_video_drive_links(self) -> None:
        links = drive_folder_links_from_text(self.video_drive_text.get("1.0", "end"))
        if not links:
            self._set_source_status(self.video_source_status, "Link de pasta do Google Drive não reconhecido", False, error=True)
            messagebox.showwarning("Link não reconhecido", "Cole um link de pasta do Google Drive.")
            return
        self.video_drive_links = links
        self.video_paths = []
        self.video_root = None
        self._set_source_status(
            self.video_source_status,
            f"Recebidos {len(links)} link(s) de pasta do Google Drive",
            True,
        )

    def choose_authorization_folder(self) -> None:
        value = filedialog.askdirectory(title="Base de autorizações")
        if value:
            self.authorization_folder = Path(value)
            self.authorization_drive_links = []
            count = len(authorization_files_from_folder(self.authorization_folder))
            self._set_source_status(
                self.authorization_source_status,
                f"Recebidas {count} autorização(ões) da pasta {self.authorization_folder.name}",
                True,
            )

    def use_authorization_drive_links(self) -> None:
        links = drive_folder_links_from_text(self.authorization_drive_text.get("1.0", "end"))
        if not links:
            self._set_source_status(self.authorization_source_status, "Link de pasta do Google Drive não reconhecido", False, error=True)
            messagebox.showwarning("Link não reconhecido", "Cole um link de pasta do Google Drive.")
            return
        self.authorization_drive_links = links
        self.authorization_folder = None
        self._set_source_status(
            self.authorization_source_status,
            f"Recebidos {len(links)} link(s) de pasta do Google Drive",
            True,
        )

    def start_analysis(self) -> None:
        if not self.video_paths and not self.video_drive_links:
            messagebox.showwarning("Faltam os vídeos", "Escolha arquivos, uma pasta do Finder ou um link do Google Drive.")
            return
        if not self.authorization_folder and not self.authorization_drive_links:
            messagebox.showwarning("Falta a base", "Escolha a pasta ou o link com as autorizações.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Análise em andamento", "Aguarde o lote atual terminar.")
            return
        if self.video_drive_links:
            destination = filedialog.askdirectory(title="Onde salvar as pastas AUTORIZACAO DE IMAGEM")
            if not destination:
                return
            self.drive_output_root = Path(destination)
        else:
            self.drive_output_root = None
            local_root = self.video_root or Path(os.path.commonpath([str(path.parent) for path in self.video_paths]))
            for video_path in self.video_paths:
                try:
                    authorization_output_folder(video_path, local_root).mkdir(parents=True, exist_ok=True)
                except OSError:
                    destination = filedialog.askdirectory(title="A pasta original é somente leitura. Onde salvar as autorizações?")
                    if not destination:
                        return
                    self.drive_output_root = Path(destination)
                    break
        if self.temp_holder:
            self.temp_holder.cleanup()
        self.temp_holder = tempfile.TemporaryDirectory(prefix="termos_video_audit_")
        self.batches = []
        self._clear_tabs()
        self.work_queue = queue.Queue()
        self.audit_button.state(["disabled"])
        self.progress.configure(maximum=1, value=0)
        self.status.configure(text="Preparando o lote...")
        self.worker_thread = threading.Thread(target=self._worker, args=(Path(self.temp_holder.name),), daemon=True)
        self.worker_thread.start()
        self.after(150, self._poll)

    def _download_sources(self, work_dir: Path, emit) -> tuple[list[Path], Path, list[Path]]:
        if self.video_drive_links:
            video_root = work_dir / "videos_drive"
            for index, link in enumerate(self.video_drive_links, start=1):
                download_drive_folder(
                    link,
                    video_root / f"pasta_{index}",
                    supported=VIDEO_SUPPORTED,
                    kind_label="vídeos",
                    progress_callback=lambda done, total, name, eta: emit("phase", message=f"Baixando vídeo {done}/{total}: {name}"),
                )
            video_paths = video_files_from_folder(video_root)
        else:
            video_paths = list(self.video_paths)
            video_root = self.video_root or Path(os.path.commonpath([str(path.parent) for path in video_paths]))

        authorization_roots: list[Path] = []
        if self.authorization_folder:
            authorization_roots.append(self.authorization_folder)
        for index, link in enumerate(self.authorization_drive_links, start=1):
            root = work_dir / "autorizacoes_drive" / f"pasta_{index}"
            download_drive_folder(
                link,
                root,
                supported=AUTHORIZATION_SUPPORTED,
                kind_label="autorizações",
                progress_callback=lambda done, total, name, eta: emit("phase", message=f"Baixando autorização {done}/{total}: {name}"),
            )
            authorization_roots.append(root)
        return video_paths, video_root, authorization_roots

    def _worker(self, work_dir: Path) -> None:
        def emit(kind: str, **payload) -> None:
            self.work_queue.put((kind, payload))

        try:
            video_paths, video_root, authorization_roots = self._download_sources(work_dir, emit)
            if not video_paths:
                raise ValueError("Não encontrei vídeos nas pastas selecionadas.")
            emit("phase", message="Lendo a base de autorizações...")
            authorization_people: list[PersonPhoto] = []
            authorization_failures: list[str] = []
            for root in authorization_roots:
                people, failures = authorization_people_from_folder(root)
                authorization_people.extend(people)
                authorization_failures.extend(failures)
            unique_people: dict[tuple[str, Path], PersonPhoto] = {}
            for person in authorization_people:
                unique_people[(normalize(person.label), person.image_path.resolve())] = person
            authorization_people = list(unique_people.values())
            if not authorization_people:
                raise ValueError("Não encontrei autorizações com nome reconhecível na base.")
            emit("phase", message="Preparando referências faciais locais das autorizações...")
            face_references = build_authorization_face_references(
                authorization_people,
                work_dir / "referencias_faciais",
                progress_callback=lambda current, total, count: emit(
                    "phase",
                    message=f"Referências faciais {current}/{total} · {count} rosto(s) utilizável(is)",
                ),
            )
            emit("batch_count", total=len(video_paths), authorizations=len(authorization_people), failures=authorization_failures)

            for index, video_path in enumerate(video_paths):
                batch_dir = work_dir / "resultados" / f"video_{index + 1:03d}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                try:
                    relative = video_path.relative_to(video_root)
                except ValueError:
                    relative = Path(video_path.name)
                output_dir = authorization_output_folder(video_path, video_root, self.drive_output_root)
                batch = VideoAuditBatch(video_path=video_path, source_root=video_root, relative_path=relative, output_dir=output_dir)
                emit("phase", message=f"{index + 1}/{len(video_paths)} · Ouvindo {video_path.name}")
                try:
                    transcript = _faster_whisper_transcribe_video(
                        video_path,
                        max_seconds=VIDEO_AUDIT_TRANSCRIPTION_SECONDS,
                        careful=False,
                        roster_names=None,
                        name_prompt=False,
                    )
                except Exception as exc:
                    transcript = ""
                    batch.error = f"Áudio: {exc}"
                batch.transcript = transcript
                visual_evidence: list[VideoVisualEvidence] = []
                emit("phase", message=f"{index + 1}/{len(video_paths)} · Buscando rostos em {video_path.name}")
                try:
                    previews = extract_video_face_previews(
                        video_path,
                        batch_dir / "rostos",
                        visual_text_callback=lambda timestamp, text: visual_evidence.append(VideoVisualEvidence(timestamp, text)),
                        visual_name_hints=(person.label for person in authorization_people),
                        progress_callback=lambda current, total, count, i=index: emit(
                            "video_progress", index=i, current=current, total=total, faces=count,
                        ),
                    )
                except Exception as exc:
                    previews = []
                    batch.error = f"{batch.error} · Rostos: {exc}".strip(" ·")
                batch.visual_evidence = visual_evidence
                batch.face_previews = previews
                name_suggestions = video_authorization_suggestions(
                    transcript,
                    visual_evidence,
                    authorization_people,
                )
                # A tarja inequívoca identifica a miniatura e também serve de
                # referência para reencontrar a mesma pessoa no próprio vídeo
                # e nos vídeos seguintes, sempre de forma local.
                named_references = named_face_references(
                    previews,
                    visual_evidence,
                    authorization_people,
                )
                face_suggestions = video_face_identity_suggestions(
                    previews,
                    face_references + named_references,
                )
                batch.suggestions = merge_video_authorization_suggestions(
                    name_suggestions,
                    face_suggestions,
                )
                face_references.extend(named_references)
                batch.face_photos = {}
                for preview in previews:
                    if preview.authorization_path and preview.authorization_path not in batch.face_photos:
                        batch.face_photos[preview.authorization_path] = preview.image_path
                batch.filled_photo_count = sum(
                    1
                    for suggestion in batch.suggestions
                    if suggestion.pdf_path in batch.face_photos
                    and suggestion.pdf_path.suffix.lower() == ".pdf"
                    and "semfoto" in normalize(suggestion.pdf_path.stem)
                )
                if batch.filled_photo_count:
                    batch.suggestions = [
                        VideoAuditSuggestion(
                            suggestion.pdf_path,
                            suggestion.confidence,
                            suggestion.reason
                            + (
                                " + foto recuperada do vídeo"
                                if suggestion.pdf_path in batch.face_photos
                                and suggestion.pdf_path.suffix.lower() == ".pdf"
                                and "semfoto" in normalize(suggestion.pdf_path.stem)
                                else ""
                            ),
                        )
                        for suggestion in batch.suggestions
                    ]
                batch.sheet_path = video_face_contact_sheet(previews, batch_dir)
                try:
                    automatic = [
                        suggestion for suggestion in batch.suggestions
                        if suggestion.confidence >= 0.78
                    ]
                    batch.copied_paths = copy_authorization_suggestions(
                        automatic,
                        output_dir,
                        batch.face_photos,
                    )
                except Exception as exc:
                    batch.copied_paths = []
                    batch.error = f"{batch.error} · Saída: {exc}".strip(" ·")
                emit("batch_done", index=index, batch=batch, total=len(video_paths))
            emit("all_done", total=len(video_paths))
        except Exception as exc:
            emit("error", message=str(exc))

    def _poll(self) -> None:
        terminal = False
        while True:
            try:
                kind, payload = self.work_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "phase":
                self.status.configure(text=payload["message"])
            elif kind == "batch_count":
                self.progress.configure(maximum=max(1, payload["total"]), value=0)
                self.status.configure(text=f"{payload['total']} vídeo(s) · {payload['authorizations']} autorização(ões) disponíveis")
            elif kind == "video_progress":
                current = float(payload.get("current") or 0)
                total = float(payload.get("total") or 0)
                detail = f" · {format_timestamp(current)}/{format_timestamp(total)}" if total else ""
                self.status.configure(text=f"Vídeo {payload['index'] + 1}{detail} · {payload['faces']} rosto(s)")
            elif kind == "batch_done":
                batch = payload["batch"]
                self.batches.append(batch)
                self._create_batch_tab(payload["index"], batch)
                self.progress.configure(value=payload["index"] + 1)
            elif kind == "all_done":
                terminal = True
                self.audit_button.state(["!disabled"])
                total_terms = sum(len(batch.suggestions or []) for batch in self.batches)
                total_faces = sum(len(batch.face_previews or []) for batch in self.batches)
                self.status.configure(text=f"Concluído: {payload['total']} vídeo(s) · {total_terms} autorização(ões) · {total_faces} rosto(s)")
            elif kind == "error":
                terminal = True
                self.audit_button.state(["!disabled"])
                self.status.configure(text="Não foi possível analisar o lote.")
                messagebox.showerror("Autorizações por vídeo", payload["message"])
        if not terminal and self.worker_thread and self.worker_thread.is_alive():
            self.after(200, self._poll)

    def _create_batch_tab(self, index: int, batch: VideoAuditBatch) -> None:
        page = ttk.Frame(self.notebook, padding=10)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(1, weight=1)
        title = safe_filename(batch.video_path.stem)[:24]
        self.notebook.add(page, text=f"{index + 1}. {title}")
        summary = f"{len(batch.suggestions or [])} autorização(ões) · {len(batch.face_previews or [])} rosto(s) · {batch.output_dir}"
        if batch.filled_photo_count:
            summary += f" · {batch.filled_photo_count} termo(s) sem foto preenchido(s) pelo vídeo"
        if batch.error:
            summary += f" · {batch.error}"
        ttk.Label(page, text=summary, style="Hint.TLabel", wraplength=1200).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        left = ttk.Frame(page, style="Card.TFrame", padding=8)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="Autorizações encontradas", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        tree = ttk.Treeview(left, columns=("arquivo", "conf", "motivo"), show="headings", selectmode="extended", height=18)
        tree.heading("arquivo", text="Arquivo")
        tree.heading("conf", text="%")
        tree.heading("motivo", text="Evidência")
        tree.column("arquivo", width=260, anchor="w")
        tree.column("conf", width=48, anchor="center")
        tree.column("motivo", width=250, anchor="w")
        tree.tag_configure("strong", foreground=PALETTE["success"])
        tree.tag_configure("review", foreground=PALETTE["warning"])
        tree.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.grid(row=1, column=1, sticky="ns")
        for suggestion_index, suggestion in enumerate(batch.suggestions or []):
            tree.insert("", "end", iid=str(suggestion_index), values=(suggestion.pdf_path.name, f"{suggestion.confidence:.0%}", suggestion.reason), tags=("strong" if suggestion.confidence >= 0.82 else "review",))
        if batch.suggestions:
            tree.selection_set(tree.get_children())
        ttk.Button(left, text="Salvar seleção", command=lambda i=index: self._save_batch_selection(i), style="Secondary.TButton").grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        right = ttk.Frame(page, style="Card.TFrame", padding=8)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="Rostos em primeiro plano", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        canvas_frame = ttk.Frame(right, style="Card.TFrame")
        canvas_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 6))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        canvas = tk.Canvas(canvas_frame, bg=PALETTE["surface_alt"], highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        hscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        transcript = tk.Text(right, height=5, wrap="word", bg=PALETTE["surface_alt"], fg=PALETTE["ink"], insertbackground=PALETTE["ink"], relief="flat", padx=8, pady=6)
        transcript.grid(row=2, column=0, sticky="ew")
        transcript.insert("1.0", batch.transcript or "Nenhuma apresentação falada foi reconhecida.")
        transcript.configure(state="disabled")
        self.tab_views[index] = {"tree": tree, "canvas": canvas, "transcript": transcript}
        self._render_batch_sheet(index, batch)
        if index == 0:
            self.notebook.select(page)

    def _render_batch_sheet(self, index: int, batch: VideoAuditBatch) -> None:
        canvas: tk.Canvas = self.tab_views[index]["canvas"]
        canvas.delete("all")
        if not batch.sheet_path:
            canvas.configure(scrollregion=(0, 0, 760, 460))
            canvas.create_text(380, 230, text="Nenhum rosto em primeiro plano detectado.", fill=PALETTE["muted"], font=("Helvetica", 14, "bold"))
            return
        image = Image.open(batch.sheet_path).convert("RGB")
        photo = ImageTk.PhotoImage(image)
        self.sheet_images[index] = photo
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, image.width, image.height))

    def _save_batch_selection(self, index: int) -> None:
        batch = next((item for item in self.batches if item.video_path == self.batches[index].video_path), self.batches[index])
        view = self.tab_views.get(index)
        if not view:
            return
        selected = [batch.suggestions[int(iid)] for iid in view["tree"].selection() if int(iid) < len(batch.suggestions or [])]
        if not selected:
            messagebox.showwarning("Nenhuma autorização selecionada", "Selecione um ou mais arquivos na lista.")
            return
        destination = batch.output_dir
        if destination is None:
            value = filedialog.askdirectory(title="Pasta para salvar as autorizações")
            if not value:
                return
            destination = Path(value)
        try:
            paths = copy_authorization_suggestions(selected, destination, batch.face_photos)
        except Exception as exc:
            messagebox.showerror("Salvar autorizações", str(exc))
            return
        messagebox.showinfo("Autorizações salvas", f"{len(paths)} arquivo(s) em:\n{destination}")

    def _close(self) -> None:
        if self.temp_holder:
            self.temp_holder.cleanup()
            self.temp_holder = None
        self.destroy()


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Automatizador de Termos")
        self.geometry("1440x940")
        self.minsize(1080, 760)
        self.configure(background=PALETTE["bg"])
        self.models = load_models()
        self.zip_path: Path | None = None
        self.photo_folder: Path | None = None
        self.video_folder: Path | None = None
        self.term_paths: list[Path] = []
        self.output_dir: Path | None = None
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        self.people: list[PersonPhoto] = []
        self.results: list[TermResult] = []
        self.person_by_label: dict[str, PersonPhoto] = {}
        self.work_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.phase_message = ""
        self.phase_started = 0.0
        self.current_filter = "all"
        self.review_preview_image: ImageTk.PhotoImage | None = None
        self.shared_photo_path: Path | None = None
        self.text_menus: list[tk.Menu] = []
        self.link_refresh_after: str | None = None
        self.preview_zoom = 1.0
        self.preview_offset_x = 0
        self.preview_offset_y = 0
        self.preview_drag_origin: tuple[int, int, int, int] | None = None
        self._setup_styles()
        icon_path = APP_DIR / "assets" / "app_icon_128.png"
        if icon_path.exists():
            self.window_icon = tk.PhotoImage(file=str(icon_path))
            self.iconphoto(True, self.window_icon)
            self.header_icon = ImageTk.PhotoImage(Image.open(icon_path).resize((78, 78), Image.Resampling.LANCZOS))
        else:
            self.window_icon = self.header_icon = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = PALETTE["bg"]
        card = PALETTE["surface"]
        card_alt = PALETTE["surface_alt"]
        ink = PALETTE["ink"]
        muted = PALETTE["muted"]
        subtle = PALETTE["subtle"]
        accent = PALETTE["accent"]
        accent_hover = PALETTE["accent_hover"]
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=ink, font=("Helvetica", 12))
        style.configure("Header.TLabel", background=bg, foreground=ink)
        style.configure("Title.TLabel", background=bg, foreground=ink, font=("Helvetica", 25, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground=muted, font=("Helvetica", 12))
        style.configure("Badge.TLabel", background=card_alt, foreground=accent, padding=(10, 5), font=("Helvetica", 10, "bold"))
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("Card.TLabel", background=card, foreground=ink, font=("Helvetica", 12))
        style.configure("Step.TLabel", background=card, foreground=accent, font=("Helvetica", 10, "bold"))
        style.configure("CardTitle.TLabel", background=card, foreground=ink, font=("Helvetica", 16, "bold"))
        style.configure("Hint.TLabel", background=card, foreground=muted, font=("Helvetica", 10))
        style.configure("SmallHint.TLabel", background=card, foreground=muted, font=("Helvetica", 9))
        style.configure("Metric.TLabel", background=card, foreground=ink, font=("Helvetica", 18, "bold"))
        style.configure("PanelTitle.TLabel", background=card, foreground=ink, font=("Helvetica", 14, "bold"))
        style.configure("Status.TLabel", background=PALETTE["accent_soft"], foreground=accent, padding=(8, 4), font=("Helvetica", 10, "bold"))
        style.configure("TButton", background=card_alt, foreground=ink, borderwidth=0, padding=(10, 8), font=("Helvetica", 10, "bold"))
        style.map("TButton", background=[("active", "#272727"), ("disabled", "#1A1A1A")], foreground=[("disabled", subtle)])
        style.configure("Accent.TButton", background=accent, foreground="#121212", borderwidth=0, padding=(20, 12), font=("Helvetica", 13, "bold"))
        style.map("Accent.TButton", background=[("active", accent_hover), ("disabled", "#3A331E")], foreground=[("disabled", "#7E744D")])
        style.configure("Secondary.TButton", background=PALETTE["accent_soft"], foreground=accent, borderwidth=0, padding=(12, 9), font=("Helvetica", 11, "bold"))
        style.map("Secondary.TButton", background=[("active", PALETTE["surface_soft"])])
        style.configure("Quiet.TButton", background=card_alt, foreground=ink, borderwidth=0, padding=(10, 8), font=("Helvetica", 10, "bold"))
        style.map("Quiet.TButton", background=[("active", "#2B2B2B")])
        style.configure(
            "TEntry",
            fieldbackground=card_alt,
            background=card_alt,
            foreground=ink,
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            insertcolor=ink,
            padding=(8, 6),
        )
        style.map("TEntry", fieldbackground=[("focus", "#242424")])
        style.configure(
            "TCombobox",
            fieldbackground=card_alt,
            background=card_alt,
            foreground=ink,
            arrowcolor=accent,
            bordercolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            darkcolor=PALETTE["border"],
            padding=(8, 5),
        )
        style.map("TCombobox", fieldbackground=[("readonly", card_alt)], foreground=[("readonly", ink)])
        style.configure("Horizontal.TScale", background=bg, troughcolor=card_alt)
        style.configure("Vertical.TScrollbar", background=card_alt, troughcolor=bg, bordercolor=bg, arrowcolor=accent)
        style.map("Vertical.TScrollbar", background=[("active", "#2B2B2B")])
        style.configure(
            "Horizontal.TProgressbar",
            background=accent,
            troughcolor=card_alt,
            bordercolor=card_alt,
            lightcolor=accent,
            darkcolor=accent,
        )
        style.configure("Treeview", background=card, fieldbackground=card, foreground=ink, rowheight=30, borderwidth=0, font=("Helvetica", 10))
        style.configure("Treeview.Heading", background=card_alt, foreground=accent, relief="flat", padding=(6, 9), font=("Helvetica", 10, "bold"))
        style.map("Treeview", background=[("selected", PALETTE["accent_soft"])], foreground=[("selected", ink)])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=card_alt, foreground=muted, padding=(14, 8), font=("Helvetica", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", card), ("active", "#252525")], foreground=[("selected", accent), ("active", ink)])
        style.configure("Editor.TLabelframe", background=card, bordercolor=PALETTE["border"], relief="solid")
        style.configure("Editor.TLabelframe.Label", background=bg, foreground=ink, font=("Helvetica", 11, "bold"))
        style.configure("Editor.TFrame", background=card)
        style.configure("Editor.TLabel", background=card, foreground=ink, font=("Helvetica", 10))

    def _build_ui(self) -> None:
        colors = PALETTE
        shell = ttk.Frame(self, padding=(22, 14, 22, 14))
        shell.pack(fill="x")

        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 12))
        if self.header_icon:
            ttk.Label(header, image=self.header_icon, style="Header.TLabel").pack(side="left", padx=(0, 12))
        header_text = ttk.Frame(header)
        header_text.pack(side="left", fill="x", expand=True)
        ttk.Label(header_text, text="Automatizador de Termos", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header_text, text="Fotos, termos e PDFs prontos em um fluxo visual de revisão.", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        header_status = ttk.Frame(header)
        header_status.pack(side="right", anchor="n", pady=5)
        ttk.Label(header_status, text="VERSÃO 6.1", style="Badge.TLabel").pack(anchor="e")
        engine = ocr_engine_name()
        tk.Label(
            header_status,
            text="✓ LEITURA AUTOMÁTICA ATIVA" if engine else "! LEITURA AUTOMÁTICA INDISPONÍVEL",
            bg=colors["bg"],
            fg=colors["success"] if engine else colors["danger"],
            font=("Helvetica", 9, "bold"),
        ).pack(anchor="e", pady=(3, 0))

        steps = ttk.Frame(shell)
        steps.pack(fill="x", pady=(0, 10))
        self.setup_panel = steps
        for column in range(3):
            steps.columnconfigure(column, weight=1, uniform="steps")

        photos = ttk.Frame(steps, style="Card.TFrame", padding=16)
        photos.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Label(photos, text="PASSO 1 · OPCIONAL", style="Step.TLabel").pack(anchor="w")
        ttk.Label(photos, text="Fotos ou vídeos das pessoas", style="CardTitle.TLabel").pack(anchor="w", pady=(3, 2))
        ttk.Label(photos, text="Use WhatsApp, fotos nomeadas ou vídeos onde a pessoa fala o nome.", style="Hint.TLabel", wraplength=340).pack(anchor="w", pady=(0, 10))
        photo_buttons = ttk.Frame(photos, style="Card.TFrame")
        photo_buttons.pack(fill="x")
        ttk.Button(photo_buttons, text="ZIP do WhatsApp", command=self.choose_zip, style="Secondary.TButton").pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(photo_buttons, text="Pasta de fotos", command=self.choose_photo_folder, style="Quiet.TButton").pack(side="left", fill="x", expand=True, padx=(4, 0))
        video_buttons = ttk.Frame(photos, style="Card.TFrame")
        video_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(video_buttons, text="Vídeos do WhatsApp", command=self.choose_whatsapp_videos, style="Quiet.TButton").pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(video_buttons, text="Pasta de vídeos", command=self.choose_video_folder, style="Quiet.TButton").pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(photos, text="ou cole links de pastas do Google Drive com vídeos", style="Hint.TLabel").pack(anchor="w", pady=(8, 3))
        video_drive_row = ttk.Frame(photos, style="Card.TFrame")
        video_drive_row.pack(fill="x")
        self.video_drive_text = tk.Text(
            video_drive_row,
            height=2,
            wrap="word",
            bg=colors["surface_alt"],
            fg=colors["ink"],
            insertbackground=colors["ink"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["accent"],
            padx=8,
            pady=6,
            font=("Helvetica", 10),
        )
        self.video_drive_text.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._attach_text_menu(self.video_drive_text)
        self._watch_drive_text(self.video_drive_text)
        ttk.Button(video_drive_row, text="Usar links", command=self.confirm_video_drive, style="Quiet.TButton").pack(side="right")
        self.photo_status = tk.Label(photos, text="○ Nenhuma pessoa adicionada — os termos sairão como SEMFOTO", bg=colors["surface"], fg=colors["muted"], font=("Helvetica", 10), anchor="w", justify="left", wraplength=350)
        self.photo_status.pack(fill="x", pady=(11, 0))

        terms = ttk.Frame(steps, style="Card.TFrame", padding=16)
        terms.grid(row=0, column=1, sticky="nsew", padx=6)
        ttk.Label(terms, text="PASSO 2 · OBRIGATÓRIO", style="Step.TLabel").pack(anchor="w")
        ttk.Label(terms, text="Termos assinados", style="CardTitle.TLabel").pack(anchor="w", pady=(3, 2))
        ttk.Label(terms, text="Escolha arquivos, uma pasta do Finder/Drive ou cole um ou mais links.", style="Hint.TLabel", wraplength=340).pack(anchor="w", pady=(0, 8))
        term_buttons = ttk.Frame(terms, style="Card.TFrame")
        term_buttons.pack(fill="x")
        ttk.Button(term_buttons, text="Selecionar arquivos", command=self.choose_terms, style="Secondary.TButton").pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(term_buttons, text="Selecionar pasta", command=self.choose_term_folder, style="Quiet.TButton").pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(terms, text="ou cole links de pastas do Google Drive, um por linha", style="Hint.TLabel").pack(anchor="w", pady=(8, 3))
        drive_row = ttk.Frame(terms, style="Card.TFrame")
        drive_row.pack(fill="x")
        self.term_drive_text = tk.Text(
            drive_row,
            height=3,
            wrap="word",
            bg=colors["surface_alt"],
            fg=colors["ink"],
            insertbackground=colors["ink"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["accent"],
            padx=8,
            pady=6,
            font=("Helvetica", 10),
        )
        self.term_drive_text.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._attach_text_menu(self.term_drive_text)
        self._watch_drive_text(self.term_drive_text)
        ttk.Button(drive_row, text="Usar links", command=self.confirm_term_drive, style="Quiet.TButton").pack(side="right")
        self.term_status = tk.Label(terms, text="○ Nenhum termo adicionado", bg=colors["surface"], fg=colors["muted"], font=("Helvetica", 10), anchor="w", justify="left", wraplength=350)
        self.term_status.pack(fill="x", pady=(9, 0))

        output = ttk.Frame(steps, style="Card.TFrame", padding=16)
        output.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        ttk.Label(output, text="PASSO 3 · OBRIGATÓRIO", style="Step.TLabel").pack(anchor="w")
        ttk.Label(output, text="Onde salvar", style="CardTitle.TLabel").pack(anchor="w", pady=(3, 2))
        ttk.Label(output, text="Escolha a pasta que receberá os PDFs finalizados.", style="Hint.TLabel", wraplength=340).pack(anchor="w", pady=(0, 10))
        ttk.Button(output, text="Escolher pasta de saída", command=self.choose_output, style="Secondary.TButton").pack(fill="x")
        self.output_status = tk.Label(output, text="○ Nenhuma pasta selecionada", bg=colors["surface"], fg=colors["muted"], font=("Helvetica", 10), anchor="w", justify="left", wraplength=350)
        self.output_status.pack(fill="x", pady=(11, 0))

        action = ttk.Frame(shell, style="Card.TFrame", padding=(16, 10))
        action.pack(fill="x", pady=(0, 10))
        self.action_panel = action
        action_left = ttk.Frame(action, style="Card.TFrame")
        action_left.pack(side="left", fill="x", expand=True)
        ttk.Label(action_left, text="PASSO 4", style="Step.TLabel").pack(anchor="w")
        self.process_status = ttk.Label(action_left, text="Adicione os termos e escolha a pasta de saída.", style="Hint.TLabel")
        self.process_status.pack(anchor="w", pady=(2, 4))
        self.progress = ttk.Progressbar(action_left, mode="indeterminate")
        self.progress.pack(fill="x", padx=(0, 18))
        self.combine_button = ttk.Button(action, text="Iniciar combinação", command=self.analyze, style="Accent.TButton")
        self.combine_button.pack(side="right")

        video_tool = ttk.Frame(shell, style="Card.TFrame", padding=(16, 12))
        video_tool.pack(fill="x", pady=(0, 10))
        self.video_tool_panel = video_tool
        video_tool_left = ttk.Frame(video_tool, style="Card.TFrame")
        video_tool_left.pack(side="left", fill="x", expand=True)
        ttk.Label(video_tool_left, text="FERRAMENTA SEPARADA", style="Step.TLabel").pack(anchor="w")
        ttk.Label(video_tool_left, text="Encontrar autorizações nos vídeos", style="PanelTitle.TLabel").pack(anchor="w", pady=(2, 2))
        ttk.Label(
            video_tool_left,
            text="Use depois de gerar os PDFs. Ela confere um ou vários vídeos, mostra rostos em primeiro plano e separa as autorizações necessárias para cada vídeo.",
            style="Hint.TLabel",
            wraplength=880,
        ).pack(anchor="w")
        ttk.Button(video_tool, text="Abrir ferramenta de vídeo", command=self.open_video_audit, style="Secondary.TButton").pack(side="right")

        self.results_hint = ttk.Frame(self, style="Card.TFrame", padding=20)
        self.results_hint.pack(fill="x", padx=22, pady=(0, 14))
        ttk.Label(
            self.results_hint,
            text="A revisão visual e a exportação dos PDFs aparecerão aqui depois de combinar.",
            style="Card.TLabel",
        ).pack(anchor="center")

        self.review_panel = ttk.Frame(self, padding=(22, 0, 22, 14))
        self.review_panel.pack(fill="both", expand=True)
        self._build_review_ui(self.review_panel)
        self._set_results_visible(False)
        self._update_ready_state()

    def _build_review_ui(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="Card.TFrame", padding=(16, 12))
        top.pack(fill="x", pady=(0, 10))
        top_left = ttk.Frame(top, style="Card.TFrame")
        top_left.pack(side="left", fill="x", expand=True)
        ttk.Label(top_left, text="Revisão do lote", style="CardTitle.TLabel").pack(anchor="w")
        self.review_summary = ttk.Label(top_left, text="Nenhum lote combinado.", style="Hint.TLabel")
        self.review_summary.pack(anchor="w", pady=(3, 0))
        top_actions = ttk.Frame(top, style="Card.TFrame")
        top_actions.pack(side="right")
        ttk.Button(top_actions, text="Rodar novamente", command=self._show_setup, style="Quiet.TButton").pack(side="left", padx=(0, 8))
        self.photo_audit_button = ttk.Button(top_actions, text="Pessoas", command=self.open_photo_audit, style="Quiet.TButton")
        self.photo_audit_button.pack(side="left", padx=(0, 8))
        self.export_button = ttk.Button(top_actions, text="Exportar PDFs", command=self.generate, style="Accent.TButton")
        self.export_button.pack(side="left")

        filters = ttk.Frame(parent)
        filters.pack(fill="x", pady=(0, 8))
        self.filter_buttons: dict[str, ttk.Button] = {}
        for key, label in (
            ("all", "Todos"),
            ("review", "Revisar"),
            ("missing", "Sem foto"),
            ("duplicates", "Repetidos"),
            ("ok", "OK"),
        ):
            button = ttk.Button(filters, text=label, command=lambda value=key: self.set_filter(value), style="Quiet.TButton")
            button.pack(side="left", padx=(0, 6))
            self.filter_buttons[key] = button

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=305)
        body.columnconfigure(1, weight=1, minsize=340)
        body.columnconfigure(2, weight=0, minsize=305)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Card.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="Termos", style="PanelTitle.TLabel").pack(anchor="w")
        self.review_list_hint = ttk.Label(left, text="", style="SmallHint.TLabel")
        self.review_list_hint.pack(anchor="w", pady=(2, 8))
        columns = ("nome", "status", "conf")
        self.review_tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse", height=18)
        headings = {"nome": "Nome do PDF", "status": "Situação", "conf": "%"}
        widths = {"nome": 140, "status": 80, "conf": 42}
        for col in columns:
            self.review_tree.heading(col, text=headings[col])
            self.review_tree.column(col, width=widths[col], anchor="w", stretch=(col == "nome"))
        self.review_tree.tag_configure("ok", foreground=PALETTE["success"])
        self.review_tree.tag_configure("review", foreground=PALETTE["warning"])
        self.review_tree.tag_configure("missing", foreground=PALETTE["danger"])
        self.review_tree.tag_configure("duplicates", foreground=PALETTE["accent"])
        list_scroll = ttk.Scrollbar(left, orient="vertical", command=self.review_tree.yview)
        self.review_tree.configure(yscrollcommand=list_scroll.set)
        list_scroll.pack(side="right", fill="y")
        self.review_tree.pack(side="left", fill="both", expand=True)
        self.review_tree.bind("<<TreeviewSelect>>", self.on_select)

        center = ttk.Frame(body, style="Card.TFrame", padding=12)
        center.grid(row=0, column=1, sticky="nsew", padx=8)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)
        preview_toolbar = ttk.Frame(center, style="Card.TFrame")
        preview_toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Label(preview_toolbar, text="Prévia do termo", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(preview_toolbar, text="−", command=lambda: self.adjust_preview_zoom(0.82), style="Quiet.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(preview_toolbar, text="+", command=lambda: self.adjust_preview_zoom(1.22), style="Quiet.TButton").pack(side="right", padx=(4, 0))
        ttk.Button(preview_toolbar, text="100%", command=self.reset_preview_view, style="Quiet.TButton").pack(side="right", padx=(4, 0))
        self.preview_zoom_label = ttk.Label(preview_toolbar, text="100%", style="SmallHint.TLabel")
        self.preview_zoom_label.pack(side="right", padx=(0, 4))
        self.preview_canvas = tk.Canvas(center, background=PALETTE["surface_alt"], highlightthickness=0)
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        self.preview_canvas.bind("<Configure>", lambda _event: self._render_selected_preview())
        self.preview_canvas.bind("<ButtonPress-1>", self._start_preview_drag)
        self.preview_canvas.bind("<B1-Motion>", self._drag_preview)
        self.preview_canvas.bind("<MouseWheel>", self._preview_mousewheel)
        self.preview_info = ttk.Label(center, text="Selecione um termo na lista.", style="Hint.TLabel")
        self.preview_info.grid(row=2, column=0, sticky="w")

        right = ttk.Frame(body, style="Card.TFrame", padding=12)
        right.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        ttk.Label(right, text="Correções", style="PanelTitle.TLabel").pack(anchor="w")
        self.selected_status = ttk.Label(right, text="-", style="Status.TLabel")
        self.selected_status.pack(anchor="w", pady=(6, 12))

        ttk.Label(right, text="Pessoa/foto", style="Editor.TLabel").pack(anchor="w")
        self.person_var = tk.StringVar(value="(Sem foto)")
        self.person_combo = ttk.Combobox(right, textvariable=self.person_var, state="readonly")
        self.person_combo.pack(fill="x", pady=(4, 10))
        self.person_combo.bind("<<ComboboxSelected>>", self.on_person_changed)
        ttk.Button(right, text="Escolher foto visualmente", command=self.open_photo_picker, style="Quiet.TButton").pack(fill="x", pady=(0, 10))

        ttk.Label(right, text="Nome do PDF", style="Editor.TLabel").pack(anchor="w")
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(right, textvariable=self.name_var)
        self.name_entry.pack(fill="x", pady=(4, 10))
        self._attach_text_menu(self.name_entry)

        ttk.Label(right, text="Modelo", style="Editor.TLabel").pack(anchor="w")
        self.model_var = tk.StringVar()
        ttk.Combobox(right, textvariable=self.model_var, state="readonly", values=list(self.models)).pack(fill="x", pady=(4, 12))

        ttk.Button(right, text="Salvar correção", command=self.apply_edit, style="Secondary.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Pré-visualizar em janela", command=self.preview, style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Corrigir recorte do termo", command=self.edit_term_crop, style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Adicionar foto individual", command=self.add_manual_photo, style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Foto compartilhada + recorte", command=self.add_shared_photo_crop, style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Reusar última foto compartilhada", command=lambda: self.add_shared_photo_crop(reuse=True), style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Ajustar recorte", command=self.edit_photo_crop, style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Marcar como sem foto", command=self.mark_no_photo, style="Quiet.TButton").pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Confirmar que não é repetido", command=self.confirm_not_duplicate, style="Quiet.TButton").pack(fill="x")

        self.selected_warning = ttk.Label(right, text="", style="SmallHint.TLabel", wraplength=300)
        self.selected_warning.pack(anchor="w", fill="x", pady=(12, 0))

    def _set_results_visible(self, visible: bool) -> None:
        if visible:
            self.results_hint.pack_forget()
            self.setup_panel.pack_forget()
            self.action_panel.pack_forget()
            if hasattr(self, "video_tool_panel"):
                self.video_tool_panel.pack_forget()
            if not self.review_panel.winfo_manager():
                self.review_panel.pack(fill="both", expand=True)
        else:
            self.review_panel.pack_forget()
            if not self.setup_panel.winfo_manager():
                self.setup_panel.pack(fill="x", pady=(0, 10))
            if not self.action_panel.winfo_manager():
                self.action_panel.pack(fill="x", pady=(0, 10))
            if hasattr(self, "video_tool_panel") and not self.video_tool_panel.winfo_manager():
                self.video_tool_panel.pack(fill="x", pady=(0, 10))
            if not self.results_hint.winfo_manager():
                self.results_hint.pack(fill="x", padx=22, pady=(0, 14))

    def _show_setup(self) -> None:
        self._set_results_visible(False)
        self._update_ready_state(preserve_message=True)

    def _attach_text_menu(self, widget: tk.Widget) -> None:
        menu = tk.Menu(
            self,
            tearoff=0,
            background=PALETTE["surface_alt"],
            foreground=PALETTE["ink"],
            activebackground=PALETTE["accent_soft"],
            activeforeground=PALETTE["ink"],
        )
        menu.add_command(label="Recortar", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Copiar", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Colar", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Selecionar tudo", command=lambda: self._select_entry_text(widget))
        self.text_menus.append(menu)

        def show_menu(event) -> str:
            widget.focus_set()
            menu.tk_popup(event.x_root, event.y_root)
            menu.grab_release()
            return "break"

        for sequence in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            widget.bind(sequence, show_menu)

    def _watch_drive_text(self, widget: tk.Text) -> None:
        for sequence in ("<KeyRelease>", "<FocusOut>", "<<Paste>>", "<<Cut>>"):
            widget.bind(sequence, lambda _event: self._schedule_link_status_refresh(), add="+")

    def _schedule_link_status_refresh(self) -> None:
        if self.link_refresh_after is not None:
            try:
                self.after_cancel(self.link_refresh_after)
            except tk.TclError:
                pass
        self.link_refresh_after = self.after(120, self._refresh_link_statuses)

    def _refresh_link_statuses(self) -> None:
        self.link_refresh_after = None
        if not hasattr(self, "term_status") or not hasattr(self, "photo_status"):
            return
        video_text = self._video_drive_links_text()
        video_links = self._video_drive_links()
        if video_links and not (self.zip_path or self.photo_folder or self.video_folder):
            plural = "link" if len(video_links) == 1 else "links"
            self._set_card_status(self.photo_status, f"{len(video_links)} {plural} do Drive recebido(s) · os vídeos serão lidos ao combinar", True)
        elif video_text and not video_links and not (self.zip_path or self.photo_folder or self.video_folder):
            self._set_card_status(self.photo_status, "Não encontrei um link de pasta do Google Drive com vídeos", False, error=True)
        elif not (self.zip_path or self.photo_folder or self.video_folder):
            self._set_card_status(self.photo_status, "Nenhuma pessoa adicionada — os termos sairão como SEMFOTO", False)

        term_text = self._drive_links_text()
        term_links = self._drive_links()
        if term_links and not self.term_paths:
            plural = "link" if len(term_links) == 1 else "links"
            self._set_card_status(self.term_status, f"{len(term_links)} {plural} do Drive recebidos · os termos serão baixados ao combinar", True)
        elif term_text and not term_links and not self.term_paths:
            self._set_card_status(self.term_status, "Não encontrei um link de pasta do Google Drive", False, error=True)
        elif not self.term_paths:
            self._set_card_status(self.term_status, "Nenhum termo adicionado", False)
        self._update_ready_state()

    @staticmethod
    def _select_entry_text(widget: tk.Widget) -> None:
        try:
            if isinstance(widget, tk.Text):
                widget.tag_add("sel", "1.0", "end-1c")
                widget.mark_set("insert", "end-1c")
            else:
                widget.selection_range(0, "end")
                widget.icursor("end")
        except tk.TclError:
            pass

    def _clear_drive_links(self) -> None:
        if hasattr(self, "term_drive_text"):
            self.term_drive_text.delete("1.0", "end")

    def _clear_video_drive_links(self) -> None:
        if hasattr(self, "video_drive_text"):
            self.video_drive_text.delete("1.0", "end")

    def _drive_links_text(self) -> str:
        if not hasattr(self, "term_drive_text"):
            return ""
        return self.term_drive_text.get("1.0", "end").strip()

    def _drive_links(self) -> list[str]:
        return drive_folder_links_from_text(self._drive_links_text())

    def _video_drive_links_text(self) -> str:
        if not hasattr(self, "video_drive_text"):
            return ""
        return self.video_drive_text.get("1.0", "end").strip()

    def _video_drive_links(self) -> list[str]:
        return drive_folder_links_from_text(self._video_drive_links_text())

    def _invalidate_results(self) -> None:
        if self.results:
            self.results = []
            self.refresh_trees()
        self._set_results_visible(False)

    def _make_tree(self, parent: ttk.Frame) -> ttk.Treeview:
        columns = ("arquivo", "ocr", "pessoa", "modelo", "confianca", "status", "aviso")
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
        widths = {"arquivo": 150, "ocr": 205, "pessoa": 135, "modelo": 115, "confianca": 70, "status": 80, "aviso": 270}
        labels = {"arquivo": "Termo", "ocr": "Leitura automática", "pessoa": "Pessoa/foto", "modelo": "Modelo", "confianca": "Confiança", "status": "Situação", "aviso": "Aviso de repetição"}
        for col in columns:
            tree.heading(col, text=labels[col])
            tree.column(col, width=widths[col], anchor="w")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        tree.bind("<<TreeviewSelect>>", self.on_select)
        return tree

    def choose_zip(self) -> None:
        value = filedialog.askopenfilename(title="ZIP exportado do WhatsApp", filetypes=[("Arquivo ZIP", "*.zip")])
        if value:
            path = Path(value)
            try:
                with zipfile.ZipFile(path) as archive:
                    image_count = sum(Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"} for name in archive.namelist())
            except zipfile.BadZipFile:
                messagebox.showerror("Arquivo inválido", "Esse arquivo não é um ZIP válido do WhatsApp.")
                return
            self.zip_path = path
            self.photo_folder = None
            self.video_folder = None
            self._clear_video_drive_links()
            self._invalidate_results()
            self._set_card_status(self.photo_status, f"WhatsApp recebido: {path.name} · {image_count} imagem(ns)", True)
            self._update_ready_state()

    def choose_whatsapp_videos(self) -> None:
        value = filedialog.askopenfilename(
            title="ZIP exportado do WhatsApp com vídeos",
            filetypes=[("Arquivo ZIP", "*.zip")],
        )
        if not value:
            return
        zip_path = Path(value)
        try:
            with zipfile.ZipFile(zip_path):
                pass
        except zipfile.BadZipFile:
            messagebox.showerror("Arquivo inválido", "Esse arquivo não é um ZIP válido do WhatsApp.")
            return
        destination_value = filedialog.askdirectory(title="Onde salvar os vídeos renomeados pelas legendas")
        if not destination_value:
            return
        try:
            exported = export_whatsapp_videos(
                zip_path,
                self._new_temp_dir("termos_whatsapp_videos_"),
                Path(destination_value),
            )
        except Exception as exc:
            messagebox.showerror("Vídeos do WhatsApp", str(exc))
            return
        self.video_folder = Path(destination_value)
        self.zip_path = None
        self.photo_folder = None
        self._clear_video_drive_links()
        self._invalidate_results()
        self._set_card_status(
            self.photo_status,
            f"WhatsApp recebido: {len(exported)} vídeo(s) renomeado(s) pelas legendas",
            True,
        )
        self._update_ready_state()

    def choose_terms(self) -> None:
        values = filedialog.askopenfilenames(title="Selecione os termos", filetypes=[("Termos", "*.jpg *.jpeg *.png *.pdf"), ("Todos", "*.*")])
        if values:
            self.term_paths = [Path(v) for v in values if Path(v).suffix.lower() in SUPPORTED]
            if not self.term_paths:
                messagebox.showwarning("Nenhum termo", "Selecione imagens JPG/PNG ou arquivos PDF.")
                return
            self._clear_drive_links()
            self._invalidate_results()
            self._set_card_status(self.term_status, f"{len(self.term_paths)} termo(s) selecionado(s) deste Mac", True)
            self._update_ready_state()

    def choose_term_folder(self) -> None:
        value = filedialog.askdirectory(title="Pasta com os termos (inclui Google Drive no Finder)")
        if not value:
            return
        folder = Path(value)
        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED
        )
        if not paths:
            messagebox.showwarning("Pasta sem termos", "Não encontrei JPG, JPEG, PNG ou PDF nessa pasta.")
            return
        self.term_paths = paths
        self._clear_drive_links()
        self._invalidate_results()
        self._set_card_status(self.term_status, f"{len(paths)} termo(s) recebidos da pasta {folder.name}", True)
        self._update_ready_state()

    def choose_output(self) -> None:
        value = filedialog.askdirectory(title="Pasta para os PDFs")
        if value:
            self.output_dir = Path(value)
            self._set_card_status(self.output_status, f"Pasta selecionada: {self.output_dir}", True)
            self._update_ready_state()

    def choose_photo_folder(self) -> None:
        value = filedialog.askdirectory(title="Pasta com as fotos das pessoas")
        if value:
            folder = Path(value)
            image_count = len(people_from_folder(folder))
            if not image_count:
                messagebox.showwarning("Pasta sem fotos", "Não encontrei imagens JPG ou PNG nessa pasta.")
                return
            self.photo_folder = folder
            self.zip_path = None
            self.video_folder = None
            self._clear_video_drive_links()
            self._invalidate_results()
            self._set_card_status(self.photo_status, f"Pasta de fotos recebida: {folder.name} · {image_count} imagem(ns)", True)
            self._update_ready_state()

    def choose_video_folder(self) -> None:
        value = filedialog.askdirectory(title="Pasta com os vídeos das pessoas")
        if value:
            folder = Path(value)
            video_count = len(video_files_from_folder(folder))
            if not video_count:
                messagebox.showwarning("Pasta sem vídeos", "Não encontrei vídeos MP4, MOV, M4V, AVI, MKV ou WEBM nessa pasta.")
                return
            self.video_folder = folder
            self.zip_path = None
            self.photo_folder = None
            self._clear_video_drive_links()
            self._invalidate_results()
            self._set_card_status(self.photo_status, f"Pasta de vídeos recebida: {folder.name} · {video_count} vídeo(s)", True)
            self._update_ready_state()

    def confirm_video_drive(self) -> None:
        links = self._video_drive_links()
        if not links:
            self._set_card_status(self.photo_status, "Cole pelo menos um link de pasta do Google Drive com vídeos", False, error=True)
            messagebox.showwarning("Links do Drive", "Cole um ou mais links de pastas do Google Drive com vídeos, um por linha.")
            return
        self.zip_path = None
        self.photo_folder = None
        self.video_folder = None
        self._invalidate_results()
        plural = "link" if len(links) == 1 else "links"
        self._set_card_status(self.photo_status, f"{len(links)} {plural} do Drive recebido(s) · os vídeos serão lidos ao combinar", True)
        self._update_ready_state()

    def confirm_term_drive(self) -> None:
        links = self._drive_links()
        if not links:
            self._set_card_status(self.term_status, "Cole pelo menos um link de pasta do Google Drive", False, error=True)
            messagebox.showwarning("Links do Drive", "Cole um ou mais links de pastas do Google Drive, um por linha.")
            return
        self.term_paths = []
        self._invalidate_results()
        plural = "link" if len(links) == 1 else "links"
        self._set_card_status(self.term_status, f"{len(links)} {plural} do Drive recebidos · os termos serão baixados ao combinar", True)
        self._update_ready_state()

    def _set_card_status(self, label: tk.Label, text: str, ready: bool, error: bool = False) -> None:
        if error:
            symbol, color = "!", PALETTE["danger"]
        elif ready:
            symbol, color = "✓", PALETTE["success"]
        else:
            symbol, color = "○", PALETTE["muted"]
        label.configure(text=f"{symbol} {text}", fg=color)

    def _has_drive_terms(self) -> bool:
        return bool(self._drive_links())

    def _has_person_source(self) -> bool:
        return bool(self.zip_path or self.photo_folder or self.video_folder or self._video_drive_links())

    def _update_ready_state(self, preserve_message: bool = False) -> None:
        has_terms = bool(self.term_paths) or self._has_drive_terms()
        if has_terms and self.output_dir:
            self.combine_button.state(["!disabled"])
            if not preserve_message:
                photo_text = "Pessoas prontas" if self._has_person_source() else "Sem pessoas: PDFs usarão _SEMFOTO"
                drive_count = len(self._drive_links())
                source_text = "termos do Mac" if self.term_paths else f"termos de {drive_count} link(s) do Google Drive"
                self.process_status.configure(text=f"Pronto para combinar {source_text}. {photo_text}.")
        else:
            self.combine_button.state(["disabled"])
            if not preserve_message:
                missing = []
                if not has_terms:
                    missing.append("os termos")
                if not self.output_dir:
                    missing.append("a pasta de saída")
                self.process_status.configure(text="Falta selecionar " + " e ".join(missing) + ".")

    def _new_temp_dir(self, prefix: str) -> Path:
        holder = tempfile.TemporaryDirectory(prefix=prefix)
        self.temp_dirs.append(holder)
        return Path(holder.name)

    def _person_choices(
        self,
        include_used: bool = False,
        current_person: PersonPhoto | None = None,
        include_unidentified: bool = False,
    ) -> list[tuple[str, PersonPhoto]]:
        choices: list[tuple[str, PersonPhoto]] = []
        used: dict[str, int] = {}
        linked = set(self._linked_photo_names()) if not include_used else set()
        current_key = person_identity(current_person) if current_person is not None else None
        for person in sorted(self.people, key=lambda item: (alpha_key(item.label), alpha_key(item.image_path.name))):
            key = person_identity(person)
            if not include_used and key in linked and key != current_key:
                continue
            identified = person_is_identified(person)
            if not identified and not include_unidentified and key != current_key:
                continue
            base = person.label if identified else f"Sem nome · {person.image_path.name}"
            label = base
            if label in used:
                used[label] += 1
                label = f"{base} · {person.image_path.name}"
                while any(existing == label for existing, _ in choices):
                    used[base] = used.get(base, 1) + 1
                    label = f"{base} · {person.image_path.stem} {used[base]}"
            else:
                used[label] = 1
            choices.append((label, person))
        return choices

    def _label_for_person(self, person: PersonPhoto | None) -> str:
        if person is None:
            return "(Sem foto)"
        return next((key for key, value in self.person_by_label.items() if value is person), person.label)

    def _register_people(self, current_person: PersonPhoto | None = None, include_used: bool = False) -> None:
        self.person_by_label = {}
        labels = ["(Sem foto)"]
        for label, person in self._person_choices(include_used=include_used, current_person=current_person):
            self.person_by_label[label] = person
            labels.append(label)
        self.person_combo.configure(values=labels)

    def _linked_photo_names(self) -> dict[tuple[str, str], list[str]]:
        linked: dict[tuple[str, str], list[str]] = {}
        for result in self.results:
            if result.person is None:
                continue
            linked.setdefault(person_identity(result.person), []).append(result.output_name or result.source.stem)
        return linked

    def _photo_usage_lookup(self) -> dict[tuple[str, str], str]:
        lookup: dict[tuple[str, str], str] = {}
        linked = self._linked_photo_names()
        for person in self.people:
            names = linked.get(person_identity(person), [])
            if not names:
                lookup[person_identity(person)] = "Sem termo"
            elif len(names) == 1:
                lookup[person_identity(person)] = f"Selecionada: {names[0]}"
            else:
                lookup[person_identity(person)] = f"Selecionada em {len(names)} termos"
        return lookup

    def _unused_photo_count(self) -> int:
        linked = self._linked_photo_names()
        return sum(
            1 for person in self.people
            if person_is_identified(person) and person_identity(person) not in linked
        )

    def open_video_audit(self) -> None:
        VideoAuditWindow(self, self.output_dir)

    def analyze(self) -> None:
        if not (self.term_paths or self._has_drive_terms()):
            messagebox.showwarning("Faltam os termos", "No passo 2, selecione as imagens/PDFs ou confirme um link de pasta do Google Drive.")
            return
        if not self.output_dir:
            messagebox.showwarning("Falta a pasta de saída", "No passo 3, escolha onde os PDFs finalizados serão salvos.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Processamento em andamento", "Aguarde a combinação atual terminar.")
            return
        engine = ocr_engine_name()
        if not engine:
            messagebox.showerror(
                "Leitura automática indisponível",
                "O leitor nativo do macOS não foi instalado. Feche e abra novamente esta versão do aplicativo para concluir a atualização.",
            )
            return

        for holder in self.temp_dirs:
            holder.cleanup()
        self.temp_dirs = []
        self.results = []
        self.refresh_trees()
        self._set_results_visible(False)
        self.work_queue = queue.Queue()
        self.combine_button.state(["disabled"])
        self.phase_message = f"Preparando arquivos com {engine}"
        self.phase_started = time.monotonic()
        self.process_status.configure(text=self.phase_message)
        self.progress.stop()
        self.progress.configure(mode="indeterminate", maximum=100, value=0)
        self.progress.start(12)

        inputs = {
            "zip_path": self.zip_path,
            "photo_folder": self.photo_folder,
            "video_folder": self.video_folder,
            "video_drive_urls": self._video_drive_links(),
            "term_paths": list(self.term_paths),
            "drive_urls": self._drive_links(),
        }
        self.worker_thread = threading.Thread(target=self._analyze_worker, args=(inputs,), daemon=True)
        self.worker_thread.start()
        self.after(100, self._poll_worker)

    def _analyze_worker(self, inputs: dict) -> None:
        holders: list[tempfile.TemporaryDirectory] = []

        def temp_dir(prefix: str) -> Path:
            holder = tempfile.TemporaryDirectory(prefix=prefix)
            holders.append(holder)
            return Path(holder.name)

        def emit(kind: str, **payload) -> None:
            self.work_queue.put((kind, payload))

        try:
            people: list[PersonPhoto] = []
            video_roots: list[Path] = []
            if inputs["zip_path"]:
                emit("phase", message="Lendo fotos e legendas do WhatsApp")
                people.extend(parse_whatsapp_export(inputs["zip_path"], temp_dir("termos_whatsapp_")))
            if inputs["photo_folder"]:
                people.extend(people_from_folder(inputs["photo_folder"]))
            video_failures: list[str] = []
            if inputs["video_folder"]:
                video_roots.append(inputs["video_folder"])
            video_drive_urls = list(inputs.get("video_drive_urls") or [])
            if video_drive_urls:
                downloaded_videos_root = temp_dir("termos_video_drive_")
                video_sources: list[Path] = []
                for link_index, drive_url in enumerate(video_drive_urls, start=1):
                    emit("phase", message=f"Baixando vídeos {link_index}/{len(video_drive_urls)} do Google Drive")
                    link_destination = downloaded_videos_root / f"link_{link_index:02d}"
                    try:
                        download_drive_folder(
                            drive_url,
                            link_destination,
                            supported=VIDEO_SUPPORTED,
                            kind_label="vídeos",
                            progress_callback=lambda current, total, name, eta, link_index=link_index, link_total=len(video_drive_urls): emit(
                                "drive_progress",
                                current=current,
                                total=total,
                                name=name,
                                eta=eta,
                                link_index=link_index,
                                link_total=link_total,
                            ),
                        )
                    except Exception as exc:
                        video_failures.append(f"Vídeos link {link_index}: {exc}")
                        continue
                    video_sources.extend(video_files_from_folder(link_destination))
                if not video_sources:
                    detail = "\n".join(video_failures[:5])
                    raise ValueError(
                        "Os links do Google Drive foram acessados, mas nenhum vídeo foi baixado."
                        + (f"\n\nDetalhe:\n{detail}" if detail else "")
                    )
                video_roots.append(downloaded_videos_root)

            if inputs["term_paths"]:
                term_sources = list(inputs["term_paths"])
            else:
                drive_urls = list(inputs.get("drive_urls") or [])
                if not drive_urls:
                    raise ValueError("Cole pelo menos um link de pasta do Google Drive.")
                destination = temp_dir("termos_drive_")
                term_sources = []
                drive_failures = []
                drive_duplicate_terms = 0
                for link_index, drive_url in enumerate(drive_urls, start=1):
                    emit("phase", message=f"Baixando link {link_index}/{len(drive_urls)} do Google Drive")
                    link_destination = destination / f"link_{link_index:02d}"
                    try:
                        _, link_failures, link_duplicates = download_drive_folder(
                            drive_url,
                            link_destination,
                            progress_callback=lambda current, total, name, eta, link_index=link_index, link_total=len(drive_urls): emit(
                                "drive_progress",
                                current=current,
                                total=total,
                                name=name,
                                eta=eta,
                                link_index=link_index,
                                link_total=link_total,
                            ),
                        )
                    except Exception as exc:
                        drive_failures.append(f"Link {link_index}: {exc}")
                        continue
                    drive_duplicate_terms += link_duplicates
                    drive_failures.extend(f"Link {link_index}: {failure}" for failure in link_failures)
                    term_sources.extend(
                        path for path in link_destination.rglob("*")
                        if path.is_file() and path.suffix.lower() in SUPPORTED
                    )
                if not term_sources:
                    detail = "\n".join(drive_failures[:5])
                    raise ValueError(
                        "Os links do Google Drive foram acessados, mas nenhum termo JPG, JPEG, PNG ou PDF foi baixado."
                        + (f"\n\nDetalhe:\n{detail}" if detail else "")
                    )
                emit("terms_found", count=len(term_sources), failures=drive_failures)
            if inputs["term_paths"]:
                drive_failures = []
                drive_duplicate_terms = 0

            emit("phase", message="Removendo termos duplicados")
            term_sources, duplicate_terms = unique_term_sources(term_sources)
            duplicate_terms += drive_duplicate_terms
            term_sources = sorted(term_sources, key=lambda path: alpha_key(path.name))
            if not term_sources:
                raise ValueError("Depois de remover duplicatas, não sobrou nenhum termo válido para processar.")

            results: list[TermResult] = []
            total = len(term_sources)
            started = time.monotonic()
            worker_count = analysis_worker_count(total)
            emit("progress", current=0, total=total, eta=None, message=f"Iniciando leitura automática com {worker_count} frente(s)")
            ordered_results: list[TermResult | None] = [None] * total
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(analyze_term_source, source, self.models, []): (idx, source)
                    for idx, source in enumerate(term_sources)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    idx, source = futures[future]
                    ordered_results[idx] = future.result()
                    elapsed = time.monotonic() - started
                    eta = (elapsed / completed) * (total - completed) if completed else None
                    emit(
                        "progress",
                        current=completed,
                        total=total,
                        eta=eta,
                        message=f"Lendo termos: {source.name}",
                    )
            results = [result for result in ordered_results if result is not None]
            roster_names = _term_roster_names(results)
            if video_roots:
                emit("phase", message="Lendo os nomes falados com apoio da lista do lote")
                video_thumbnails = temp_dir("termos_video_fotos_")
                for root in video_roots:
                    video_people, root_failures = people_from_video_folder(
                        root,
                        video_thumbnails,
                        progress_callback=lambda current, total, name, eta: emit(
                            "video_progress",
                            current=current,
                            total=total,
                            name=name,
                            eta=eta,
                        ),
                        roster_names=roster_names,
                    )
                    people.extend(video_people)
                    video_failures.extend(root_failures)

            seen = set()
            unique_people: list[PersonPhoto] = []
            for person in people:
                key = person_identity(person)
                if key not in seen:
                    seen.add(key)
                    unique_people.append(person)

            emit("phase", message="Reservando os casamentos mais claros do lote")
            assign_people_to_results(results, unique_people)

            # Primeiro retiramos da disputa os pares fortes. Só então a leitura
            # guiada consulta as pessoas que ainda estão sem termo; essa ordem é
            # decisiva em lotes pequenos com nomes próximos, como Higor/Hugo.
            recovered_video_names = relabel_unidentified_people_from_terms(unique_people, results)
            assign_people_to_results(results, unique_people)
            remaining_people = people_without_assigned_term(results, unique_people)

            emit("phase", message="Relendo apenas os nomes ainda sem par")
            roster_ocr_recovered = enhance_weak_terms_with_roster(results, remaining_people)
            recovered_video_names += relabel_unidentified_people_from_terms(unique_people, results)
            assign_people_to_results(results, unique_people)
            emit(
                "done",
                results=results,
                people=unique_people,
                holders=holders,
                drive_failures=drive_failures,
                video_failures=video_failures,
                duplicate_terms=duplicate_terms,
                recovered_video_names=recovered_video_names,
                roster_ocr_recovered=roster_ocr_recovered,
            )
        except Exception as exc:
            for holder in holders:
                holder.cleanup()
            emit("error", message=str(exc))

    def _poll_worker(self) -> None:
        terminal = False
        while True:
            try:
                kind, payload = self.work_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "phase":
                self.phase_message = payload["message"]
                self.phase_started = time.monotonic()
                self.progress.stop()
                self.progress.configure(mode="indeterminate", maximum=100, value=0)
                self.progress.start(12)
            elif kind == "drive_progress":
                current, total = payload["current"], max(1, payload["total"])
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=total, value=current)
                percent = round(current * 100 / total)
                eta_text = f"aprox. {self._format_duration(payload['eta'])} restantes"
                link_prefix = ""
                if payload.get("link_total", 1) > 1:
                    link_prefix = f"Link {payload['link_index']}/{payload['link_total']} · "
                self.process_status.configure(
                    text=f"{link_prefix}Baixando do Drive: {payload['name']} · {current}/{total} ({percent}%) · {eta_text}"
                )
            elif kind == "video_progress":
                current, total = payload["current"], max(1, payload["total"])
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=total, value=current)
                percent = round(current * 100 / total)
                eta = payload.get("eta")
                eta_text = "calculando tempo restante" if eta is None else f"aprox. {self._format_duration(eta)} restantes"
                self.process_status.configure(text=f"Lendo vídeo: {payload['name']} · {current}/{total} ({percent}%) · {eta_text}")
            elif kind == "terms_found":
                self._set_card_status(self.term_status, f"Drive recebido: {payload['count']} termo(s) encontrado(s)", True)
            elif kind == "progress":
                current, total = payload["current"], max(1, payload["total"])
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=total, value=current)
                percent = round(current * 100 / total)
                eta = payload.get("eta")
                eta_text = "calculando tempo restante" if eta is None else f"aprox. {self._format_duration(eta)} restantes"
                self.process_status.configure(text=f"{payload['message']} · {current}/{total} ({percent}%) · {eta_text}")
                self.phase_message = ""
            elif kind == "done":
                terminal = True
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=1, value=1)
                self.temp_dirs = payload["holders"]
                self.people = payload["people"]
                self.results = payload["results"]
                self._register_people()
                self._set_results_visible(True)
                self._detect_duplicates()
                missing = sum(result.person is None for result in self.results)
                review = sum(result.status == "REVISAR" for result in self.results)
                duplicates = sum(bool(result.duplicate_note and not result.duplicate_confirmed) for result in self.results)
                unused_photos = self._unused_photo_count()
                self.current_filter = "review" if review else ("missing" if missing else "all")
                self.refresh_trees()
                failures = payload.get("drive_failures", [])
                duplicate_terms = payload.get("duplicate_terms", 0)
                warning_parts = []
                if duplicate_terms:
                    warning_parts.append(f"{duplicate_terms} duplicata(s) ignorada(s)")
                if failures:
                    warning_parts.append(f"{len(failures)} arquivo(s) do Drive com erro")
                video_failures = payload.get("video_failures", [])
                if video_failures:
                    warning_parts.append(f"{len(video_failures)} vídeo(s) com erro")
                recovered_video_names = int(payload.get("recovered_video_names", 0) or 0)
                if recovered_video_names:
                    warning_parts.append(f"{recovered_video_names} nome(s) de vídeo recuperado(s)")
                roster_ocr_recovered = int(payload.get("roster_ocr_recovered", 0) or 0)
                if roster_ocr_recovered:
                    warning_parts.append(f"{roster_ocr_recovered} termo(s) relido(s) com nomes do lote")
                if unused_photos:
                    warning_parts.append(f"{unused_photos} pessoa(s) sem termo")
                warning = " · " + " · ".join(warning_parts) if warning_parts else ""
                self.process_status.configure(text=f"Concluído: {len(self.results)} termos · {review} para revisar · {missing} sem foto · {duplicates} possíveis repetidos{warning}")
                self._update_ready_state(preserve_message=True)
                all_failures = list(failures) + list(video_failures)
                if all_failures:
                    messagebox.showwarning(
                        "Alguns arquivos não foram processados",
                        "Os demais termos foram processados. Arquivos ignorados:\n\n" + "\n".join(all_failures[:8]),
                    )
            elif kind == "error":
                terminal = True
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=1, value=0)
                self._set_results_visible(False)
                self.process_status.configure(text="Não foi possível concluir. Confira a mensagem e tente novamente.")
                self._update_ready_state(preserve_message=True)
                messagebox.showerror("Não foi possível combinar", payload["message"])

        if self.phase_message and str(self.progress.cget("mode")) == "indeterminate":
            elapsed = time.monotonic() - self.phase_started
            self.process_status.configure(text=f"{self.phase_message} · {self._format_duration(elapsed)} decorridos")
        if not terminal and self.worker_thread and self.worker_thread.is_alive():
            self.after(200, self._poll_worker)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes}min {seconds:02d}s" if minutes else f"{seconds}s"

    def _insert_result(self, tree: ttk.Treeview, idx: int, result: TermResult) -> None:
        ocr_line = " / ".join(line.strip() for line in result.ocr_text.splitlines()[:4]) or "OCR indisponível"
        person = result.person.label if result.person else "-"
        warning = result.duplicate_note if not result.duplicate_confirmed else "Confirmado: não é repetido"
        tree.insert("", "end", iid=str(idx), values=(result.source.name, ocr_line[:100], person, result.model, f"{result.confidence:.0%}", result.status, warning or "-"))

    def _status_tag(self, result: TermResult) -> str:
        if result.person is None:
            return "missing"
        if result.status == "REVISAR":
            return "review"
        if result.duplicate_note and not result.duplicate_confirmed:
            return "duplicates"
        return "ok"

    def _matches_filter(self, result: TermResult) -> bool:
        if self.current_filter == "all":
            return True
        if self.current_filter == "ok":
            return result.status == "OK"
        if self.current_filter == "review":
            return result.status == "REVISAR"
        if self.current_filter == "missing":
            return result.person is None
        if self.current_filter == "duplicates":
            return bool(result.duplicate_note and not result.duplicate_confirmed)
        return True

    def set_filter(self, key: str) -> None:
        self.current_filter = key
        self.refresh_trees()

    def _display_indices(self) -> list[int]:
        indices = [
            idx for idx, result in enumerate(self.results)
            if self._matches_filter(result)
        ]
        return sorted(
            indices,
            key=lambda idx: (
                alpha_key(self.results[idx].output_name or self.results[idx].source.stem),
                alpha_key(self.results[idx].source.name),
                idx,
            ),
        )

    def _update_review_summary(self) -> None:
        total = len(self.results)
        review = sum(result.status == "REVISAR" for result in self.results)
        missing = sum(result.person is None for result in self.results)
        duplicates = sum(bool(result.duplicate_note and not result.duplicate_confirmed) for result in self.results)
        ok = sum(result.status == "OK" for result in self.results)
        unused_photos = self._unused_photo_count()
        summary = f"{total} termos · {ok} OK · {review} revisar · {missing} sem foto · {duplicates} repetidos · {unused_photos} pessoas sem termo"
        self.review_summary.configure(text=summary)
        pending = review + missing
        self.export_button.configure(text="Exportar PDFs" if pending == 0 else f"Exportar PDFs ({pending} pendências)")
        if hasattr(self, "photo_audit_button"):
            total_photos = len(self.people)
            self.photo_audit_button.configure(
                text=f"Pessoas ({unused_photos} sem termo)" if total_photos else "Pessoas"
            )
        if hasattr(self, "review_tree"):
            visible = len(self.review_tree.get_children())
            self.review_list_hint.configure(text=f"{visible} de {total} termos neste filtro")
        for key, button in getattr(self, "filter_buttons", {}).items():
            if key == self.current_filter:
                button.state(["disabled"])
            else:
                button.state(["!disabled"])

    def _detect_duplicates(self) -> None:
        names = ["" if result.duplicate_confirmed else result.output_name for result in self.results]
        notes = duplicate_notes(names, load_history())
        for idx, result in enumerate(self.results):
            result.duplicate_note = notes.get(idx, "")

    def refresh_trees(self, preferred_idx: int | None = None) -> None:
        self._detect_duplicates()
        if not hasattr(self, "review_tree"):
            return
        selected = preferred_idx if preferred_idx is not None else self.selected_index()
        self.review_tree.delete(*self.review_tree.get_children())
        for idx in self._display_indices():
            result = self.results[idx]
            self.review_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(result.output_name or result.source.stem, result.status, f"{result.confidence:.0%}"),
                tags=(self._status_tag(result),),
            )
        self._update_review_summary()
        children = self.review_tree.get_children()
        target = None
        if selected is not None and self.review_tree.exists(str(selected)):
            target = str(selected)
        elif children:
            target = children[0]
        if target:
            self.review_tree.selection_set(target)
            self.review_tree.focus(target)
            self.review_tree.see(target)
            self.on_select()
        else:
            self._clear_selected_details()

    def _refresh_and_advance(self, idx: int, old_order: list[int]) -> None:
        self._detect_duplicates()
        new_order = self._display_indices()
        if not new_order:
            self.refresh_trees()
            return
        try:
            old_position = old_order.index(idx)
        except ValueError:
            old_position = 0
        if idx in new_order:
            new_position = min(new_order.index(idx) + 1, len(new_order) - 1)
        else:
            new_position = min(old_position, len(new_order) - 1)
        self.refresh_trees(preferred_idx=new_order[new_position])

    def _active_tree(self) -> ttk.Treeview:
        return self.review_tree

    def selected_index(self) -> int | None:
        if not hasattr(self, "review_tree"):
            return None
        tree = self._active_tree()
        selection = tree.selection()
        return int(selection[0]) if selection else None

    def _select_in_all(self, idx: int) -> None:
        tree = self.review_tree
        if not tree.exists(str(idx)):
            self.current_filter = "all"
            self.refresh_trees()
        if tree.exists(str(idx)):
            tree.selection_set(str(idx))
            tree.focus(str(idx))
            tree.see(str(idx))
            self.on_select()

    def _clear_selected_details(self) -> None:
        self.person_var.set("(Sem foto)")
        self.name_var.set("")
        self.model_var.set("")
        self.selected_status.configure(text="-")
        self.selected_warning.configure(text="")
        self.preview_info.configure(text="Selecione um termo na lista.")
        self.preview_canvas.delete("all")

    def adjust_preview_zoom(self, factor: float) -> None:
        self.preview_zoom = max(0.35, min(4.5, self.preview_zoom * factor))
        self._render_selected_preview()

    def reset_preview_view(self) -> None:
        self.preview_zoom = 1.0
        self.preview_offset_x = 0
        self.preview_offset_y = 0
        self._render_selected_preview()

    def _start_preview_drag(self, event) -> None:
        self.preview_drag_origin = (event.x, event.y, self.preview_offset_x, self.preview_offset_y)

    def _drag_preview(self, event) -> None:
        if not self.preview_drag_origin:
            return
        x0, y0, ox, oy = self.preview_drag_origin
        self.preview_offset_x = ox + event.x - x0
        self.preview_offset_y = oy + event.y - y0
        self._render_selected_preview()

    def _preview_mousewheel(self, event) -> str:
        self.adjust_preview_zoom(1.12 if event.delta > 0 else 0.89)
        return "break"

    def _render_selected_preview(self) -> None:
        if not hasattr(self, "preview_canvas"):
            return
        self.preview_canvas.delete("all")
        idx = self.selected_index()
        width = max(self.preview_canvas.winfo_width(), 320)
        height = max(self.preview_canvas.winfo_height(), 420)
        if hasattr(self, "preview_zoom_label"):
            self.preview_zoom_label.configure(text=f"{round(self.preview_zoom * 100)}%")
        if idx is None or not (0 <= idx < len(self.results)):
            self.preview_canvas.create_text(
                width // 2,
                height // 2,
                text="Selecione um termo",
                fill=PALETTE["muted"],
                font=("Helvetica", 14, "bold"),
            )
            return
        result = self.results[idx]
        try:
            page = self.compose(result)
        except Exception:
            page = result.page.copy()
        fit_w = max(120, width - 28)
        fit_h = max(160, height - 28)
        scale = min(fit_w / page.width, fit_h / page.height) * self.preview_zoom
        preview_size = (
            max(1, round(page.width * scale)),
            max(1, round(page.height * scale)),
        )
        preview = page.resize(preview_size, Image.Resampling.LANCZOS)
        self.review_preview_image = ImageTk.PhotoImage(preview)
        self.preview_canvas.create_image(
            width // 2 + self.preview_offset_x,
            height // 2 + self.preview_offset_y,
            image=self.review_preview_image,
            anchor="center",
        )

    def on_person_changed(self, _event=None) -> None:
        label = self.person_var.get()
        person = self.person_by_label.get(label)
        if person is not None:
            if person_is_identified(person):
                self.name_var.set(person.label or label_from_photo_path(person.image_path))
            elif not self.name_var.get().strip():
                idx = self.selected_index()
                if idx is not None and 0 <= idx < len(self.results):
                    self.name_var.set(result_output_name(self.results[idx], people=self.people))
        elif label == "(Sem foto)" and not self.name_var.get().strip():
            idx = self.selected_index()
            if idx is not None and 0 <= idx < len(self.results):
                self.name_var.set(self.results[idx].source.stem)

    def open_photo_picker(self) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo que receberá a foto.")
            return
        choices = self._person_choices(current_person=self.results[idx].person, include_unidentified=True)
        if not choices:
            if self.people:
                messagebox.showinfo("Sem pessoas disponíveis", "Todas as pessoas do lote já estão ligadas a outros termos.")
            else:
                messagebox.showinfo("Sem pessoas", "Adicione um ZIP do WhatsApp, uma pasta de fotos, vídeos ou uma foto manual primeiro.")
            return

        def pick(label: str, person: PersonPhoto) -> None:
            self._register_people(current_person=person)
            self.person_var.set(self._label_for_person(person) or label)
            if person_is_identified(person):
                result = self.results[idx]
                self.name_var.set(
                    result_output_name(
                        result,
                        person,
                        self.people,
                        _person_token_counts(matchable_people(self.people)),
                    )
                )
            self.apply_edit()

        PhotoPicker(
            self,
            choices,
            self.person_var.get(),
            pick,
            usage_lookup=self._photo_usage_lookup(),
            title="Escolher pessoa",
            action_text="Usar no termo selecionado",
        )

    def open_photo_audit(self) -> None:
        choices = self._person_choices(include_used=True, include_unidentified=True)
        if not choices:
            messagebox.showinfo("Sem pessoas", "Nenhuma pessoa foi adicionada neste lote.")
            return

        def pick(label: str, person: PersonPhoto) -> None:
            if self.selected_index() is None:
                messagebox.showwarning("Selecione um termo", "Selecione um termo na lista antes de aplicar uma foto.")
                return
            self._register_people(current_person=person)
            self.person_var.set(self._label_for_person(person) or label)
            if person_is_identified(person):
                idx = self.selected_index()
                result = self.results[idx]
                self.name_var.set(
                    result_output_name(
                        result,
                        person,
                        self.people,
                        _person_token_counts(matchable_people(self.people)),
                    )
                )
            self.apply_edit()

        PhotoPicker(
            self,
            choices,
            self.person_var.get(),
            pick,
            usage_lookup=self._photo_usage_lookup(),
            title="Pessoas do lote",
            action_text="Usar no termo selecionado",
        )

    def on_select(self, _event=None) -> None:
        idx = self.selected_index()
        if idx is None:
            self._clear_selected_details()
            return
        result = self.results[idx]
        self.preview_offset_x = 0
        self.preview_offset_y = 0
        self._register_people(current_person=result.person)
        self.person_var.set(self._label_for_person(result.person))
        self.name_var.set(result.output_name)
        self.model_var.set(result.model)
        self.selected_status.configure(text=f"{result.status} · confiança {result.confidence:.0%}")
        warning = result.duplicate_note if result.duplicate_note and not result.duplicate_confirmed else ""
        if result.person is None:
            warning = (warning + "\n" if warning else "") + "Será exportado com o sufixo _SEMFOTO."
        elif result.manual_crop:
            warning = (warning + "\n" if warning else "") + f"Recorte manual aplicado em {result.person.image_path.name}."
        self.selected_warning.configure(text=warning)
        source = result.source.name
        photo = result.person.image_path.name if result.person else "sem foto"
        self.preview_info.configure(text=f"{source} · {photo}")
        self._render_selected_preview()

    def apply_edit(self, silent: bool = False) -> int | None:
        idx = self.selected_index()
        if idx is None:
            if not silent:
                messagebox.showwarning("Selecione uma linha", "Clique primeiro no termo que deseja corrigir.")
            return None
        old_order = self._display_indices()
        result = self.results[idx]
        old_person = result.person
        old_name = normalize(result.output_name)
        label = self.person_var.get()
        result.person = None if label == "(Sem foto)" else self.person_by_label.get(label)
        result.output_name = self.name_var.get().strip() or (result.person.label if result.person else result.source.stem)
        if normalize(result.output_name) != old_name:
            result.duplicate_confirmed = False
        if self.model_var.get() in self.models:
            result.model = self.model_var.get()
        if result.person is not old_person:
            result.manual_crop = False
            result.photo_zoom = 1.0
            result.photo_offset_x = result.photo_offset_y = 0.0
        if result.person is None:
            result.confidence = 0.0
        elif result.person is not old_person or not silent:
            result.confidence = 1.0
        if silent:
            self.refresh_trees(preferred_idx=idx)
        else:
            self._refresh_and_advance(idx, old_order)
        return idx

    def confirm_not_duplicate(self) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo que deseja confirmar.")
            return
        result = self.results[idx]
        if not result.duplicate_note:
            messagebox.showinfo("Sem aviso", "Este termo não está marcado como possível repetido.")
            return
        old_order = self._display_indices()
        result.duplicate_confirmed = True
        self._refresh_and_advance(idx, old_order)

    def add_manual_photo(self) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo que receberá a foto.")
            return
        value = filedialog.askopenfilename(title="Escolher foto", filetypes=[("Imagens", "*.jpg *.jpeg *.png"), ("Todos", "*.*")])
        if not value:
            return
        path = Path(value)
        old_order = self._display_indices()
        label = label_from_photo_path(path)
        person = PersonPhoto(label=label, caption="Foto adicionada manualmente", image_path=path)
        self.people.append(person)
        self._register_people(current_person=person)
        result = self.results[idx]
        result.person = person
        result.output_name = label
        result.confidence = 1.0
        result.manual_crop = False
        result.photo_zoom = 1.0
        result.photo_offset_x = result.photo_offset_y = 0.0
        self._refresh_and_advance(idx, old_order)

    def add_shared_photo_crop(self, reuse: bool = False) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo que receberá o recorte.")
            return
        old_order = self._display_indices()
        if reuse and self.shared_photo_path:
            path = self.shared_photo_path
        else:
            value = filedialog.askopenfilename(
                title="Foto com uma ou mais pessoas",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png"), ("Todos", "*.*")],
            )
            if not value:
                return
            path = Path(value)
            self.shared_photo_path = path
        result = self.results[idx]
        label = self.name_var.get().strip() or result.output_name or re.sub(r"[_-]+", " ", path.stem).strip()
        person = PersonPhoto(label=label, caption=f"Foto compartilhada: {path.name}", image_path=path)
        self.people.append(person)
        self._register_people(current_person=person)
        result.person = person
        result.output_name = label
        result.confidence = 1.0
        result.manual_crop = False
        result.photo_zoom = 1.0
        result.photo_offset_x = result.photo_offset_y = 0.0
        PhotoEditor(self, result, safe_photo_rect(result.page, self.models[result.model]["foto"]), lambda: self._refresh_and_advance(idx, old_order))

    def mark_no_photo(self) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo.")
            return
        old_order = self._display_indices()
        result = self.results[idx]
        result.person = None
        result.confidence = 0.0
        result.manual_crop = False
        result.output_name = self.name_var.get().strip() or result.output_name or result.source.stem
        self._refresh_and_advance(idx, old_order)

    def edit_photo_crop(self) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo.")
            return
        result = self.results[idx]
        if result.person is None:
            messagebox.showwarning("Sem foto", "Adicione ou selecione uma foto antes de ajustar o enquadramento.")
            return
        old_order = self._display_indices()
        PhotoEditor(self, result, safe_photo_rect(result.page, self.models[result.model]["foto"]), lambda: self._refresh_and_advance(idx, old_order))

    def edit_term_crop(self) -> None:
        idx = self.selected_index()
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Selecione primeiro o termo.")
            return
        result = self.results[idx]

        def save_crop() -> None:
            try:
                source_name = source_name_candidate(result.source)
                focused = focused_name_ocr(
                    result.page,
                    [source_name] if source_name else [],
                    max_regions=MAX_FOCUSED_NAME_REGIONS_WITHOUT_HINT,
                )
                result.ocr_text = focused if focused.strip() else (f"NOME_CANDIDATO\n{source_name}" if source_name else "")
                person, confidence = suggest_person_for_result(self.people, result)
                result.person = person
                result.confidence = confidence
                result.output_name = result_output_name(
                    result,
                    person,
                    self.people,
                    _person_token_counts(self.people),
                )
            except Exception as exc:
                messagebox.showwarning(
                    "Recorte salvo",
                    "O recorte foi salvo, mas não consegui refazer a leitura automática deste termo.\n\n"
                    f"Detalhe: {exc}",
                )
            self.refresh_trees(preferred_idx=idx)
            self._render_selected_preview()

        TermCropEditor(self, result, save_crop)

    def compose(self, result: TermResult) -> Image.Image:
        if result.person is None:
            return result.page.copy()
        return place_photo(
            result.page,
            result.person.image_path,
            self.models[result.model]["foto"],
            result.manual_crop,
            result.photo_zoom,
            result.photo_offset_x,
            result.photo_offset_y,
        )

    def preview(self) -> None:
        idx = self.apply_edit(silent=True)
        if idx is None:
            messagebox.showwarning("Selecione uma linha", "Clique primeiro em um termo.")
            return
        try:
            page = self.compose(self.results[idx])
        except Exception as exc:
            messagebox.showerror("Pré-visualização", str(exc))
            return
        preview = page.copy()
        preview.thumbnail((700, 780), Image.Resampling.LANCZOS)
        window = tk.Toplevel(self)
        suffix = " - SEM FOTO" if self.results[idx].person is None else ""
        window.title(f"Pré-visualização - {self.results[idx].output_name}{suffix}")
        window.configure(background=PALETTE["bg"])
        tk_image = ImageTk.PhotoImage(preview)
        label = ttk.Label(window, image=tk_image)
        label.image = tk_image
        label.pack(padx=10, pady=10)

    def generate(self) -> None:
        if not self.results or not self.output_dir:
            messagebox.showwarning("Nada para gerar", "Analise os termos primeiro.")
            return
        if any(result.status == "REVISAR" for result in self.results):
            if not messagebox.askyesno("Há linhas para revisar", "Algumas associações têm baixa confiança. Deseja gerar mesmo assim?"):
                return
        generated = []
        no_photo_count = 0
        try:
            self.config(cursor="watch")
            self.update_idletasks()
            used: set[str] = set()
            for result in self.results:
                base = safe_filename(result.output_name)
                name = base
                counter = 2
                while normalize(name) in used:
                    name = f"{base} {counter}"
                    counter += 1
                used.add(normalize(name))
                suffix = "_SEMFOTO" if result.person is None else ""
                no_photo_count += int(result.person is None)
                output = self.output_dir / f"{name} - Termo de Autorização{suffix}.pdf"
                save_pdf(self.compose(result), output)
                generated.append(output)
            record_history(self.results)
            messagebox.showinfo("Concluído", f"{len(generated)} PDF(s) gerado(s), incluindo {no_photo_count} sem foto.\n\nPasta:\n{self.output_dir}")
            try:
                subprocess.run(["open", str(self.output_dir)], check=False)
            except OSError:
                pass
        except Exception as exc:
            messagebox.showerror("Erro ao gerar", str(exc))
        finally:
            self.config(cursor="")

    def _on_close(self) -> None:
        for holder in self.temp_dirs:
            holder.cleanup()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
