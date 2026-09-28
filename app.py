import asyncio
import os
import json
import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date
from enum import Enum
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.background import BackgroundTask

from transcriber import (
    Cancelled,
    CONTEXT_OPTION_PROMPTS,
    OUTPUT_PROFILES,
    REGISTER_PROMPTS,
    Segment,
    SPEECH_TYPE_PROMPTS,
    SUPPORTED_LANGUAGES,
    Transcriber,
    WATERMARK_DURATION,
    WATERMARK_GAP,
    WATERMARK_LEAD_SEC,
    WATERMARK_TEXT,
    WRAP_WIDTH,
    _build_initial_prompt,
    to_srt,
    to_vtt,
)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "~/whisper-service/output")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "~/whisper-service/uploads")).expanduser()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Extensions de fichier acceptées pour un upload direct (mp4 ou wav, cf. demande).
ALLOWED_UPLOAD_SUFFIXES = {".mp4", ".wav"}

# Nom des fichiers de sortie : Titre_Transcript_langue_KTT_AI_yyyymmdd.{srt,vtt}
TITLE_MAX_LEN = 20
_TITLE_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_title(raw: str) -> str:
    """Dérive un titre court et sûr pour un nom de fichier à partir d'une URL ou
    d'un nom de fichier uploadé (dernier segment, sans extension)."""
    stem = Path(raw.split("?", 1)[0]).stem
    cleaned = _TITLE_UNSAFE_RE.sub("_", stem).strip("_")
    return (cleaned or "Video")[:TITLE_MAX_LEN]

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("Variable d'environnement API_KEY manquante")

_api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(_api_key_header)) -> None:
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Clé API invalide")


class Status(str, Enum):
    queued = "queued"
    converting = "converting"
    transcribing = "transcribing"
    done = "done"
    error = "error"
    cancelling = "cancelling"
    cancelled = "cancelled"


class Job(BaseModel):
    id: str
    source: str
    title: str
    status: Status = Status.queued
    progress: float = 0.0
    srt_path: str | None = None
    vtt_path: str | None = None
    qc_path: str | None = None        # log QC (JSON) : cps + warnings par cue
    error: str | None = None
    output_profile: str | None = None  # None = profil par défaut (42 car/ligne)
    language: str | None = None            # détectée par Whisper, connue une fois le job "done"
    output_basename: str | None = None     # Titre_Transcript_langue_KTT_AI_yyyymmdd (sans extension)
    speech_type: str | None = None         # None = défaut "group" (cf. SPEECH_TYPE_PROMPTS)
    context_options: list[str] = []        # sous-ensemble de CONTEXT_OPTION_PROMPTS
    register: str | None = None            # None = défaut "standard" (cf. REGISTER_PROMPTS)
    forced_language: str | None = None     # None = détection auto (comportement historique)
    proper_nouns: list[str] = []           # glossaire optionnel (prompt + normalisation post-traitement)
    initial_prompt: str | None = None      # prompt Whisper résolu, pour vérification/QA


# État partagé (en mémoire pour démarrer)
JOBS: dict[str, Job] = {}
CANCEL_EVENTS: dict[str, threading.Event] = {}
# Fichier uploadé (si la source du job est un upload direct, pas une URL) à
# supprimer une fois le job terminé — même logique de nettoyage que les sorties.
UPLOAD_PATHS: dict[str, Path] = {}
# Largeur de ligne résolue pour chaque job (cf. output_profile) — gardée à part
# du modèle Job pour ne pas exposer la valeur numérique interne au client.
CHAR_LIMITS: dict[str, int] = {}
# initial_prompt Whisper résolu pour chaque job (cf. _build_initial_prompt) —
# transmis au worker séparément, comme CHAR_LIMITS.
INITIAL_PROMPTS: dict[str, str] = {}
QUEUE: asyncio.Queue[str] = asyncio.Queue()
STATE: dict = {}


async def worker():
    """Consomme la file, une transcription à la fois."""
    transcriber: Transcriber = STATE["transcriber"]
    loop = asyncio.get_running_loop()
    while True:
        job_id = await QUEUE.get()
        job = JOBS[job_id]
        upload_path = UPLOAD_PATHS.pop(job_id, None)
        if job.status == Status.cancelled:
            # Annulé pendant qu'il attendait encore en file.
            CHAR_LIMITS.pop(job_id, None)
            INITIAL_PROMPTS.pop(job_id, None)
            QUEUE.task_done()
            if upload_path:
                upload_path.unlink(missing_ok=True)
            continue
        job.status = Status.converting
        job.progress = 0.0
        cancel_event = CANCEL_EVENTS[job_id]
        try:
            # run_in_executor : sort le travail bloquant de la boucle asyncio.
            # Les callbacks sont appelés depuis le thread executor ; l'affectation
            # d'un attribut simple sur l'objet Job est atomique (GIL), pas besoin de lock.
            def on_phase(phase: str) -> None:
                # Si une annulation est déjà demandée, ne pas écraser "cancelling"
                # avec la phase suivante (ex : conversion finie juste avant le kill).
                if not cancel_event.is_set():
                    job.status = Status(phase)
                    job.progress = 0.0

            def on_progress(p: float) -> None:
                job.progress = p

            char_limit = CHAR_LIMITS.pop(job_id, WRAP_WIDTH)
            initial_prompt = INITIAL_PROMPTS.pop(job_id, None)
            segments, language, duration = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe(
                    job.source,
                    on_phase=on_phase,
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                    char_limit=char_limit,
                    initial_prompt=initial_prompt,
                    language=job.forced_language,
                    proper_nouns=job.proper_nouns,
                ),
            )
            if not segments:
                raise RuntimeError("Aucune parole détectée dans le média")
            # Ancré sur la durée réelle du média (pas sur la fin de la
            # dernière parole détectée) : une vidéo peut continuer bien après
            # le dernier mot transcrit (musique, silence, générique...), le
            # filigrane doit rester proche de la vraie fin du fichier.
            watermark_start = max(
                duration - WATERMARK_LEAD_SEC,
                segments[-1].end + WATERMARK_GAP,
            )
            segments = segments + [
                Segment(watermark_start, watermark_start + WATERMARK_DURATION, WATERMARK_TEXT)
            ]
            job.language = language
            job.output_basename = (
                f"{job.title}_Transcript_{language}_KTT_AI_{date.today():%Y%m%d}"
            )
            srt_path = OUTPUT_DIR / f"{job_id}.srt"
            vtt_path = OUTPUT_DIR / f"{job_id}.vtt"
            qc_path = OUTPUT_DIR / f"{job_id}.qc.json"
            srt_path.write_text(to_srt(segments), encoding="utf-8")
            vtt_path.write_text(to_vtt(segments), encoding="utf-8")
            qc_path.write_text(
                json.dumps(
                    {
                        "cues": [
                            {
                                "index": i,
                                "start": round(seg.start, 3),
                                "end": round(seg.end, 3),
                                "text": seg.text,
                                "cps": round(
                                    len(seg.text.replace("\n", " ").strip())
                                    / max(seg.end - seg.start, 1e-6),
                                    1,
                                ),
                                "warnings": seg.warnings,
                            }
                            for i, seg in enumerate(segments)
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            job.srt_path = str(srt_path)
            job.vtt_path = str(vtt_path)
            job.qc_path = str(qc_path)
            job.progress = 1.0
            job.status = Status.done
        except Cancelled:
            job.status = Status.cancelled
        except Exception as exc:  # noqa: BLE001
            job.status = Status.error
            job.error = str(exc)
        finally:
            QUEUE.task_done()
            if upload_path:
                upload_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle au démarrage (pas par requête).
    STATE["transcriber"] = Transcriber()
    task = asyncio.create_task(worker())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, title="Service de sous-titrage")


@app.post("/jobs", response_model=Job, dependencies=[Depends(require_api_key)])
async def create_job(
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
    output_profile: str | None = Form(None),
    speech_type: str | None = Form(None),
    context_options: list[str] = Form([]),
    register: str | None = Form(None),
    language: str | None = Form(None),
    proper_nouns: list[str] = Form([]),
):
    """Crée un job à partir d'une URL (mp4/wav) OU d'un fichier uploadé (mp4/wav) — pas les deux.

    output_profile (optionnel) fixe les caractères/ligne selon le support de sortie
    (normes ATAA) : "tv" (36), "video" (39), "streaming" (42), "cinema" (40).
    Omis -> comportement historique (42 car/ligne), aucun changement pour les clients existants.

    speech_type, context_options, register (tous optionnels) composent le
    initial_prompt Whisper ("Contexte de transcription" côté Master Hub) :
    - speech_type : "monologue", "dialogue", "group" (défaut), "interview", "song"
    - context_options : sous-ensemble de "proper_nouns", "crude_language",
      "noisy_recording", "short_sentences", "long_sentences" (répéter le champ
      pour en envoyer plusieurs ; "short_sentences" prime si les deux "*_sentences"
      sont envoyées ensemble)
    - register : "urban", "suburb", "rural", "bourgeois", "standard" (défaut),
      "casual", "crude", "historical", "comedic", "thriller", "intimate_drama", "youth"
    Omis -> comportement par défaut (group + standard, sans option), aucun
    changement pour les clients existants.

    language (optionnel) force la langue de transcription (code ISO 639-1,
    ex. "fr", "en") au lieu de la détection automatique. Omis -> comportement
    historique (détection auto), aucun changement pour les clients existants.

    proper_nouns (optionnel, répéter le champ pour en envoyer plusieurs) :
    glossaire de noms propres attendus dans l'audio. Ajoute une phrase les
    citant nommément au initial_prompt (plus efficace que la phrase générique
    de l'option "proper_nouns") et corrige après coup, par similarité
    Levenshtein, les variantes d'orthographe proches vers la forme donnée.
    Omis -> comportement historique inchangé.
    """
    if (url is None) == (file is None):
        raise HTTPException(
            status_code=400,
            detail="Fournir soit 'url', soit 'file' (multipart/form-data), pas les deux ni aucun des deux",
        )
    if output_profile is not None and output_profile not in OUTPUT_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"output_profile invalide : {output_profile!r} (attendu : {', '.join(OUTPUT_PROFILES)})",
        )
    if speech_type is not None and speech_type not in SPEECH_TYPE_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"speech_type invalide : {speech_type!r} (attendu : {', '.join(SPEECH_TYPE_PROMPTS)})",
        )
    if register is not None and register not in REGISTER_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"register invalide : {register!r} (attendu : {', '.join(REGISTER_PROMPTS)})",
        )
    invalid_options = [o for o in context_options if o not in CONTEXT_OPTION_PROMPTS]
    if invalid_options:
        raise HTTPException(
            status_code=400,
            detail=f"context_options invalide(s) : {invalid_options!r} (attendu : {', '.join(CONTEXT_OPTION_PROMPTS)})",
        )
    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"language invalide : {language!r} (attendu : code ISO 639-1, ex. 'fr', 'en')",
        )
    proper_nouns = [n.strip() for n in proper_nouns if n.strip()]
    if len(proper_nouns) > 30 or any(len(n) > 60 for n in proper_nouns):
        raise HTTPException(
            status_code=400,
            detail="proper_nouns : 30 noms maximum, 60 caractères maximum chacun",
        )

    job_id = uuid.uuid4().hex[:12]
    upload_path: Path | None = None

    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Extension de fichier non supportée : {suffix or '(aucune)'} (attendu : .mp4 ou .wav)",
            )
        upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
        with upload_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        await file.close()
        source = str(upload_path)
    else:
        source = url

    title = _sanitize_title(file.filename if file is not None else url)
    resolved_prompt = _build_initial_prompt(speech_type, context_options, register, proper_nouns)
    job = Job(
        id=job_id,
        source=source,
        title=title,
        output_profile=output_profile,
        speech_type=speech_type,
        context_options=context_options,
        register=register,
        forced_language=language,
        proper_nouns=proper_nouns,
        initial_prompt=resolved_prompt,
    )
    JOBS[job_id] = job
    CHAR_LIMITS[job_id] = OUTPUT_PROFILES.get(output_profile, WRAP_WIDTH)
    INITIAL_PROMPTS[job_id] = resolved_prompt
    CANCEL_EVENTS[job_id] = threading.Event()
    if upload_path is not None:
        UPLOAD_PATHS[job_id] = upload_path
    await QUEUE.put(job_id)
    return job


@app.get("/jobs/{job_id}", response_model=Job, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job inconnu")
    return job


_TERMINAL_STATUSES = (Status.done, Status.error, Status.cancelled)


@app.post("/jobs/{job_id}/cancel", response_model=Job, dependencies=[Depends(require_api_key)])
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job inconnu")
    if job.status in _TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail=f"Job déjà terminé (status={job.status})")
    CANCEL_EVENTS[job_id].set()
    if job.status == Status.queued:
        # Pas encore pris en charge par le worker : on marque directement.
        job.status = Status.cancelled
    elif job.status != Status.cancelling:
        # En cours de traitement : le worker confirmera "cancelled" une fois arrêté.
        job.status = Status.cancelling
    return job


def _job_file(job_id: str, attr: str) -> tuple[Job, Path]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job inconnu")
    if job.status != Status.done:
        raise HTTPException(status_code=409, detail=f"Job pas encore terminé (status={job.status})")
    value = getattr(job, attr)
    if value is None:
        raise HTTPException(status_code=404, detail="Fichier déjà récupéré")
    path = Path(value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return job, path


def _delete_after_send(job: Job, attr: str, path: Path) -> None:
    path.unlink(missing_ok=True)
    setattr(job, attr, None)


@app.get("/jobs/{job_id}/srt", dependencies=[Depends(require_api_key)])
async def get_srt(job_id: str):
    job, path = _job_file(job_id, "srt_path")
    return FileResponse(
        path, media_type="text/plain", filename=f"{job.output_basename}.srt",
        background=BackgroundTask(_delete_after_send, job, "srt_path", path),
    )


@app.get("/jobs/{job_id}/vtt", dependencies=[Depends(require_api_key)])
async def get_vtt(job_id: str):
    job, path = _job_file(job_id, "vtt_path")
    return FileResponse(
        path, media_type="text/vtt", filename=f"{job.output_basename}.vtt",
        background=BackgroundTask(_delete_after_send, job, "vtt_path", path),
    )


@app.get("/jobs/{job_id}/qc", dependencies=[Depends(require_api_key)])
async def get_qc(job_id: str):
    job, path = _job_file(job_id, "qc_path")
    return FileResponse(
        path, media_type="application/json", filename=f"{job.output_basename}.qc.json",
        background=BackgroundTask(_delete_after_send, job, "qc_path", path),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "en_file": QUEUE.qsize()}

