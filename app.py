import asyncio
import os
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.background import BackgroundTask

from transcriber import Cancelled, Transcriber, to_srt, to_vtt

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "~/whisper-service/output")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "~/whisper-service/uploads")).expanduser()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Extensions de fichier acceptées pour un upload direct (mp4 ou wav, cf. demande).
ALLOWED_UPLOAD_SUFFIXES = {".mp4", ".wav"}

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
    status: Status = Status.queued
    progress: float = 0.0
    srt_path: str | None = None
    vtt_path: str | None = None
    error: str | None = None


# État partagé (en mémoire pour démarrer)
JOBS: dict[str, Job] = {}
CANCEL_EVENTS: dict[str, threading.Event] = {}
# Fichier uploadé (si la source du job est un upload direct, pas une URL) à
# supprimer une fois le job terminé — même logique de nettoyage que les sorties.
UPLOAD_PATHS: dict[str, Path] = {}
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

            segments = await loop.run_in_executor(
                None,
                lambda: transcriber.transcribe(
                    job.source,
                    on_phase=on_phase,
                    on_progress=on_progress,
                    cancel_event=cancel_event,
                ),
            )
            if not segments:
                raise RuntimeError("Aucune parole détectée dans le média")
            srt_path = OUTPUT_DIR / f"{job_id}.srt"
            vtt_path = OUTPUT_DIR / f"{job_id}.vtt"
            srt_path.write_text(to_srt(segments), encoding="utf-8")
            vtt_path.write_text(to_vtt(segments), encoding="utf-8")
            job.srt_path = str(srt_path)
            job.vtt_path = str(vtt_path)
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
):
    """Crée un job à partir d'une URL (mp4/wav) OU d'un fichier uploadé (mp4/wav) — pas les deux."""
    if (url is None) == (file is None):
        raise HTTPException(
            status_code=400,
            detail="Fournir soit 'url', soit 'file' (multipart/form-data), pas les deux ni aucun des deux",
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

    job = Job(id=job_id, source=source)
    JOBS[job_id] = job
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
        path, media_type="text/plain", filename=f"{job_id}.srt",
        background=BackgroundTask(_delete_after_send, job, "srt_path", path),
    )


@app.get("/jobs/{job_id}/vtt", dependencies=[Depends(require_api_key)])
async def get_vtt(job_id: str):
    job, path = _job_file(job_id, "vtt_path")
    return FileResponse(
        path, media_type="text/vtt", filename=f"{job_id}.vtt",
        background=BackgroundTask(_delete_after_send, job, "vtt_path", path),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "en_file": QUEUE.qsize()}

