import asyncio
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from transcriber import Transcriber, to_srt, to_vtt

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "~/whisper-service/output")).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("Variable d'environnement API_KEY manquante")

_api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(_api_key_header)) -> None:
    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Clé API invalide")


class Status(str, Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Job(BaseModel):
    id: str
    url: str
    status: Status = Status.queued
    srt_path: str | None = None
    vtt_path: str | None = None
    error: str | None = None


# État partagé (en mémoire pour démarrer)
JOBS: dict[str, Job] = {}
QUEUE: asyncio.Queue[str] = asyncio.Queue()
STATE: dict = {}


async def worker():
    """Consomme la file, une transcription à la fois."""
    transcriber: Transcriber = STATE["transcriber"]
    loop = asyncio.get_running_loop()
    while True:
        job_id = await QUEUE.get()
        job = JOBS[job_id]
        job.status = Status.processing
        try:
            # run_in_executor : sort le travail bloquant de la boucle asyncio.
            segments = await loop.run_in_executor(
                None, transcriber.transcribe, job.url
            )
            srt_path = OUTPUT_DIR / f"{job_id}.srt"
            vtt_path = OUTPUT_DIR / f"{job_id}.vtt"
            srt_path.write_text(to_srt(segments), encoding="utf-8")
            vtt_path.write_text(to_vtt(segments), encoding="utf-8")
            job.srt_path = str(srt_path)
            job.vtt_path = str(vtt_path)
            job.status = Status.done
        except Exception as exc:  # noqa: BLE001
            job.status = Status.error
            job.error = str(exc)
        finally:
            QUEUE.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle au démarrage (pas par requête).
    STATE["transcriber"] = Transcriber()
    task = asyncio.create_task(worker())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, title="Service de sous-titrage")


class JobRequest(BaseModel):
    url: str


@app.post("/jobs", response_model=Job, dependencies=[Depends(require_api_key)])
async def create_job(req: JobRequest):
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, url=req.url)
    JOBS[job_id] = job
    await QUEUE.put(job_id)
    return job


@app.get("/jobs/{job_id}", response_model=Job, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job inconnu")
    return job


def _job_file(job_id: str, attr: str) -> Path:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job inconnu")
    if job.status != Status.done:
        raise HTTPException(status_code=409, detail=f"Job pas encore terminé (status={job.status})")
    path = Path(getattr(job, attr))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return path


@app.get("/jobs/{job_id}/srt", dependencies=[Depends(require_api_key)])
async def get_srt(job_id: str):
    path = _job_file(job_id, "srt_path")
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}.srt")


@app.get("/jobs/{job_id}/vtt", dependencies=[Depends(require_api_key)])
async def get_vtt(job_id: str):
    path = _job_file(job_id, "vtt_path")
    return FileResponse(path, media_type="text/vtt", filename=f"{job_id}.vtt")


@app.get("/health")
async def health():
    return {"status": "ok", "en_file": QUEUE.qsize()}

