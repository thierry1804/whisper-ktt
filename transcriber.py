"""Cœur de transcription : source (URL ou fichier) -> cues courts -> SRT/VTT."""
import io
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable

from faster_whisper import WhisperModel

# --- Configuration (le seul endroit à modifier pour CPU/GPU) ---------------
MODEL_SIZE = "small"
DEVICE = "cpu"            # "cpu" ou "cuda"
COMPUTE_TYPE = "int8"     # "int8" (CPU), "float16" ou "int8_float16" (GPU)
LANGUAGE = None           # code langue, ou None pour laisser Whisper détecter
LANGUAGE_DETECTION_THRESHOLD = 0.8   # confiance mini avant d'arrêter la détection
LANGUAGE_DETECTION_SEGMENTS = 3      # nb de segments audio testés pour l'atteindre

# Découpage des sous-titres (timecodes fins) --------------------------------
MAX_CUE_DURATION = 6.0    # durée max d'un sous-titre (secondes)
MAX_CUE_CHARS = 84        # caractères max par sous-titre (~2 lignes de 42)
MAX_WORD_GAP = 0.8        # coupe si le silence entre 2 mots dépasse ce seuil (s)
WRAP_WIDTH = 42           # largeur d'une ligne de sous-titre

# VAD : moins agressif = moins de risque de rogner la parole ----------------
VAD_ENABLED = True
VAD_PARAMETERS = dict(min_silence_duration_ms=500, speech_pad_ms=400)
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
_OUT_TIME_RE = re.compile(r"out_time_ms=(\d+)")

# faster-whisper ne "yield" un segment qu'une fois toute sa fenêtre de ~30s
# décodée (beam search bloquant), donc aucune progression réelle n'arrive
# entre-temps. HEARTBEAT_INTERVAL fait avancer une estimation basée sur le
# temps écoulé pendant ces trous ; RTF_ESTIMATE (secondes de calcul par
# seconde d'audio) est la conjecture de départ, recalibrée dès le premier
# segment réel pour coller au rythme observé.
TRANSCRIBE_HEARTBEAT_INTERVAL = 1.0
TRANSCRIBE_RTF_ESTIMATE = 3.0
# Valeur plancher du tout premier tick de heartbeat (déclenché immédiatement,
# pas au bout d'un TRANSCRIBE_HEARTBEAT_INTERVAL) : signale que la phase
# "transcribing" a bien démarré, même sur un job qui échoue en moins d'1s
# (ex : aucune parole détectée) et n'aurait sinon jamais rien reporté.
TRANSCRIBE_HEARTBEAT_FLOOR = 0.01


class Cancelled(Exception):
    """Levée quand un job est annulé pendant la conversion ou la transcription."""


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber:
    """Charge le modèle UNE fois, puis transcrit à la demande."""

    def __init__(self):
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    def _extract_audio(
        self,
        source: str,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bytes:
        """ffmpeg streame la source (URL ou fichier) -> WAV 16 kHz mono sur stdout.

        La progression (-progress sur stderr) est lue dans un thread séparé
        pendant que stdout (l'audio) est consommé, pour éviter un deadlock
        (les deux pipes se remplissent en parallèle).
        """
        proc = subprocess.Popen(
            [
                "ffmpeg", "-i", source,
                "-vn", "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", "-f", "wav",
                "-progress", "pipe:2", "-nostats",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        duration: float | None = None
        stderr_lines: list[str] = []

        def read_stderr() -> None:
            nonlocal duration
            for raw in proc.stderr:
                line = raw.decode("utf-8", errors="ignore")
                stderr_lines.append(line)
                if duration is None:
                    m = _DURATION_RE.search(line)
                    if m:
                        h, mi, s = m.groups()
                        duration = int(h) * 3600 + int(mi) * 60 + float(s)
                m = _OUT_TIME_RE.search(line)
                if m and on_progress and duration:
                    on_progress(min(int(m.group(1)) / 1_000_000 / duration, 1.0))

        def watch_cancel() -> None:
            # Se réveille toutes les 0.5s pour vérifier l'annulation ; s'arrête
            # de lui-même dès que le process ffmpeg se termine normalement.
            while proc.poll() is None:
                if cancel_event.wait(timeout=0.5):
                    proc.kill()
                    return

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        if cancel_event is not None:
            threading.Thread(target=watch_cancel, daemon=True).start()

        audio = proc.stdout.read()
        proc.stdout.close()
        proc.wait()
        stderr_thread.join()

        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("Conversion audio annulée")
        if proc.returncode != 0:
            err = "".join(stderr_lines)[-500:]
            raise RuntimeError(f"Échec ffmpeg : {err}")
        return audio

    def _collect_words(
        self,
        audio: bytes,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Word]:
        """Transcription non-batchée : timestamps par mot fiables + moins de pertes."""
        # Estimation immédiate de la durée à partir du WAV brut (16 kHz mono 16 bits,
        # en-tête canonique 44 octets — c'est le format que produit _extract_audio).
        # Nécessaire pour démarrer le heartbeat AVANT l'appel à model.transcribe()
        # ci-dessous : ce dernier calcule le VAD et les features de tout l'audio de
        # façon synchrone/bloquante avant de renvoyer `info.duration`, donc sans
        # cette estimation, aucune progression ne serait visible pendant cette phase
        # (le cas typique : un média sans parole, où aucun segment n'est jamais produit
        # et où le VAD engloutit tout le temps de la phase "transcribing").
        estimated_duration = max(len(audio) - 44, 0) / (16000 * 2)

        words: list[Word] = []

        # Progression : la valeur reportée ne redescend jamais (le max des
        # estimations "heartbeat" et des positions réelles des segments).
        best_progress = 0.0
        progress_lock = threading.Lock()
        rtf = TRANSCRIBE_RTF_ESTIMATE
        start_ts = time.monotonic()
        stop_heartbeat = threading.Event()

        def report(p: float) -> None:
            nonlocal best_progress
            with progress_lock:
                if p <= best_progress:
                    return
                best_progress = p
            on_progress(p)

        def heartbeat(duration: float) -> None:
            first = True
            while True:
                elapsed = time.monotonic() - start_ts
                p = min(elapsed / (duration * rtf), 0.95)
                if first:
                    p = max(p, TRANSCRIBE_HEARTBEAT_FLOOR)
                    first = False
                report(p)
                if stop_heartbeat.wait(TRANSCRIBE_HEARTBEAT_INTERVAL):
                    return

        hb_thread = None
        if on_progress and estimated_duration:
            hb_thread = threading.Thread(target=heartbeat, args=(estimated_duration,), daemon=True)
            hb_thread.start()

        segments, info = self.model.transcribe(
            io.BytesIO(audio),
            language=LANGUAGE,
            language_detection_threshold=LANGUAGE_DETECTION_THRESHOLD,
            language_detection_segments=LANGUAGE_DETECTION_SEGMENTS,
            vad_filter=VAD_ENABLED,
            vad_parameters=VAD_PARAMETERS if VAD_ENABLED else None,
            word_timestamps=True,
        )

        try:
            for seg in segments:          # itérer déclenche le calcul
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled("Transcription annulée")
                if seg.words:
                    words.extend(Word(w.start, w.end, w.word) for w in seg.words)
                else:
                    words.append(Word(seg.start, seg.end, seg.text))
                if on_progress and info.duration:
                    if seg.end > 0:
                        # Recalibre le rythme estimé sur le rythme réel observé.
                        rtf = max((time.monotonic() - start_ts) / seg.end, 0.1)
                    report(min(seg.end / info.duration, 1.0))
        finally:
            stop_heartbeat.set()
            if hb_thread:
                hb_thread.join(timeout=TRANSCRIBE_HEARTBEAT_INTERVAL)

        # Le VAD/décodage a parcouru la totalité de l'audio pour arriver jusqu'ici
        # (avec ou sans parole trouvée) : la phase "transcribing" est terminée à 100%,
        # même si l'estimation temporelle du heartbeat n'a pas eu le temps d'y arriver
        # (cas typique : un média sans parole, où le VAD seul est bien plus rapide
        # qu'une vraie transcription et termine son parcours complet en un instant).
        if on_progress:
            report(1.0)

        return words

    def transcribe(
        self,
        source: str,
        on_phase: Callable[[str], None] | None = None,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[Segment]:
        """Renvoie des sous-titres courts, découpés à partir des mots.

        on_phase("converting" | "transcribing") signale le changement d'étape ;
        on_progress(0.0..1.0) la progression au sein de l'étape courante ;
        cancel_event, si activé, interrompt le traitement (lève Cancelled).
        """
        if on_phase:
            on_phase("converting")
        audio = self._extract_audio(source, on_progress, cancel_event)
        if on_phase:
            on_phase("transcribing")
        words = self._collect_words(audio, on_progress, cancel_event)
        return self._build_cues(words)

    @staticmethod
    def _build_cues(words: list[Word]) -> list[Segment]:
        cues: list[Segment] = []
        buf: list[Word] = []

        def flush():
            if not buf:
                return
            text = "".join(w.text for w in buf).strip()
            cues.append(Segment(buf[0].start, buf[-1].end, _wrap(text)))
            buf.clear()

        for w in words:
            if buf:
                gap = w.start - buf[-1].end
                cur_text = "".join(x.text for x in buf).strip()
                cur_dur = buf[-1].end - buf[0].start
                if (gap > MAX_WORD_GAP
                        or cur_dur >= MAX_CUE_DURATION
                        or len(cur_text) + len(w.text) > MAX_CUE_CHARS):
                    flush()
            buf.append(w)
            if w.text.strip().endswith((".", "!", "?", ":")):
                flush()
        flush()
        return cues


def _wrap(text: str, width: int = WRAP_WIDTH) -> str:
    """Répartit le texte sur 1 ou 2 lignes équilibrées pour la lisibilité."""
    if len(text) <= width:
        return text
    words = text.split()
    best = None
    for i in range(1, len(words)):
        l1, l2 = " ".join(words[:i]), " ".join(words[i:])
        longest = max(len(l1), len(l2))
        if best is None or longest < best[0]:
            best = (longest, l1, l2)
    return f"{best[1]}\n{best[2]}"


# --- Formatage des sous-titres --------------------------------------------
def _ts(seconds: float, sep: str) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_ts(seg.start, ',')} --> {_ts(seg.end, ',')}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts(seg.start, '.')} --> {_ts(seg.end, '.')}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)

