"""Cœur de transcription : source (URL ou fichier) -> cues courts -> SRT/VTT."""
import io
import subprocess
from dataclasses import dataclass

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

    def _extract_audio(self, source: str) -> bytes:
        """ffmpeg streame la source (URL ou fichier) -> WAV 16 kHz mono sur stdout."""
        proc = subprocess.run(
            [
                "ffmpeg", "-i", source,
                "-vn", "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="ignore")[-500:]
            raise RuntimeError(f"Échec ffmpeg : {err}")
        return proc.stdout

    def _collect_words(self, source: str) -> list[Word]:
        """Transcription non-batchée : timestamps par mot fiables + moins de pertes."""
        audio = self._extract_audio(source)
        segments, _info = self.model.transcribe(
            io.BytesIO(audio),
            language=LANGUAGE,
            language_detection_threshold=LANGUAGE_DETECTION_THRESHOLD,
            language_detection_segments=LANGUAGE_DETECTION_SEGMENTS,
            vad_filter=VAD_ENABLED,
            vad_parameters=VAD_PARAMETERS if VAD_ENABLED else None,
            word_timestamps=True,
        )
        words: list[Word] = []
        for seg in segments:          # itérer déclenche le calcul
            if seg.words:
                words.extend(Word(w.start, w.end, w.word) for w in seg.words)
            else:
                words.append(Word(seg.start, seg.end, seg.text))
        return words

    def transcribe(self, source: str) -> list[Segment]:
        """Renvoie des sous-titres courts, découpés à partir des mots."""
        return self._build_cues(self._collect_words(source))

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

