"""Cœur de transcription : source (URL ou fichier) -> cues courts -> SRT/VTT."""
import io
import re
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from faster_whisper import WhisperModel
from faster_whisper.tokenizer import _LANGUAGE_CODES

SUPPORTED_LANGUAGES = frozenset(_LANGUAGE_CODES)  # codes ISO 639-1 valides pour `language`

# --- Configuration (le seul endroit à modifier pour CPU/GPU) ---------------
MODEL_SIZE = "large-v3-turbo"
DEVICE = "cpu"            # "cpu" ou "cuda"
COMPUTE_TYPE = "int8"     # "int8" (CPU), "float16" ou "int8_float16" (GPU)
LANGUAGE = None           # code langue, ou None pour laisser Whisper détecter
LANGUAGE_DETECTION_THRESHOLD = 0.8   # confiance mini avant d'arrêter la détection
LANGUAGE_DETECTION_SEGMENTS = 3      # nb de segments audio testés pour l'atteindre

# Découpage manuel en tranches (pipeline Master Hub FR V1 §2.2) : Whisper est
# entraîné sur des clips de ~30s, l'attention (et la fiabilité) se dégrade
# sur des séquences continues bien plus longues, ce qui favorise les
# hallucinations sur les fichiers longs. Chevauchement entre tranches
# consécutives pour ne pas couper une phrase pile à la frontière ; la zone
# de chevauchement est dédupliquée en son milieu (cf. _collect_words),
# jamais laissée en double dans les mots collectés.
CHUNK_DURATION_SEC = 30.0
CHUNK_OVERLAP_SEC = 1.5

# Seuils anti-hallucination (pipeline Master Hub FR V1 §1.4) — épinglés
# explicitement plutôt que laissés aux défauts implicites de faster-whisper
# (qui les égalent aujourd'hui, sans garantie si la librairie change).
COMPRESSION_RATIO_THRESHOLD = 2.4
LOG_PROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6
TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Format audio produit par _extract_audio (WAV PCM 16 kHz mono 16 bits, en-tête
# canonique) — utilisé pour découper l'audio brut en tranches indépendantes.
_WAV_HEADER_SIZE = 44
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2

# Filigrane ajouté en fin de chaque transcription (SRT + VTT) -------------
WATERMARK_TEXT = "Transcription by MASTER HUB\nwww.killthetape.com"
WATERMARK_GAP = 0.5        # silence mini avant le filigrane si la parole va jusqu'au bout (secondes)
WATERMARK_LEAD_SEC = 8.0   # le filigrane démarre à ce nombre de secondes de la fin réelle du média
WATERMARK_DURATION = 3.0   # durée d'affichage du filigrane (secondes)

# Découpage des sous-titres (timecodes fins) --------------------------------
MAX_CUE_DURATION = 5.0    # durée max d'un sous-titre (secondes) — norme ATAA/KTT
MIN_CUE_DURATION = 0.67   # durée min d'un sous-titre (secondes) — ~16 images ATAA
MAX_WORD_GAP = 0.5        # coupe si le silence entre 2 mots dépasse ce seuil (s) —
                          # aligné sur min_silence_duration_ms du VAD (voir plus bas)
WRAP_WIDTH = 42           # largeur de ligne par défaut (aucun profil de sortie précisé)

# Condensation déterministe (module KTT §4.1, règles 1-2 uniquement) --------
# Volontairement restreint aux mots qui n'ont JAMAIS de sens propre en
# français/anglais (interjections pures) : "donc", "voilà", "enfin", "bon"
# etc. sont ambigus (parfois du remplissage, parfois du sens réel) et ne
# sont donc pas supprimés automatiquement — risque trop élevé d'altérer le
# sens d'une phrase sur un cas particulier.
_FILLER_WORDS = {"euh", "heu", "hum", "hmm", "um", "uh", "er", "erm"}
_REPEAT_MAX_GAP = 0.5     # au-delà, deux mots identiques ne sont plus une
                          # simple répétition/bégaiement mais une reprise
                          # volontaire (ex. emphase) — on ne la supprime pas

# Contexte de transcription (initial_prompt Whisper) -------------------------
# Composé ici à partir de valeurs catégorielles envoyées par KTT (même
# principe que OUTPUT_PROFILES) plutôt que côté front KTT : un seul endroit
# à faire évoluer si le phrasing doit être ajusté après des tests réels.
SPEECH_TYPE_PROMPTS = {
    "monologue": "Monologue / voiceover with a single narrator.",
    "dialogue": "Dialogue between two speakers.",
    "group": "Group conversation with three to four speakers.",
    "interview": "Interview with one interviewer and one interviewee.",
    "song": (
        "This is a song with sung lyrics, not spoken dialogue. "
        "Melodic phrasing, repetitions, occasional vocal effects."
    ),
}
DEFAULT_SPEECH_TYPE = "group"

CONTEXT_OPTION_PROMPTS = {
    "proper_nouns": (
        "Many proper nouns: character names, street names, organizations; "
        "preserve capitalization and spelling."
    ),
    "crude_language": "Strong language is expected and should be transcribed verbatim.",
    "noisy_recording": "Raw on-set recording, background noise, overlapping speech, interruptions.",
    "short_sentences": "Short sentences, clear punctuation, frequent full stops.",
    "long_sentences": "Longer sentences, flowing style, fewer full stops, more commas.",
}

# NOTE : seul le texte "urban" est repris tel quel de la spec fournie. Les 11
# autres sont des PREMIERS JETS (même style/structure que l'exemple donné) —
# à valider ou remplacer par les versions déjà rédigées mentionnées dans la
# spec mais non incluses dans le document reçu.
# Textes officiels Master Hub (source : "Prompts contexte pour
# transcription/traduction", remis par mail le 24/09/2026) — remplacent les
# premiers jets précédents pour les 12 presets.
REGISTER_PROMPTS = {
    "urban": (
        "This is a film set in an urban environment, big city. "
        "Street language, casual to very casual register, urban slang, "
        "short sentences, frequent contractions, oral phrasing. "
        "Direct tone, sometimes tense."
    ),
    "suburb": (
        "This is a film set in suburban working-class neighborhoods. "
        "Very casual register, specific urban slang, clipped sentences, "
        "contractions, ellipses, occasional strong language. "
        "Frank tone, sometimes confrontational."
    ),
    "rural": (
        "This is a film set in a rural, countryside environment. "
        "Casual to familiar register, possible regional expressions, "
        "slower pace, sometimes incomplete sentences, no formal language. "
        "Simple and direct tone."
    ),
    "bourgeois": (
        "This is a film set in an upper-class, bourgeois milieu. "
        "Formal register, complete sentences, polished vocabulary, "
        "no slang or strong language. "
        "Serious or composed tone, intense but well-formed dialogue."
    ),
    "standard": (
        "This is a film with everyday dialogue in standard language. "
        "Neutral register, sometimes incomplete sentences, light contractions, "
        "no marked slang. "
        "Neutral tone, neither very formal nor very casual."
    ),
    "casual": (
        "This is a film with everyday dialogue between close people. "
        "Casual register, short sentences, frequent contractions, "
        "oral phrasing, no strong language. "
        "Natural, relaxed or intimate tone."
    ),
    "crude": (
        "This is a film with raw, coarse dialogue. "
        "Very casual register, frequent strong language, short sentences, "
        "direct and sometimes aggressive tone, no formal language. "
        "Urban or working-class environment."
    ),
    "historical": (
        "This is a historical film with formal language. "
        "Formal register, complete sentences, precise vocabulary, "
        "no modern slang, serious and composed tone. "
        "References to period events or characters."
    ),
    "comedic": (
        "This is a comedy with humorous dialogue. "
        "Casual register, wordplay, common expressions, short sentences, "
        "light and friendly tone. "
        "No formal language, occasional slang expressions."
    ),
    "thriller": (
        "This is a crime film / noir. "
        "Direct language, casual to very casual register, short sentences, "
        "dry and tense tone, occasional strong language. "
        "Dark urban environment, investigative or criminal dialogue."
    ),
    "intimate_drama": (
        "This is an intimate drama with dialogue between close people. "
        "Casual register, sentences sometimes broken by emotion, "
        "serious and sincere tone, no formal language. "
        "Scenes of conflict or personal revelation."
    ),
    "youth": (
        "This is a film centered on teenagers. "
        "Very casual register, youth slang, short sentences, contractions, "
        "oral phrasing, occasional strong language. "
        "Spontaneous tone, sometimes provocative or ironic."
    ),
}
DEFAULT_REGISTER = "standard"

# Cap défensif : Whisper tronque de toute façon en interne si le prompt
# dépasse sa fenêtre de contexte, mais on évite d'envoyer une chaîne absurde
# au modèle si un appelant mal formé envoyait un pavé de texte.
INITIAL_PROMPT_MAX_CHARS = 1000


def _build_initial_prompt(
    speech_type: str | None,
    context_options: list[str] | None,
    register: str | None,
    proper_nouns: list[str] | None = None,
) -> str:
    """Compose le initial_prompt Whisper à partir de valeurs catégorielles
    (registre, type de parole, options de contexte) — cf. spec KTT "Contexte
    de transcription". Toujours renvoie une chaîne non vide (au moins le
    registre + type de parole par défaut), même si l'appelant ne renseigne
    aucun champ : décision produit explicite, le prompt par défaut doit
    s'appliquer à tous les jobs.

    proper_nouns (optionnel) : liste de noms propres attendus dans l'audio
    (cf. pipeline Master Hub FR V1 §1.10). Quand fournie, remplace la phrase
    générique de l'option "proper_nouns" par une phrase citant les noms
    eux-mêmes — Whisper conditionne sur le texte du prompt, donc voir les
    noms réels biaise bien plus efficacement l'orthographe que la phrase
    générique seule. Plafonnée à 10 noms pour rester dans le budget de
    tokens de Whisper (~224)."""
    register_key = register if register in REGISTER_PROMPTS else DEFAULT_REGISTER
    speech_key = speech_type if speech_type in SPEECH_TYPE_PROMPTS else DEFAULT_SPEECH_TYPE

    parts = [REGISTER_PROMPTS[register_key], SPEECH_TYPE_PROMPTS[speech_key]]

    options = set(context_options or [])
    if "short_sentences" in options and "long_sentences" in options:
        options.discard("long_sentences")  # priorité aux phrases courtes
    nouns = [n.strip() for n in (proper_nouns or []) if n.strip()]
    if nouns:
        options.discard("proper_nouns")  # la liste réelle remplace la phrase générique
    for key in ("proper_nouns", "crude_language", "noisy_recording", "short_sentences", "long_sentences"):
        if key in options:
            parts.append(CONTEXT_OPTION_PROMPTS[key])
    if nouns:
        names = ", ".join(nouns[:10])
        parts.append(f"Many proper nouns: {names}; preserve capitalization and spelling.")

    return " ".join(parts)[:INITIAL_PROMPT_MAX_CHARS]

# Découpage par unité de sens (module KTT §5.1) : quand une coupure est
# forcée par la durée/longueur (pas par un vrai silence), on cherche un point
# de coupure naturel dans le tampon, et on évite de terminer un sous-titre
# sur un connecteur court si la phrase continue juste après (norme ATAA :
# ne pas séparer une préposition de son infinitif, un sujet de son verbe).
_BREAK_PUNCT = (",", ";", ":", ".", "!", "?")

# Marge tolérée pour ne pas couper au milieu d'un nom propre/composé (deux
# mots capitalisés consécutifs, ex. "Seven Mile") : si la coupure tomberait
# pile entre les deux, on tolère un léger dépassement de la LONGUEUR max
# (le texte reste lisible sur 2 lignes) — mais jamais de la DURÉE max :
# plusieurs mots capitalisés consécutifs sans rapport (ex. deux noms propres
# différents à la suite) peuvent chaîner cette protection et pousser la
# durée bien au-delà de 5s si on la tolère aussi ; la norme ATAA/KTT sur la
# durée n'a pas d'exception documentée, donc on la fait toujours respecter.
PROPER_NOUN_SLACK_CHARS = 20
# Marge de durée tolérée pour ne pas couper un nom composé/propre au
# milieu (ex. "Seven Mile Miracle"), bornée par MAX_NAME_RUN_DEFERS pour
# empêcher un enchaînement de plusieurs mots capitalisés SANS rapport
# (bug distinct observé en français) de repousser la durée bien au-delà
# de MAX_CUE_DURATION.
PROPER_NOUN_SLACK_SEC = 1.5
MAX_NAME_RUN_DEFERS = 2
_CONJUNCTIONS = {
    "et", "mais", "ou", "donc", "car", "ni", "or",
    "and", "but", "so", "because", "nor",
}
_TRAILING_STOPWORDS = {
    "à", "de", "du", "des", "au", "aux", "que", "qui", "pour", "dans", "sur",
    "avec", "sans", "le", "la", "les", "un", "une",
    "to", "of", "in", "on", "at", "for", "with", "the", "a", "an",
} | _CONJUNCTIONS

# Alignement ATAA (normes de sous-titrage FR) --------------------------------
FRAME_RATE = 25.0         # hypothèse : broadcast FR (25 im/s) — sert de base au calcul du gap
MIN_GAP_FRAMES = 4        # gap minimal ATAA entre 2 sous-titres (6 sur changement de plan,
                          # non applicable ici : pas d'analyse vidéo/scène dans ce pipeline audio)
MIN_GAP = MIN_GAP_FRAMES / FRAME_RATE

# Vitesse de lecture cible ATAA : 14-15 c/s en moyenne, 17 c/s = trop rapide.
# N'est PAS utilisé comme déclencheur de découpage (testé le 18/09/2026 :
# re-couper le texte ne réduit pas la densité de parole sous-jacente, ça
# fragmente sans améliorer le taux de conformité) — sert uniquement à
# journaliser un warning QC (CPS_HIGH) sur le cue final.
MAX_CPS = 17.0

# Garde-fou contre un timestamp de mot aberrant côté Whisper (observé : un
# mot isolé suivi d'un long silence peut se voir attribuer une fin de mot
# étirée jusqu'au silence/mot suivant au lieu de sa vraie durée d'énoncé —
# ex. un mot de 4 caractères mesuré à 14,88s). Sans ce plafond, un seul mot
# corrompu peut produire un cue qui dépasse MAX_CUE_DURATION de façon
# spectaculaire, sans qu'aucune des vérifications de _build_cues ne s'en
# aperçoive (elles ne s'appliquent qu'à partir du 2e mot d'un tampon).
WORD_MAX_DURATION = 2.0

# Caractères par ligne selon le support de sortie (norme ATAA, valeurs médianes) --
OUTPUT_PROFILES = {
    "tv": 36,         # broadcast/TV : 36-37
    "video": 39,      # vidéo/production : 38-40
    "streaming": 42,  # streaming : 40-42
    "cinema": 40,     # cinéma : 40-41
}

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


def _wav_header(pcm_len: int, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """En-tête WAV canonique (44 octets) pour un flux PCM mono 16 bits."""
    byte_rate = sample_rate * _BYTES_PER_SAMPLE
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + pcm_len, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, byte_rate, _BYTES_PER_SAMPLE, 16,
        b"data", pcm_len,
    )


def _slice_wav(audio: bytes, start_sec: float, end_sec: float) -> bytes:
    """Extrait une tranche de l'audio WAV brut et la réemballe avec un
    en-tête WAV valide — chaque tranche doit être un fichier autonome pour
    être passée indépendamment à Whisper."""
    pcm = audio[_WAV_HEADER_SIZE:]
    bytes_per_sec = _SAMPLE_RATE * _BYTES_PER_SAMPLE
    start_byte = min(int(start_sec * bytes_per_sec), len(pcm))
    end_byte = min(int(end_sec * bytes_per_sec), len(pcm))
    # Aligné sur une frontière d'échantillon (2 octets) pour ne pas corrompre
    # le flux PCM 16 bits.
    start_byte -= start_byte % _BYTES_PER_SAMPLE
    end_byte -= end_byte % _BYTES_PER_SAMPLE
    return _wav_header(max(end_byte - start_byte, 0)) + pcm[start_byte:end_byte]


def _chunk_starts(total_duration: float) -> list[float]:
    """Bornes de départ (secondes) des tranches de CHUNK_DURATION_SEC, avec
    un chevauchement de CHUNK_OVERLAP_SEC entre tranches consécutives. Un
    fichier plus court qu'une tranche renvoie une seule tranche couvrant
    tout l'audio — comportement identique à l'ancien mode "un seul appel"."""
    if total_duration <= CHUNK_DURATION_SEC:
        return [0.0]
    stride = CHUNK_DURATION_SEC - CHUNK_OVERLAP_SEC
    starts = []
    t = 0.0
    while True:
        starts.append(t)
        if t + CHUNK_DURATION_SEC >= total_duration:
            break
        t += stride
    return starts


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
    warnings: list[str] = field(default_factory=list)  # log QC (cf. module KTT)


def _remove_disfluencies(words: list[Word]) -> list[Word]:
    """Condensation déterministe (module KTT §4.1, règles 1-2) : supprime les
    interjections pures de remplissage et les répétitions immédiates par
    bégaiement (ex. "je je pense" -> "je pense"). Ne touche jamais à un mot
    qui porte du sens, même ambigu (cf. _FILLER_WORDS, volontairement
    restreint)."""
    cleaned: list[Word] = []
    for w in words:
        token = w.text.strip().strip(".,;:!?").lower()
        if token in _FILLER_WORDS:
            continue
        if cleaned:
            prev = cleaned[-1]
            prev_token = prev.text.strip().strip(".,;:!?").lower()
            if prev_token == token and token and (w.start - prev.end) < _REPEAT_MAX_GAP:
                cleaned.pop()
        cleaned.append(w)
    return cleaned


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
        initial_prompt: str | None = None,
        language: str | None = None,
    ) -> tuple[list[Word], str, float]:
        """Transcrit l'audio par tranches de CHUNK_DURATION_SEC avec
        chevauchement (pipeline Master Hub FR V1 §2.2-2.3) plutôt qu'en un
        seul appel : réduit le risque d'hallucination/dérive sur les
        fichiers longs. Le chevauchement entre deux tranches consécutives
        est coupé en son milieu (temps global) — jamais laissé en double
        dans les mots collectés. Un fichier ≤ CHUNK_DURATION_SEC tient dans
        une seule tranche : comportement identique à l'ancien mode "un seul
        appel" (aucune régression pour les fichiers courts déjà testés)."""
        total_duration = max(len(audio) - _WAV_HEADER_SIZE, 0) / (_SAMPLE_RATE * _BYTES_PER_SAMPLE)
        chunk_starts = _chunk_starts(total_duration)
        # Frontière (temps global) entre la tranche i et la tranche i+1, au
        # milieu de leur zone de chevauchement commune.
        boundaries = [
            chunk_starts[i] + CHUNK_DURATION_SEC - CHUNK_OVERLAP_SEC / 2
            for i in range(len(chunk_starts) - 1)
        ]

        words: list[Word] = []
        detected_language: str | None = None

        # Progression : la valeur reportée ne redescend jamais (le max des
        # estimations "heartbeat" et des positions réelles, mises à l'échelle
        # de la durée TOTALE — pas seulement de la tranche en cours).
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
        if on_progress and total_duration:
            hb_thread = threading.Thread(target=heartbeat, args=(total_duration,), daemon=True)
            hb_thread.start()

        try:
            for idx, chunk_start in enumerate(chunk_starts):
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled("Transcription annulée")
                chunk_end = min(chunk_start + CHUNK_DURATION_SEC, total_duration)
                chunk_audio = _slice_wav(audio, chunk_start, chunk_end)
                # Une fois la langue détectée sur la 1ère tranche, on la fige
                # pour les suivantes : chaque tranche ne contient parfois que
                # quelques mots, une détection indépendante par tranche
                # risquerait de donner une langue différente d'une tranche à
                # l'autre sur le même fichier.
                chunk_language = language if language is not None else detected_language

                segments, info = self.model.transcribe(
                    io.BytesIO(chunk_audio),
                    language=chunk_language,
                    language_detection_threshold=LANGUAGE_DETECTION_THRESHOLD,
                    language_detection_segments=LANGUAGE_DETECTION_SEGMENTS,
                    vad_filter=VAD_ENABLED,
                    vad_parameters=VAD_PARAMETERS if VAD_ENABLED else None,
                    word_timestamps=True,
                    initial_prompt=initial_prompt,
                    # Défaut faster-whisper = True : chaque segment est conditionné
                    # sur le texte du précédent, ce qui favorise les répétitions
                    # d'un sous-titre à l'autre (cf. pipeline Master Hub FR V1 §1.1).
                    condition_on_previous_text=False,
                    compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                    log_prob_threshold=LOG_PROB_THRESHOLD,
                    no_speech_threshold=NO_SPEECH_THRESHOLD,
                    temperature=TEMPERATURE_FALLBACK,
                )
                if detected_language is None:
                    detected_language = info.language

                lower = boundaries[idx - 1] if idx > 0 else None
                upper = boundaries[idx] if idx < len(boundaries) else None

                for seg in segments:          # itérer déclenche le calcul
                    if cancel_event is not None and cancel_event.is_set():
                        raise Cancelled("Transcription annulée")
                    if seg.words:
                        for w in seg.words:
                            g_start = w.start + chunk_start
                            g_end = min(w.end, w.start + WORD_MAX_DURATION) + chunk_start
                            if lower is not None and g_end <= lower:
                                continue  # déjà couvert par la tranche précédente
                            if upper is not None and g_start >= upper:
                                continue  # couvert par la tranche suivante
                            words.append(Word(g_start, g_end, w.word))
                    else:
                        g_start = seg.start + chunk_start
                        g_end = min(seg.end, seg.start + WORD_MAX_DURATION) + chunk_start
                        in_overlap = (lower is not None and g_end <= lower) or (
                            upper is not None and g_start >= upper
                        )
                        if not in_overlap:
                            words.append(Word(g_start, g_end, seg.text))

                    global_end = chunk_start + seg.end
                    if on_progress and total_duration:
                        if global_end > 0:
                            # Recalibre le rythme estimé sur le rythme réel observé.
                            rtf = max((time.monotonic() - start_ts) / global_end, 0.1)
                        report(min(global_end / total_duration, 1.0))
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

        return words, detected_language, total_duration

    def transcribe(
        self,
        source: str,
        on_phase: Callable[[str], None] | None = None,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
        char_limit: int = WRAP_WIDTH,
        initial_prompt: str | None = None,
        language: str | None = None,
        proper_nouns: list[str] | None = None,
    ) -> tuple[list[Segment], str, float]:
        """Renvoie (sous-titres courts découpés à partir des mots, langue
        détectée, durée réelle de l'audio en secondes).

        on_phase("converting" | "transcribing") signale le changement d'étape ;
        on_progress(0.0..1.0) la progression au sein de l'étape courante ;
        cancel_event, si activé, interrompt le traitement (lève Cancelled) ;
        char_limit fixe la largeur de ligne (voir OUTPUT_PROFILES).
        """
        if on_phase:
            on_phase("converting")
        audio = self._extract_audio(source, on_progress, cancel_event)
        if on_phase:
            on_phase("transcribing")
        words, detected_language, duration = self._collect_words(
            audio, on_progress, cancel_event, initial_prompt, language
        )
        words = _remove_disfluencies(words)
        cues = self._build_cues(words, char_limit)
        cues = _dedup_internal(cues, char_limit)
        cues = _dedup_cross_segment(cues, char_limit)
        cues = _merge_stray_connectors(cues, char_limit)
        cues = _merge_split_names(cues, char_limit)
        cues = _merge_short_cues(cues, char_limit)
        cues = _enforce_min_gap(cues)
        cues = _normalize_proper_nouns(cues, proper_nouns, char_limit)
        _compute_warnings(cues)
        return cues, detected_language, duration

    @staticmethod
    def _build_cues(words: list[Word], char_limit: int = WRAP_WIDTH) -> list[Segment]:
        max_cue_chars = 2 * char_limit
        cues: list[Segment] = []
        buf: list[Word] = []
        name_run_defers = 0

        def natural_break_index() -> int | None:
            """Cherche en arrière dans buf un point de coupure naturel
            (ponctuation interne ou conjonction) pour ne pas couper au milieu
            d'une unité de sens. Renvoie l'index après lequel couper, ou None
            si aucun point de coupure exploitable n'a été trouvé."""
            for i in range(len(buf) - 1, 0, -1):
                if buf[i - 1].text.strip().endswith(_BREAK_PUNCT):
                    return i
            for i in range(len(buf) - 1, 0, -1):
                token = buf[i].text.strip().strip(".,;:!?").lower()
                if token in _CONJUNCTIONS:
                    return i
            return None

        def flush(forced: bool = False, allow_hold: bool = True) -> None:
            if not buf:
                return
            held: list[Word] = []
            # Coupure forcée (durée/longueur, pas un vrai silence) : on tente
            # d'abord de reculer jusqu'à un point de coupure naturel...
            if allow_hold and forced and len(buf) > 1:
                idx = natural_break_index()
                if idx is not None and (buf[idx - 1].end - buf[0].start) >= MIN_CUE_DURATION:
                    held = buf[idx:]
                    del buf[idx:]
            # NB : la protection des connecteurs isolés et des noms propres
            # coupés se fait en post-traitement (_merge_stray_connectors,
            # _merge_split_names) plutôt qu'ici — un mot problématique se
            # retrouve trop souvent seul en tête de tampon après un flush
            # précédent, où "len(buf) > 1" ne peut pas s'appliquer.
            text = "".join(w.text for w in buf).strip()
            cues.append(Segment(buf[0].start, buf[-1].end, _wrap(text, char_limit)))
            buf.clear()
            buf.extend(held)

        for w in words:
            if buf:
                gap = w.start - buf[-1].end
                cur_text = "".join(x.text for x in buf).strip()
                # Durée qu'aurait le cue SI w y était ajouté : le seuil doit
                # empêcher le dépassement, pas seulement le constater après
                # coup (sinon le dernier mot ajouté peut faire dépasser
                # MAX_CUE_DURATION avant que la vérification suivante ne s'en
                # aperçoive).
                prospective_dur = w.end - buf[0].start
                prospective_chars = len(cur_text) + len(w.text)
                gap_triggered = gap > MAX_WORD_GAP
                overflow_triggered = (
                    prospective_dur > MAX_CUE_DURATION
                    or prospective_chars > max_cue_chars
                )
                if gap_triggered or overflow_triggered:
                    # Ne pas couper pile entre deux mots capitalisés
                    # consécutifs (nom propre/composé probable en français,
                    # ex. "Seven Mile") si le dépassement reste dans la
                    # marge tolérée : on laisse le mot rejoindre le tampon
                    # au lieu de flusher maintenant.
                    last_tok = buf[-1].text.strip()
                    new_tok = w.text.strip().strip(".,;:!?")
                    continues_name = (
                        last_tok[:1].isalpha() and last_tok[:1].isupper()
                        and new_tok[:1].isalpha() and new_tok[:1].isupper()
                    )
                    within_slack = (
                        prospective_dur <= MAX_CUE_DURATION + PROPER_NOUN_SLACK_SEC
                        and len(cur_text) + len(w.text) <= max_cue_chars + PROPER_NOUN_SLACK_CHARS
                    )
                    if continues_name and within_slack and name_run_defers < MAX_NAME_RUN_DEFERS:
                        name_run_defers += 1
                    else:
                        flush(forced=overflow_triggered and not gap_triggered)
                        name_run_defers = 0
            buf.append(w)
            if w.text.strip().endswith((".", "!", "?", ":")):
                flush(forced=False)
                name_run_defers = 0
        # Dernier flush : rien à qui transmettre un mot mis de côté, donc pas
        # de "hold" possible ici (sinon ce mot serait silencieusement perdu).
        flush(forced=False, allow_hold=False)
        return cues


def _wrap(text: str, width: int = WRAP_WIDTH) -> str:
    """Répartit le texte sur 1 ou 2 lignes équilibrées pour la lisibilité."""
    if len(text) <= width:
        return text
    words = text.split()
    if len(words) <= 1:
        return text
    best = None
    for i in range(1, len(words)):
        l1, l2 = " ".join(words[:i]), " ".join(words[i:])
        longest = max(len(l1), len(l2))
        if best is None or longest < best[0]:
            best = (longest, l1, l2)
    return f"{best[1]}\n{best[2]}"


def _dedup_internal(cues: list[Segment], char_limit: int = WRAP_WIDTH) -> list[Segment]:
    """Supprime, à l'intérieur d'un même cue, une séquence de 2 à 8 mots
    immédiatement répétée (bégaiement/boucle du modèle) — sauf répétition
    courte manifestement volontaire ("non, non", "oui, oui"). Distinct de
    _remove_disfluencies (qui ne traite que la répétition d'un seul mot au
    niveau des mots bruts, avant le découpage en cues). Cf. module de
    post-traitement Master Hub (pipeline FR V1, passe 2)."""
    result: list[Segment] = []
    for seg in cues:
        text = seg.text.replace("\n", " ")
        tokens = text.split()
        changed = False
        if len(tokens) >= 4:
            max_size = min(8, len(tokens) // 2)
            for size in range(max_size, 1, -1):
                for i in range(0, len(tokens) - 2 * size + 1):
                    a = tokens[i:i + size]
                    b = tokens[i + size:i + 2 * size]
                    if a == b:
                        norm_a = [t.strip(".,;:!?").lower() for t in a]
                        # Répétition courte manifestement volontaire à l'oral
                        # ("non, non", "oui, oui", "allez, allez") : jamais
                        # supprimée, même si elle ressemble à un bégaiement.
                        if size == 2 and norm_a == norm_a[:1] * 2 and norm_a[0] in {"non", "oui", "allez", "vas-y"}:
                            continue
                        tokens = tokens[:i + size] + tokens[i + 2 * size:]
                        changed = True
                        break
                if changed:
                    break
        if changed:
            new_seg = Segment(seg.start, seg.end, _wrap(" ".join(tokens), char_limit))
            new_seg.warnings = list(seg.warnings) + ["INTERNAL_DUPLICATE_REMOVED"]
            result.append(new_seg)
        else:
            result.append(seg)
    return result


def _dedup_cross_segment(cues: list[Segment], char_limit: int = WRAP_WIDTH) -> list[Segment]:
    """Supprime, en début d'un cue, les mots qui répètent la fin du cue
    précédent (chevauchement de segments Whisper à la frontière) — signe
    d'un artefact de segmentation plutôt que d'une répétition voulue dans le
    dialogue. Cf. module de post-traitement Master Hub (pipeline FR V1,
    passe 3)."""
    result = list(cues)
    i = 0
    while i < len(result) - 1:
        cur, nxt = result[i], result[i + 1]
        cur_words = cur.text.replace("\n", " ").split()
        nxt_words = nxt.text.replace("\n", " ").split()
        max_len = min(len(cur_words), len(nxt_words))
        overlap_n = 0
        for n in range(max_len, 0, -1):
            a = [w.strip(".,;:!?").lower() for w in cur_words[-n:]]
            b = [w.strip(".,;:!?").lower() for w in nxt_words[:n]]
            if a == b:
                overlap_n = n
                break
        if overlap_n:
            ratio = overlap_n / max(1, min(len(cur_words), len(nxt_words)))
            if overlap_n >= 3 or ratio >= 0.5:
                remaining = nxt_words[overlap_n:]
                if remaining:
                    new_seg = Segment(nxt.start, nxt.end, _wrap(" ".join(remaining), char_limit))
                    new_seg.warnings = list(nxt.warnings) + ["CROSS_SEGMENT_OVERLAP_REMOVED"]
                    result[i + 1] = new_seg
                else:
                    del result[i + 1]
                    continue
        i += 1
    return result


def _compute_warnings(cues: list[Segment]) -> None:
    """Calcule le log QC (module KTT) sur les cues finalisés — dernière étape
    du pipeline, après toutes les fusions/découpages, pour refléter ce qui
    est réellement livré plutôt qu'un état intermédiaire. Mute les cues en
    place (ajoute à Segment.warnings), ne modifie ni le texte ni le timing."""
    for seg in cues:
        dur = seg.end - seg.start
        text = seg.text.replace("\n", " ").strip()
        if dur > 1e-6 and len(text) / dur > MAX_CPS:
            seg.warnings.append("CPS_HIGH")
        if dur < MIN_CUE_DURATION - 1e-6:
            seg.warnings.append("SHORT_CUE_KEPT")
        words_in_cue = text.split()
        if len(words_in_cue) == 1 and not text.rstrip().endswith((".", "!", "?")):
            seg.warnings.append("ISOLATED_WORD_KEPT")


def _merge_stray_connectors(cues: list[Segment], char_limit: int = WRAP_WIDTH) -> list[Segment]:
    """Fusionne un sous-titre réduit à un seul mot SANS ponctuation de fin de
    phrase (jamais un énoncé complet — contrairement à "Oui."/"Non.") avec le
    voisin le plus proche dans le temps, texte inclus. Couvre aussi bien un
    connecteur/interjection isolé (ex. "Et", "Que") qu'un mot porteur de sens
    coupé de la suite d'une phrase par une pause détectée par Whisper (ex.
    "Hier" séparé de "soir, ..."). Opère sur des cues déjà finalisés : le
    calcul de proximité (avant/après) n'est donc pas faussé par un grand
    silence traversé pendant l'accumulation des mots."""
    max_chars = 2 * char_limit
    result = list(cues)
    i = 0
    while i < len(result):
        seg = result[i]
        stripped = seg.text.strip()
        is_single_word = " " not in stripped and "\n" not in seg.text
        ends_sentence = stripped.endswith((".", "!", "?"))
        if is_single_word and stripped and not ends_sentence:
            sole = stripped.strip(".,;:!?").lower()
            is_noise_word = sole in _TRAILING_STOPWORDS or sole in _FILLER_WORDS
            prev_seg = result[i - 1] if i > 0 else None
            next_seg = result[i + 1] if i + 1 < len(result) else None
            gap_prev = (seg.start - prev_seg.end) if prev_seg is not None else None
            gap_next = (next_seg.start - seg.end) if next_seg is not None else None
            # Ne jamais recoller après une phrase déjà terminée par une
            # ponctuation forte (., !, ?) : ce serait grammaticalement
            # absurde ("...Prêt ? Que") même si durée/caractères restent
            # dans les bornes.
            prev_closed = prev_seg is not None and prev_seg.text.strip().endswith((".", "!", "?"))
            prefer_prev = (
                gap_prev is not None
                and not prev_closed
                and (gap_next is None or gap_prev <= gap_next)
            )

            merged = False
            if prefer_prev:
                combined_text = (prev_seg.text.replace("\n", " ") + " " + seg.text.replace("\n", " ")).strip()
                combined_dur = seg.end - prev_seg.start
                if len(combined_text) <= max_chars and combined_dur <= MAX_CUE_DURATION:
                    result[i - 1] = Segment(prev_seg.start, seg.end, _wrap(combined_text, char_limit))
                    del result[i]
                    merged = True
            elif next_seg is not None:
                combined_text = (seg.text.replace("\n", " ") + " " + next_seg.text.replace("\n", " ")).strip()
                combined_dur = next_seg.end - seg.start
                if len(combined_text) <= max_chars and combined_dur <= MAX_CUE_DURATION:
                    result[i] = Segment(seg.start, next_seg.end, _wrap(combined_text, char_limit))
                    del result[i + 1]
                    merged = True

            if merged:
                continue
            if is_noise_word:
                # Un connecteur/interjection isolé n'est jamais un énoncé
                # complet et ne peut pas fusionner sans violer une autre
                # règle : bruit probable (répétition parasite Whisper),
                # supprimé plutôt que laissé affiché seul.
                del result[i]
                continue
            # Un mot porteur de sens (ex. "Hier") qui ne peut pas fusionner
            # reste tel quel : on ne supprime jamais du contenu réel, même
            # imparfaitement découpé (compromis accepté, cf. module KTT §6).
        i += 1
    return result


def _merge_split_names(cues: list[Segment], char_limit: int = WRAP_WIDTH) -> list[Segment]:
    """Recolle deux cues consécutifs quand tout indique qu'un nom propre ou
    composé (ex. "Papi" / "Chouette") a été coupé entre les deux : le premier
    ne se termine pas par une ponctuation de fin de phrase et se termine par
    un mot capitalisé, le second commence par un mot capitalisé. En français,
    une majuscule en milieu de phrase signale quasi toujours un nom propre —
    ce test a donc un faible taux de faux positifs sur ce type de contenu."""
    max_chars = 2 * char_limit
    result = list(cues)
    i = 0
    while i < len(result) - 1:
        cur, nxt = result[i], result[i + 1]
        cur_text = cur.text.replace("\n", " ").strip()
        nxt_text = nxt.text.replace("\n", " ").strip()
        cur_words = cur_text.split()
        nxt_words = nxt_text.split()
        if not cur_words or not nxt_words or cur_text.endswith((".", "!", "?")):
            i += 1
            continue
        last_word = cur_words[-1].strip(".,;:!?")
        first_word = nxt_words[0].strip(".,;:!?")
        if (last_word[:1].isalpha() and last_word[:1].isupper()
                and first_word[:1].isalpha() and first_word[:1].isupper()):
            combined_text = f"{cur_text} {nxt_text}".strip()
            combined_dur = nxt.end - cur.start
            if len(combined_text) <= max_chars and combined_dur <= MAX_CUE_DURATION:
                result[i] = Segment(cur.start, nxt.end, _wrap(combined_text, char_limit))
                del result[i + 1]
                continue
        i += 1
    return result


def _merge_short_cues(
    cues: list[Segment],
    char_limit: int = WRAP_WIDTH,
    min_duration: float = MIN_CUE_DURATION,
    max_duration: float = MAX_CUE_DURATION,
    min_gap: float = None,
) -> list[Segment]:
    """Applique le "Cas 2" du module KTT : un sous-titre trop court (< durée
    min) est d'abord étiré (sans empiéter sur le suivant, en respectant le
    gap minimal), sinon fusionné avec le suivant, sinon avec le précédent —
    dans la limite des caractères/ligne et de la durée max."""
    if min_gap is None:
        min_gap = MIN_GAP
    if not cues:
        return cues

    max_chars = 2 * char_limit
    result: list[Segment] = []
    i = 0
    while i < len(cues):
        seg = cues[i]
        dur = seg.end - seg.start
        if dur >= min_duration:
            result.append(seg)
            i += 1
            continue

        next_seg = cues[i + 1] if i + 1 < len(cues) else None
        ceiling = (next_seg.start - min_gap) if next_seg is not None else (seg.start + min_duration)
        stretched_end = min(seg.start + min_duration, ceiling)
        if stretched_end - seg.start >= min_duration - 1e-6:
            result.append(Segment(seg.start, stretched_end, seg.text))
            i += 1
            continue

        if next_seg is not None:
            combined_text = (seg.text.replace("\n", " ") + " " + next_seg.text.replace("\n", " ")).strip()
            combined_dur = next_seg.end - seg.start
            if len(combined_text) <= max_chars and combined_dur <= max_duration:
                result.append(Segment(seg.start, next_seg.end, _wrap(combined_text, char_limit)))
                i += 2
                continue

        if result:
            prev = result[-1]
            combined_text = (prev.text.replace("\n", " ") + " " + seg.text.replace("\n", " ")).strip()
            combined_dur = seg.end - prev.start
            if len(combined_text) <= max_chars and combined_dur <= max_duration:
                result[-1] = Segment(prev.start, seg.end, _wrap(combined_text, char_limit))
                i += 1
                continue

        # Rien n'est possible sans dépasser les autres contraintes : on garde
        # le sous-titre tel quel (compromis accepté, cf. module KTT §6).
        result.append(seg)
        i += 1
    return result


def _levenshtein(a: str, b: str) -> int:
    """Distance d'édition classique (DP en O(len(a)*len(b))) — utilisée pour
    rapprocher une variante d'orthographe d'un nom propre de sa forme
    canonique. Les glossaires sont courts (dizaines de noms au plus), donc
    pas besoin d'une implémentation plus optimisée."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


_WORD_TOKEN_RE = re.compile(r"\S+")


def _normalize_proper_nouns(
    cues: list[Segment],
    proper_nouns: list[str] | None,
    char_limit: int = WRAP_WIDTH,
) -> list[Segment]:
    """Corrige, par similarité Levenshtein, les variantes d'orthographe d'un
    nom propre attendu (ex. "Martine"/"Martain" -> "Martin") vers la forme
    canonique fournie par l'appelant. Cf. pipeline Master Hub FR V1 §1.10.
    Ne touche qu'aux mots déjà capitalisés (candidats nom propre plausibles)
    à distance ≤ 2 d'un nom du glossaire ; laisse tout le reste intact."""
    nouns = [n.strip() for n in (proper_nouns or []) if n.strip()]
    if not nouns:
        return cues
    canon_by_lower = {n.lower(): n for n in nouns}

    result: list[Segment] = []
    for seg in cues:
        text = seg.text.replace("\n", " ")
        changed = False

        def repl(m: "re.Match[str]") -> str:
            nonlocal changed
            tok = m.group(0)
            core = tok.strip(".,;:!?\"'\u2019")
            if not core or not core[:1].isupper():
                return tok
            low = core.lower()
            if low in canon_by_lower:
                return tok  # déjà correct (à la casse près)
            best: tuple[str, int] | None = None
            for cand_low, canon in canon_by_lower.items():
                if abs(len(cand_low) - len(low)) > 2:
                    continue
                dist = _levenshtein(low, cand_low)
                if dist <= 2 and (best is None or dist < best[1]):
                    best = (canon, dist)
            if best is None:
                return tok
            prefix = tok[: len(tok) - len(tok.lstrip(".,;:!?\"'\u2019"))]
            suffix = tok[len(prefix) + len(core):]
            changed = True
            return prefix + best[0] + suffix

        new_text = _WORD_TOKEN_RE.sub(repl, text)
        if changed:
            new_seg = Segment(seg.start, seg.end, _wrap(new_text, char_limit))
            new_seg.warnings = list(seg.warnings) + ["PROPER_NOUN_NORMALIZED"]
            result.append(new_seg)
        else:
            result.append(seg)
    return result


def _enforce_min_gap(cues: list[Segment], min_gap: float = MIN_GAP) -> list[Segment]:
    """Recule la fin de chaque cue si besoin pour garantir un gap minimal (norme ATAA)
    avec le sous-titre suivant. Ne raccourcit jamais un cue sous sa durée nulle."""
    for i in range(len(cues) - 1):
        gap = cues[i + 1].start - cues[i].end
        if gap < min_gap:
            new_end = cues[i + 1].start - min_gap
            if new_end > cues[i].start:
                cues[i].end = new_end
    return cues


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

