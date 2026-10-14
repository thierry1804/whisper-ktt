"""Cœur de transcription : source (URL ou fichier) -> cues courts -> SRT/VTT."""
import io
import math
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

# Pré-détection de la langue (30/09/2026) : un fichier découpé en tranches a
# besoin d'une langue fixée AVANT de commencer la vraie transcription, pas
# verrouillée au fil de l'eau sur la première tranche assez confiante — cette
# dernière approche a laissé, en prod, un documentaire français majoritairement
# retranscrit en anglais faute de verrou fiable avant la fin du fichier.
# Échantillonne plusieurs points répartis dans le fichier (pas seulement le
# début, souvent musique/silence/générique) et retient le plus confiant.
LANGUAGE_PREPASS_SAMPLES = 3
LANGUAGE_PREPASS_SAMPLE_SEC = 30.0

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
# principe que SUBTITLE_PRESETS) plutôt que côté front KTT : un seul endroit
# à faire évoluer si le phrasing doit être ajusté après des tests réels.
# Prompt de contexte ("Contexte de transcription") — composé dans la langue
# réellement détectée/forcée (cf. _build_initial_prompt), pas toujours en
# anglais : un prompt anglais biaise Whisper vers une sortie en anglais même
# quand la langue est explicitement forcée dans l'autre sens (bug constaté en
# prod le 30/09/2026 sur un documentaire français et un film espagnol,
# entièrement contaminés par le prompt par défaut anglais malgré language="fr"
# forcé). Traductions disponibles pour les langues ci-dessous ;
# PROMPT_FALLBACK_LANG sert pour toute autre langue détectée (couverture
# partielle assumée, à étendre au besoin si un bug est signalé sur une autre
# langue).
PROMPT_FALLBACK_LANG = "en"

SPEECH_TYPE_PROMPTS_BY_LANG = {
    "en": {
        "monologue": "Monologue / voiceover with a single narrator.",
        "dialogue": "Dialogue between two speakers.",
        "group": "Group conversation with three to four speakers.",
        "interview": "Interview with one interviewer and one interviewee.",
        "song": (
            "This is a song with sung lyrics, not spoken dialogue. "
            "Melodic phrasing, repetitions, occasional vocal effects."
        ),
    },
    "fr": {
        "monologue": "Monologue / voix off avec un seul narrateur.",
        "dialogue": "Dialogue entre deux locuteurs.",
        "group": "Conversation de groupe avec trois à quatre locuteurs.",
        "interview": "Interview avec un intervieweur et un interviewé.",
        "song": (
            "Ceci est une chanson avec des paroles chantées, pas un dialogue parlé. "
            "Phrasé mélodique, répétitions, effets vocaux occasionnels."
        ),
    },
    "es": {
        "monologue": "Monólogo / voz en off con un único narrador.",
        "dialogue": "Diálogo entre dos interlocutores.",
        "group": "Conversación de grupo con tres a cuatro interlocutores.",
        "interview": "Entrevista con un entrevistador y un entrevistado.",
        "song": (
            "Esto es una canción con letra cantada, no diálogo hablado. "
            "Fraseo melódico, repeticiones, efectos vocales ocasionales."
        ),
    },
    "de": {
        "monologue": "Monolog / Voiceover mit einem einzigen Erzähler.",
        "dialogue": "Dialog zwischen zwei Sprechern.",
        "group": "Gruppengespräch mit drei bis vier Sprechern.",
        "interview": "Interview mit einem Interviewer und einem Interviewten.",
        "song": (
            "Dies ist ein Lied mit gesungenem Text, kein gesprochener Dialog. "
            "Melodische Phrasierung, Wiederholungen, gelegentliche Gesangseffekte."
        ),
    },
    "it": {
        "monologue": "Monologo / voce fuori campo con un unico narratore.",
        "dialogue": "Dialogo tra due interlocutori.",
        "group": "Conversazione di gruppo con tre o quattro interlocutori.",
        "interview": "Intervista con un intervistatore e un intervistato.",
        "song": (
            "Questa è una canzone con testo cantato, non un dialogo parlato. "
            "Fraseggio melodico, ripetizioni, effetti vocali occasionali."
        ),
    },
    "pt": {
        "monologue": "Monólogo / narração em voz off com um único narrador.",
        "dialogue": "Diálogo entre dois interlocutores.",
        "group": "Conversa em grupo com três a quatro interlocutores.",
        "interview": "Entrevista com um entrevistador e um entrevistado.",
        "song": (
            "Isto é uma música com letra cantada, não um diálogo falado. "
            "Fraseado melódico, repetições, efeitos vocais ocasionais."
        ),
    },
}
DEFAULT_SPEECH_TYPE = "group"

CONTEXT_OPTION_PROMPTS_BY_LANG = {
    "en": {
        "proper_nouns": (
            "Many proper nouns: character names, street names, organizations; "
            "preserve capitalization and spelling."
        ),
        "crude_language": "Strong language is expected and should be transcribed verbatim.",
        "noisy_recording": "Raw on-set recording, background noise, overlapping speech, interruptions.",
        "short_sentences": "Short sentences, clear punctuation, frequent full stops.",
        "long_sentences": "Longer sentences, flowing style, fewer full stops, more commas.",
    },
    "fr": {
        "proper_nouns": (
            "Beaucoup de noms propres : noms de personnages, de rues, d'organisations ; "
            "préserver la casse et l'orthographe."
        ),
        "crude_language": "Un langage grossier est attendu et doit être transcrit tel quel.",
        "noisy_recording": (
            "Enregistrement brut sur le tournage, bruit de fond, paroles qui se chevauchent, interruptions."
        ),
        "short_sentences": "Phrases courtes, ponctuation claire, points fréquents.",
        "long_sentences": "Phrases plus longues, style fluide, moins de points, plus de virgules.",
    },
    "es": {
        "proper_nouns": (
            "Muchos nombres propios: nombres de personajes, calles, organizaciones; "
            "conservar mayúsculas y ortografía."
        ),
        "crude_language": "Se espera lenguaje soez y debe transcribirse literalmente.",
        "noisy_recording": "Grabación en bruto de rodaje, ruido de fondo, voces superpuestas, interrupciones.",
        "short_sentences": "Frases cortas, puntuación clara, puntos frecuentes.",
        "long_sentences": "Frases más largas, estilo fluido, menos puntos, más comas.",
    },
    "de": {
        "proper_nouns": (
            "Viele Eigennamen: Figurennamen, Straßennamen, Organisationen; "
            "Groß-/Kleinschreibung und Schreibweise beibehalten."
        ),
        "crude_language": "Derbe Sprache ist zu erwarten und sollte wörtlich transkribiert werden.",
        "noisy_recording": (
            "Rohe Aufnahme vom Set, Hintergrundgeräusche, sich überlappende Sprache, Unterbrechungen."
        ),
        "short_sentences": "Kurze Sätze, klare Zeichensetzung, häufige Punkte.",
        "long_sentences": "Längere Sätze, flüssiger Stil, weniger Punkte, mehr Kommas.",
    },
    "it": {
        "proper_nouns": (
            "Molti nomi propri: nomi di personaggi, vie, organizzazioni; "
            "mantenere maiuscole e ortografia."
        ),
        "crude_language": "Si prevede un linguaggio volgare, da trascrivere testualmente.",
        "noisy_recording": "Registrazione grezza sul set, rumore di fondo, voci sovrapposte, interruzioni.",
        "short_sentences": "Frasi brevi, punteggiatura chiara, punti frequenti.",
        "long_sentences": "Frasi più lunghe, stile scorrevole, meno punti, più virgole.",
    },
    "pt": {
        "proper_nouns": (
            "Muitos nomes próprios: nomes de personagens, ruas, organizações; "
            "preservar maiúsculas e ortografia."
        ),
        "crude_language": "Espera-se linguagem chula, que deve ser transcrita literalmente.",
        "noisy_recording": "Gravação bruta de set, ruído de fundo, falas sobrepostas, interrupções.",
        "short_sentences": "Frases curtas, pontuação clara, pontos finais frequentes.",
        "long_sentences": "Frases mais longas, estilo fluido, menos pontos finais, mais vírgulas.",
    },
}

PROPER_NOUNS_PHRASE_BY_LANG = {
    "en": "Many proper nouns: {names}; preserve capitalization and spelling.",
    "fr": "Beaucoup de noms propres : {names} ; préserver la casse et l'orthographe.",
    "es": "Muchos nombres propios: {names}; conservar mayúsculas y ortografía.",
    "de": "Viele Eigennamen: {names}; Groß-/Kleinschreibung und Schreibweise beibehalten.",
    "it": "Molti nomi propri: {names}; mantenere maiuscole e ortografia.",
    "pt": "Muitos nomes próprios: {names}; preservar maiúsculas e ortografia.",
}

REGISTER_PROMPTS_BY_LANG = {
    "en": {
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
    },
    "fr": {
        "urban": (
            "Ceci est un film se déroulant en milieu urbain, grande ville. "
            "Langage de rue, registre familier à très familier, argot urbain, "
            "phrases courtes, contractions fréquentes, phrasé oral. "
            "Ton direct, parfois tendu."
        ),
        "suburb": (
            "Ceci est un film se déroulant dans des quartiers populaires de banlieue. "
            "Registre très familier, argot urbain spécifique, phrases hachées, "
            "contractions, phrases inachevées, langage grossier occasionnel. "
            "Ton franc, parfois conflictuel."
        ),
        "rural": (
            "Ceci est un film se déroulant en milieu rural, à la campagne. "
            "Registre familier à courant, expressions régionales possibles, "
            "rythme plus lent, phrases parfois incomplètes, pas de langage soutenu. "
            "Ton simple et direct."
        ),
        "bourgeois": (
            "Ceci est un film se déroulant dans un milieu aisé, bourgeois. "
            "Registre soutenu, phrases complètes, vocabulaire recherché, "
            "pas d'argot ni de langage grossier. "
            "Ton sérieux ou posé, dialogues intenses mais bien construits."
        ),
        "standard": (
            "Ceci est un film avec un dialogue courant, en langage standard. "
            "Registre neutre, phrases parfois incomplètes, contractions légères, "
            "pas d'argot marqué. "
            "Ton neutre, ni trop soutenu ni trop familier."
        ),
        "casual": (
            "Ceci est un film avec un dialogue quotidien entre proches. "
            "Registre familier, phrases courtes, contractions fréquentes, "
            "phrasé oral, pas de langage grossier. "
            "Ton naturel, détendu ou intime."
        ),
        "crude": (
            "Ceci est un film avec un dialogue cru et grossier. "
            "Registre très familier, langage grossier fréquent, phrases courtes, "
            "ton direct et parfois agressif, pas de langage soutenu. "
            "Milieu urbain ou populaire."
        ),
        "historical": (
            "Ceci est un film historique avec un langage soutenu. "
            "Registre soutenu, phrases complètes, vocabulaire précis, "
            "pas d'argot moderne, ton sérieux et posé. "
            "Références à des événements ou personnages d'époque."
        ),
        "comedic": (
            "Ceci est une comédie avec un dialogue humoristique. "
            "Registre familier, jeux de mots, expressions courantes, phrases courtes, "
            "ton léger et amical. "
            "Pas de langage soutenu, expressions argotiques occasionnelles."
        ),
        "thriller": (
            "Ceci est un film policier / polar noir. "
            "Langage direct, registre familier à très familier, phrases courtes, "
            "ton sec et tendu, langage grossier occasionnel. "
            "Milieu urbain sombre, dialogue d'enquête ou criminel."
        ),
        "intimate_drama": (
            "Ceci est un drame intimiste avec un dialogue entre proches. "
            "Registre familier, phrases parfois brisées par l'émotion, "
            "ton sérieux et sincère, pas de langage soutenu. "
            "Scènes de conflit ou de révélation personnelle."
        ),
        "youth": (
            "Ceci est un film centré sur des adolescents. "
            "Registre très familier, argot jeune, phrases courtes, contractions, "
            "phrasé oral, langage grossier occasionnel. "
            "Ton spontané, parfois provocateur ou ironique."
        ),
    },
    "es": {
        "urban": (
            "Esta es una película ambientada en un entorno urbano, gran ciudad. "
            "Lenguaje de calle, registro informal a muy informal, argot urbano, "
            "frases cortas, contracciones frecuentes, fraseo oral. "
            "Tono directo, a veces tenso."
        ),
        "suburb": (
            "Esta es una película ambientada en barrios populares de periferia. "
            "Registro muy informal, argot urbano específico, frases entrecortadas, "
            "contracciones, frases inacabadas, lenguaje soez ocasional. "
            "Tono franco, a veces conflictivo."
        ),
        "rural": (
            "Esta es una película ambientada en un entorno rural, en el campo. "
            "Registro informal a coloquial, posibles expresiones regionales, "
            "ritmo más lento, frases a veces incompletas, sin lenguaje formal. "
            "Tono sencillo y directo."
        ),
        "bourgeois": (
            "Esta es una película ambientada en un entorno acomodado, burgués. "
            "Registro formal, frases completas, vocabulario cuidado, "
            "sin argot ni lenguaje soez. "
            "Tono serio o mesurado, diálogos intensos pero bien construidos."
        ),
        "standard": (
            "Esta es una película con diálogo cotidiano, en lenguaje estándar. "
            "Registro neutro, frases a veces incompletas, contracciones ligeras, "
            "sin argot marcado. "
            "Tono neutro, ni muy formal ni muy informal."
        ),
        "casual": (
            "Esta es una película con diálogo cotidiano entre allegados. "
            "Registro informal, frases cortas, contracciones frecuentes, "
            "fraseo oral, sin lenguaje soez. "
            "Tono natural, relajado o íntimo."
        ),
        "crude": (
            "Esta es una película con diálogo crudo y grosero. "
            "Registro muy informal, lenguaje soez frecuente, frases cortas, "
            "tono directo y a veces agresivo, sin lenguaje formal. "
            "Entorno urbano o popular."
        ),
        "historical": (
            "Esta es una película histórica con lenguaje formal. "
            "Registro formal, frases completas, vocabulario preciso, "
            "sin argot moderno, tono serio y mesurado. "
            "Referencias a eventos o personajes de época."
        ),
        "comedic": (
            "Esta es una comedia con diálogo humorístico. "
            "Registro informal, juegos de palabras, expresiones comunes, frases cortas, "
            "tono ligero y amistoso. "
            "Sin lenguaje formal, expresiones de argot ocasionales."
        ),
        "thriller": (
            "Esta es una película policiaca / cine negro. "
            "Lenguaje directo, registro informal a muy informal, frases cortas, "
            "tono seco y tenso, lenguaje soez ocasional. "
            "Entorno urbano oscuro, diálogo de investigación o criminal."
        ),
        "intimate_drama": (
            "Este es un drama íntimo con diálogo entre allegados. "
            "Registro informal, frases a veces quebradas por la emoción, "
            "tono serio y sincero, sin lenguaje formal. "
            "Escenas de conflicto o revelación personal."
        ),
        "youth": (
            "Esta es una película centrada en adolescentes. "
            "Registro muy informal, argot juvenil, frases cortas, contracciones, "
            "fraseo oral, lenguaje soez ocasional. "
            "Tono espontáneo, a veces provocador o irónico."
        ),
    },
    "de": {
        "urban": (
            "Dies ist ein Film, der in einer urbanen Umgebung, einer Großstadt, spielt. "
            "Straßensprache, lockeres bis sehr lockeres Register, urbaner Slang, "
            "kurze Sätze, häufige Verkürzungen, mündliche Sprechweise. "
            "Direkter Ton, manchmal angespannt."
        ),
        "suburb": (
            "Dies ist ein Film, der in Arbeiterviertel-Vororten spielt. "
            "Sehr lockeres Register, spezifischer urbaner Slang, abgehackte Sätze, "
            "Verkürzungen, unvollständige Sätze, gelegentlich derbe Sprache. "
            "Offener Ton, manchmal konfrontativ."
        ),
        "rural": (
            "Dies ist ein Film, der in ländlicher Umgebung spielt. "
            "Lockeres bis vertrautes Register, mögliche regionale Ausdrücke, "
            "langsameres Tempo, manchmal unvollständige Sätze, keine gehobene Sprache. "
            "Einfacher und direkter Ton."
        ),
        "bourgeois": (
            "Dies ist ein Film, der in einem gehobenen, bürgerlichen Milieu spielt. "
            "Gehobenes Register, vollständige Sätze, gepflegter Wortschatz, "
            "kein Slang oder derbe Sprache. "
            "Ernster oder gefasster Ton, intensive, aber wohlgeformte Dialoge."
        ),
        "standard": (
            "Dies ist ein Film mit alltäglichem Dialog in Standardsprache. "
            "Neutrales Register, manchmal unvollständige Sätze, leichte Verkürzungen, "
            "kein auffälliger Slang. "
            "Neutraler Ton, weder sehr gehoben noch sehr locker."
        ),
        "casual": (
            "Dies ist ein Film mit alltäglichem Dialog zwischen nahestehenden Personen. "
            "Lockeres Register, kurze Sätze, häufige Verkürzungen, "
            "mündliche Sprechweise, keine derbe Sprache. "
            "Natürlicher, entspannter oder vertrauter Ton."
        ),
        "crude": (
            "Dies ist ein Film mit derbem, grobem Dialog. "
            "Sehr lockeres Register, häufig derbe Sprache, kurze Sätze, "
            "direkter und manchmal aggressiver Ton, keine gehobene Sprache. "
            "Urbanes oder Arbeiter-Milieu."
        ),
        "historical": (
            "Dies ist ein historischer Film mit gehobener Sprache. "
            "Gehobenes Register, vollständige Sätze, präziser Wortschatz, "
            "kein moderner Slang, ernster und gefasster Ton. "
            "Bezüge zu Ereignissen oder Figuren der jeweiligen Epoche."
        ),
        "comedic": (
            "Dies ist eine Komödie mit humorvollem Dialog. "
            "Lockeres Register, Wortspiele, gängige Ausdrücke, kurze Sätze, "
            "leichter und freundlicher Ton. "
            "Keine gehobene Sprache, gelegentliche umgangssprachliche Ausdrücke."
        ),
        "thriller": (
            "Dies ist ein Kriminalfilm / Noir. "
            "Direkte Sprache, lockeres bis sehr lockeres Register, kurze Sätze, "
            "trockener und angespannter Ton, gelegentlich derbe Sprache. "
            "Düsteres urbanes Milieu, ermittelnder oder krimineller Dialog."
        ),
        "intimate_drama": (
            "Dies ist ein intimes Drama mit Dialog zwischen nahestehenden Personen. "
            "Lockeres Register, Sätze manchmal von Emotionen unterbrochen, "
            "ernster und aufrichtiger Ton, keine gehobene Sprache. "
            "Szenen von Konflikt oder persönlicher Enthüllung."
        ),
        "youth": (
            "Dies ist ein Film über Teenager. "
            "Sehr lockeres Register, Jugendslang, kurze Sätze, Verkürzungen, "
            "mündliche Sprechweise, gelegentlich derbe Sprache. "
            "Spontaner Ton, manchmal provokant oder ironisch."
        ),
    },
    "it": {
        "urban": (
            "Questo è un film ambientato in un contesto urbano, grande città. "
            "Linguaggio di strada, registro informale o molto informale, slang urbano, "
            "frasi brevi, contrazioni frequenti, fraseggio orale. "
            "Tono diretto, a volte teso."
        ),
        "suburb": (
            "Questo è un film ambientato in quartieri popolari di periferia. "
            "Registro molto informale, slang urbano specifico, frasi spezzate, "
            "contrazioni, frasi incompiute, linguaggio volgare occasionale. "
            "Tono franco, a volte conflittuale."
        ),
        "rural": (
            "Questo è un film ambientato in un contesto rurale, in campagna. "
            "Registro informale o colloquiale, possibili espressioni regionali, "
            "ritmo più lento, frasi a volte incomplete, nessun linguaggio formale. "
            "Tono semplice e diretto."
        ),
        "bourgeois": (
            "Questo è un film ambientato in un ambiente agiato, borghese. "
            "Registro formale, frasi complete, vocabolario curato, "
            "nessuno slang o linguaggio volgare. "
            "Tono serio o composto, dialoghi intensi ma ben costruiti."
        ),
        "standard": (
            "Questo è un film con dialogo quotidiano in lingua standard. "
            "Registro neutro, frasi a volte incomplete, lievi contrazioni, "
            "nessuno slang marcato. "
            "Tono neutro, né troppo formale né troppo informale."
        ),
        "casual": (
            "Questo è un film con dialogo quotidiano tra persone vicine. "
            "Registro informale, frasi brevi, contrazioni frequenti, "
            "fraseggio orale, nessun linguaggio volgare. "
            "Tono naturale, rilassato o intimo."
        ),
        "crude": (
            "Questo è un film con dialogo crudo e volgare. "
            "Registro molto informale, linguaggio volgare frequente, frasi brevi, "
            "tono diretto e a volte aggressivo, nessun linguaggio formale. "
            "Ambiente urbano o popolare."
        ),
        "historical": (
            "Questo è un film storico con linguaggio formale. "
            "Registro formale, frasi complete, vocabolario preciso, "
            "nessuno slang moderno, tono serio e composto. "
            "Riferimenti a eventi o personaggi d'epoca."
        ),
        "comedic": (
            "Questa è una commedia con dialogo umoristico. "
            "Registro informale, giochi di parole, espressioni comuni, frasi brevi, "
            "tono leggero e amichevole. "
            "Nessun linguaggio formale, espressioni gergali occasionali."
        ),
        "thriller": (
            "Questo è un film poliziesco / noir. "
            "Linguaggio diretto, registro informale o molto informale, frasi brevi, "
            "tono secco e teso, linguaggio volgare occasionale. "
            "Ambiente urbano cupo, dialogo investigativo o criminale."
        ),
        "intimate_drama": (
            "Questo è un dramma intimo con dialogo tra persone vicine. "
            "Registro informale, frasi a volte spezzate dall'emozione, "
            "tono serio e sincero, nessun linguaggio formale. "
            "Scene di conflitto o rivelazione personale."
        ),
        "youth": (
            "Questo è un film incentrato sugli adolescenti. "
            "Registro molto informale, slang giovanile, frasi brevi, contrazioni, "
            "fraseggio orale, linguaggio volgare occasionale. "
            "Tono spontaneo, a volte provocatorio o ironico."
        ),
    },
    "pt": {
        "urban": (
            "Este é um filme ambientado num contexto urbano, grande cidade. "
            "Linguagem de rua, registo informal a muito informal, gíria urbana, "
            "frases curtas, contrações frequentes, fraseado oral. "
            "Tom direto, por vezes tenso."
        ),
        "suburb": (
            "Este é um filme ambientado em bairros populares de periferia. "
            "Registo muito informal, gíria urbana específica, frases cortadas, "
            "contrações, frases inacabadas, linguagem chula ocasional. "
            "Tom franco, por vezes conflituoso."
        ),
        "rural": (
            "Este é um filme ambientado num contexto rural, no campo. "
            "Registo informal a coloquial, possíveis expressões regionais, "
            "ritmo mais lento, frases por vezes incompletas, sem linguagem formal. "
            "Tom simples e direto."
        ),
        "bourgeois": (
            "Este é um filme ambientado num meio abastado, burguês. "
            "Registo formal, frases completas, vocabulário cuidado, "
            "sem gíria nem linguagem chula. "
            "Tom sério ou ponderado, diálogos intensos mas bem construídos."
        ),
        "standard": (
            "Este é um filme com diálogo do dia a dia, em linguagem padrão. "
            "Registo neutro, frases por vezes incompletas, contrações ligeiras, "
            "sem gíria marcada. "
            "Tom neutro, nem muito formal nem muito informal."
        ),
        "casual": (
            "Este é um filme com diálogo do dia a dia entre pessoas próximas. "
            "Registo informal, frases curtas, contrações frequentes, "
            "fraseado oral, sem linguagem chula. "
            "Tom natural, descontraído ou íntimo."
        ),
        "crude": (
            "Este é um filme com diálogo cru e grosseiro. "
            "Registo muito informal, linguagem chula frequente, frases curtas, "
            "tom direto e por vezes agressivo, sem linguagem formal. "
            "Ambiente urbano ou popular."
        ),
        "historical": (
            "Este é um filme histórico com linguagem formal. "
            "Registo formal, frases completas, vocabulário preciso, "
            "sem gíria moderna, tom sério e ponderado. "
            "Referências a eventos ou personagens da época."
        ),
        "comedic": (
            "Esta é uma comédia com diálogo humorístico. "
            "Registo informal, jogos de palavras, expressões comuns, frases curtas, "
            "tom leve e amigável. "
            "Sem linguagem formal, expressões de gíria ocasionais."
        ),
        "thriller": (
            "Este é um filme policial / noir. "
            "Linguagem direta, registo informal a muito informal, frases curtas, "
            "tom seco e tenso, linguagem chula ocasional. "
            "Ambiente urbano sombrio, diálogo de investigação ou criminal."
        ),
        "intimate_drama": (
            "Este é um drama íntimo com diálogo entre pessoas próximas. "
            "Registo informal, frases por vezes quebradas pela emoção, "
            "tom sério e sincero, sem linguagem formal. "
            "Cenas de conflito ou revelação pessoal."
        ),
        "youth": (
            "Este é um filme centrado em adolescentes. "
            "Registo muito informal, gíria jovem, frases curtas, contrações, "
            "fraseado oral, linguagem chula ocasional. "
            "Tom espontâneo, por vezes provocador ou irónico."
        ),
    },
}
DEFAULT_REGISTER = "standard"

# Ensembles de clés valides (identiques dans toutes les langues — seules les
# valeurs/traductions changent) : utilisés pour la validation côté API.
VALID_SPEECH_TYPES = frozenset(SPEECH_TYPE_PROMPTS_BY_LANG["en"])
VALID_CONTEXT_OPTIONS = frozenset(CONTEXT_OPTION_PROMPTS_BY_LANG["en"])
VALID_REGISTERS = frozenset(REGISTER_PROMPTS_BY_LANG["en"])

# Cap défensif : Whisper tronque de toute façon en interne si le prompt
# dépasse sa fenêtre de contexte, mais on évite d'envoyer une chaîne absurde
# au modèle si un appelant mal formé envoyait un pavé de texte.
INITIAL_PROMPT_MAX_CHARS = 1000


def _build_initial_prompt(
    speech_type: str | None,
    context_options: list[str] | None,
    register: str | None,
    proper_nouns: list[str] | None = None,
    prompt_language: str = PROMPT_FALLBACK_LANG,
) -> str:
    """Compose le initial_prompt Whisper à partir de valeurs catégorielles
    (registre, type de parole, options de contexte) — cf. spec KTT "Contexte
    de transcription". Toujours renvoie une chaîne non vide (au moins le
    registre + type de parole par défaut), même si l'appelant ne renseigne
    aucun champ : décision produit explicite, le prompt par défaut doit
    s'appliquer à tous les jobs.

    prompt_language : compose le prompt DANS la langue réellement détectée
    ou forcée (cf. Transcriber._collect_words), pas toujours en anglais — un
    prompt anglais biaise Whisper vers une sortie en anglais même quand
    `language` est forcé dans l'autre sens (bug constaté en prod le
    30/09/2026 : documentaire français et film espagnol entièrement
    retranscrits en anglais malgré la langue forcée côté décodage). Retombe
    sur PROMPT_FALLBACK_LANG si aucune traduction n'est disponible pour
    cette langue.

    proper_nouns (optionnel) : liste de noms propres attendus dans l'audio
    (cf. pipeline Master Hub FR V1 §1.10). Quand fournie, remplace la phrase
    générique de l'option "proper_nouns" par une phrase citant les noms
    eux-mêmes — Whisper conditionne sur le texte du prompt, donc voir les
    noms réels biaise bien plus efficacement l'orthographe que la phrase
    générique seule. Plafonnée à 10 noms pour rester dans le budget de
    tokens de Whisper (~224)."""
    lang_key = prompt_language if prompt_language in REGISTER_PROMPTS_BY_LANG else PROMPT_FALLBACK_LANG
    register_prompts = REGISTER_PROMPTS_BY_LANG[lang_key]
    speech_prompts = SPEECH_TYPE_PROMPTS_BY_LANG[lang_key]
    context_prompts = CONTEXT_OPTION_PROMPTS_BY_LANG[lang_key]
    proper_nouns_phrase = PROPER_NOUNS_PHRASE_BY_LANG.get(
        lang_key, PROPER_NOUNS_PHRASE_BY_LANG[PROMPT_FALLBACK_LANG]
    )

    register_key = register if register in register_prompts else DEFAULT_REGISTER
    speech_key = speech_type if speech_type in speech_prompts else DEFAULT_SPEECH_TYPE

    parts = [register_prompts[register_key], speech_prompts[speech_key]]

    options = set(context_options or [])
    if "short_sentences" in options and "long_sentences" in options:
        options.discard("long_sentences")  # priorité aux phrases courtes
    nouns = [n.strip() for n in (proper_nouns or []) if n.strip()]
    if nouns:
        options.discard("proper_nouns")  # la liste réelle remplace la phrase générique
    for key in ("proper_nouns", "crude_language", "noisy_recording", "short_sentences", "long_sentences"):
        if key in options:
            parts.append(context_prompts[key])
    if nouns:
        names = ", ".join(nouns[:10])
        parts.append(proper_nouns_phrase.format(names=names))

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
# Repli uniquement : la cadence réelle est désormais détectée par job depuis
# les métadonnées du média (cf. Transcriber._detect_frame_rate, 05/10/2026,
# décision client "s'adapter à la cadence du film... idéalement prendre la
# métadonnée dans le média" — cinéma 24/25/48 i/s, TV 25/29,98/50/59,97 i/s).
# Ce repli ne s'applique que si la détection échoue (pas de flux vidéo, ex.
# un .wav pur ; ffprobe indisponible ; métadonnée illisible).
FRAME_RATE = 25.0
MIN_GAP_FRAMES = 4        # gap minimal ATAA entre 2 sous-titres (6 sur changement de plan,
                          # non applicable ici : pas d'analyse vidéo/scène dans ce pipeline audio)
MIN_GAP = MIN_GAP_FRAMES / FRAME_RATE
_FRAME_RATE_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")

# Vitesse de lecture cible ATAA : 14-15 c/s en moyenne, 17 c/s = trop rapide.
# N'est PAS utilisé comme déclencheur de découpage (testé le 18/09/2026 :
# re-couper le texte ne réduit pas la densité de parole sous-jacente, ça
# fragmente sans améliorer le taux de conformité) — sert uniquement à
# journaliser un warning QC (CPS_ABOVE_WARNING) sur le cue final.
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
# Catalogue des presets de sous-titres — remplace OUTPUT_PROFILES (05/10/2026).
# Source : fiche Master Hub "master-hub-fiche-presets-sous-titres.pdf" v1.0,
# reprise telle quelle pour les 8 presets ci-dessous (valeurs JSON de
# référence §6 de la fiche). Décision client (05/10/2026) : ces 8 presets
# REMPLACENT les 4 anciens output_profile (tv/video/streaming/cinema), SAUF
# "video" qui est conservé à l'identique (demande explicite du client) — ses
# champs hors char/ligne reprennent les constantes globales historiques
# (MAX_CPS=17, MAX_CUE_DURATION=5.0, MIN_GAP_FRAMES=4, ~MIN_CUE_DURATION
# via 16 images) pour ne rien changer à son comportement actuel.
#
# Champs non repris de la fiche (hors périmètre de ce pipeline, cf. analyse
# du 29/09/2026) :
# - shot_change_clearance_frames : suppose des timestamps de plan vidéo,
#   indisponibles (pipeline audio seul) — mis de côté par le client, version
#   future. SHOT_CHANGE_TOO_CLOSE reste définitivement "non disponible".
# - text_encoding : toujours UTF-8 par construction (Python 3) — rien à
#   vérifier, INVALID_TEXT_ENCODING ne peut structurellement pas se produire.
#
# cps_min/cps_max = target_cps.minimum/maximum de la fiche. last_subtitle_min_sec
# réutilise la même valeur que first_subtitle_min_sec, symétriquement (décision
# client du 05/10/2026 : "les deux contrôles combinés" pour la fin de
# programme, pas de champ dédié fourni par la fiche).
SUBTITLE_PRESETS: dict[str, dict] = {
    "dcp_smpte_bv21_technical": dict(
        max_lines=3, max_chars=52, cps_min=14, cps_max=18, cps_warning=18,
        min_duration_frames=15, max_duration_sec=7, min_gap_frames=2,
        first_subtitle_min_sec=4,
    ),
    "dcp_cinema_fr_ktt": dict(
        max_lines=2, max_chars=40, cps_min=14, cps_max=15, cps_warning=15,
        min_duration_frames=16, max_duration_sec=5, min_gap_frames=4,
        first_subtitle_min_sec=4,
    ),
    "tv_fr_ktt": dict(
        max_lines=2, max_chars=37, cps_min=14, cps_max=15, cps_warning=15,
        min_duration_frames=16, max_duration_sec=5, min_gap_frames=4,
        first_subtitle_min_sec=0,
    ),
    "vod_standard_ktt": dict(
        max_lines=2, max_chars=42, cps_min=14, cps_max=17, cps_warning=17,
        min_duration_frames=15, max_duration_sec=7, min_gap_frames=2,
        first_subtitle_min_sec=0,
    ),
    "dcp_sme_fr_ktt": dict(
        max_lines=3, max_chars=32, cps_min=14, cps_max=15, cps_warning=15,
        min_duration_frames=15, max_duration_sec=5, min_gap_frames=2,
        first_subtitle_min_sec=4,
    ),
    "tv_sme_fr_ktt": dict(
        max_lines=2, max_chars=37, cps_min=14, cps_max=15, cps_warning=15,
        min_duration_frames=16, max_duration_sec=5, min_gap_frames=4,
        first_subtitle_min_sec=0,
    ),
    "vod_sdh_sme_ktt": dict(
        max_lines=2, max_chars=42, cps_min=14, cps_max=17, cps_warning=17,
        min_duration_frames=15, max_duration_sec=7, min_gap_frames=2,
        first_subtitle_min_sec=0,
    ),
    "transcription_tolerance_max_ktt": dict(
        max_lines=3, max_chars=52, cps_min=14, cps_max=18, cps_warning=18,
        min_duration_frames=15, max_duration_sec=7, min_gap_frames=2,
        first_subtitle_min_sec=0,
    ),
    # Conservé tel quel (demande explicite du client, 05/10/2026) — seul
    # profil non remplacé par la fiche. char/ligne inchangé (39) ; les autres
    # champs reprennent les anciennes constantes globales, identiques pour
    # tous les profils avant cette migration.
    "video": dict(
        max_lines=2, max_chars=39, cps_min=14, cps_max=15, cps_warning=17,
        min_duration_frames=16, max_duration_sec=5.0, min_gap_frames=4,
        first_subtitle_min_sec=0,
    ),
}
DEFAULT_SUBTITLE_PRESET = "video"

# VAD : moins agressif = moins de risque de rogner la parole ----------------
# Désactivé le 01/10/2026 : le VAD (Silero, via faster-whisper) classe parfois
# à tort de la parole réelle comme silence et la fait disparaître ENTIÈREMENT
# avant même d'atteindre Whisper — bug bien plus grave qu'une hallucination
# (du contenu perdu silencieusement, invisible sans comparer à la source).
# Observé en prod : ~18s de narration réelle ("un jour, je suis parti. Pour
# un très long voyage. Pour... Je ne sais plus.") disparue avec le VAD actif,
# aucun réglage de min_silence_duration_ms/speech_pad_ms testé ne corrige le
# cas. Par ailleurs le VAD ne réduit MÊME PAS le risque d'hallucination sur
# le silence : la même hallucination ("Sous-titrage ST' 501") apparaît avec
# ou sans VAD sur ce fichier — aucun bénéfice compensatoire constaté. Les
# seuils de qualité Whisper (no_speech_threshold, compression_ratio_threshold,
# log_prob_threshold) restent actifs et jouent déjà ce rôle de filtrage.
VAD_ENABLED = False
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
    # Confiance du décodeur pour ce mot (0-1, exposée par faster-whisper via
    # word_timestamps=True) — défaut 1.0 pour les rares Word construits sans
    # cette donnée (ne doit jamais faire paraître un mot comme peu fiable par
    # défaut). Cf. _compute_confidence (05/10/2026 -> 07/10/2026, ajouté sur
    # demande client : score de confiance affiché par cue côté front KTT).
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    warnings: list[str] = field(default_factory=list)  # log QC (cf. module KTT)
    # Score de confiance moyen (0-1) calculé après coup par _compute_confidence
    # à partir des Word d'origine recouvrant ce cue — None pour un cue ajouté
    # hors pipeline (ex. le filigrane, ajouté côté app.py après transcribe()).
    confidence: float | None = None


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

    @staticmethod
    def _detect_frame_rate(source: str) -> float:
        """Cadence réelle du média (images/seconde), lue depuis les
        métadonnées du flux vidéo via ffprobe — remplace l'hypothèse fixe à
        25fps (05/10/2026, décision client : "s'adapter à la cadence du
        film... idéalement prendre la métadonnée dans le média" ; valeurs
        attendues cinéma 24/25/48 i/s, TV 25/29,98/50/59,97 i/s, mais la
        valeur exacte lue dans le fichier est utilisée telle quelle, jamais
        arrondie à l'une de ces catégories).

        Repli sur FRAME_RATE (25.0, comportement historique) dans tous les
        cas d'échec — ne doit jamais bloquer une transcription : pas de flux
        vidéo (ex. un .wav pur), ffprobe indisponible/timeout, métadonnée
        absente ou illisible. r_frame_rate (cadence nominale du flux) est
        préféré ; avg_frame_rate (cadence moyenne mesurée) sert de repli
        pour un flux où r_frame_rate serait "0/0"."""
        try:
            proc = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate,avg_frame_rate",
                    "-of", "default=noprint_wrappers=1",
                    source,
                ],
                capture_output=True, text=True, timeout=20,
            )
        except (subprocess.TimeoutExpired, OSError):
            return FRAME_RATE
        if proc.returncode != 0:
            return FRAME_RATE
        values: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            key, sep, val = line.partition("=")
            if sep:
                values[key.strip()] = val.strip()
        for key in ("r_frame_rate", "avg_frame_rate"):
            m = _FRAME_RATE_FRACTION_RE.match(values.get(key, ""))
            if not m:
                continue
            num, den = int(m.group(1)), int(m.group(2))
            if num > 0 and den > 0:
                return num / den
        return FRAME_RATE

    def _detect_language(self, audio: bytes, total_duration: float) -> str | None:
        """Détecte la langue à partir de plusieurs échantillons répartis dans
        le fichier plutôt que la seule première tranche séquentielle (souvent
        peu représentative — générique, musique, silence). Renvoie la langue
        au plus haut score de confiance parmi les échantillons testés."""
        n = LANGUAGE_PREPASS_SAMPLES
        sample_len = min(LANGUAGE_PREPASS_SAMPLE_SEC, total_duration)
        best_lang: str | None = None
        best_prob = -1.0
        for i in range(n):
            center = (i + 1) / (n + 1) * total_duration
            start = max(0.0, min(center - sample_len / 2, total_duration - sample_len))
            sample = _slice_wav(audio, start, start + sample_len)
            _, info = self.model.transcribe(
                io.BytesIO(sample),
                language=None,
                language_detection_threshold=LANGUAGE_DETECTION_THRESHOLD,
                language_detection_segments=LANGUAGE_DETECTION_SEGMENTS,
                vad_filter=VAD_ENABLED,
                vad_parameters=VAD_PARAMETERS if VAD_ENABLED else None,
            )
            if info.language_probability > best_prob:
                best_lang, best_prob = info.language, info.language_probability
        return best_lang

    def _collect_words(
        self,
        audio: bytes,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
        speech_type: str | None = None,
        context_options: list[str] | None = None,
        register: str | None = None,
        proper_nouns: list[str] | None = None,
        language: str | None = None,
    ) -> tuple[list[Word], str, float, str]:
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
        # Langue déterminée AVANT de composer le prompt et de commencer la
        # vraie transcription — jamais après coup : un prompt composé dans la
        # mauvaise langue (ex. anglais par défaut sur un fichier français)
        # biaise Whisper vers une sortie dans CETTE langue, même avec
        # `language` forcé dans l'autre sens (bug constaté en prod le
        # 30/09/2026). Appliqué uniformément, y compris aux fichiers à
        # tranche unique — un fichier court non anglophone est tout aussi
        # exposé au biais.
        if language is not None:
            detected_language: str | None = language
        else:
            detected_language = self._detect_language(audio, total_duration)

        resolved_prompt = _build_initial_prompt(
            speech_type, context_options, register, proper_nouns,
            prompt_language=detected_language or PROMPT_FALLBACK_LANG,
        )

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
                # Langue déjà déterminée en amont (forcée ou pré-détectée) et
                # appliquée identiquement à TOUTES les tranches — plus de
                # verrouillage au fil de l'eau sur la 1ère tranche assez
                # confiante (fragile : voir _detect_language).
                chunk_language = detected_language

                segments, info = self.model.transcribe(
                    io.BytesIO(chunk_audio),
                    language=chunk_language,
                    language_detection_threshold=LANGUAGE_DETECTION_THRESHOLD,
                    language_detection_segments=LANGUAGE_DETECTION_SEGMENTS,
                    vad_filter=VAD_ENABLED,
                    vad_parameters=VAD_PARAMETERS if VAD_ENABLED else None,
                    word_timestamps=True,
                    initial_prompt=resolved_prompt,
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
                    # Filet de sécurité : la pré-passe n'a jamais atteint de
                    # confiance suffisante (très rare — fichier quasi muet).
                    # On retient au moins la détection de cette tranche pour
                    # le champ `language` retourné, à titre indicatif.
                    detected_language = info.language

                lower = boundaries[idx - 1] if idx > 0 else None
                upper = boundaries[idx] if idx < len(boundaries) else None

                for seg in segments:          # itérer déclenche le calcul
                    if cancel_event is not None and cancel_event.is_set():
                        raise Cancelled("Transcription annulée")
                    # Appartenance à la tranche décidée UNIQUEMENT sur g_start (jamais
                    # g_end) pour les deux bornes : partition stricte et sans recouvrement
                    # du flux de mots entre tranches voisines. Un mot dont la fin dépasse
                    # la frontière (g_start < frontière <= g_end) reste entièrement dans
                    # SA tranche — le couper au milieu casserait son propre timestamp — mais
                    # ne doit jamais être accepté par LES DEUX tranches à la fois : utiliser
                    # g_end pour une borne et g_start pour l'autre (bug corrigé le 30/09/2026)
                    # laissait passer ce cas, produisant un mot en double avec un timing
                    # légèrement désynchronisé d'une tranche à l'autre — donc un chevauchement
                    # réel entre deux cues une fois reconstruits (ex. bloc 446 signalé en prod).
                    if seg.words:
                        for w in seg.words:
                            g_start = w.start + chunk_start
                            g_end = min(w.end, w.start + WORD_MAX_DURATION) + chunk_start
                            if lower is not None and g_start < lower:
                                continue  # appartient à la tranche précédente
                            if upper is not None and g_start >= upper:
                                continue  # appartient à la tranche suivante
                            words.append(Word(g_start, g_end, w.word, w.probability))
                    else:
                        g_start = seg.start + chunk_start
                        g_end = min(seg.end, seg.start + WORD_MAX_DURATION) + chunk_start
                        in_overlap = (lower is not None and g_start < lower) or (
                            upper is not None and g_start >= upper
                        )
                        if not in_overlap:
                            # Pas de mots individuels (segment sans
                            # word_timestamps) : repli sur la confiance
                            # moyenne du segment entier (avg_logprob,
                            # log-probabilité -> probabilité via exp()).
                            words.append(Word(g_start, g_end, seg.text, math.exp(seg.avg_logprob)))

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

        return words, detected_language, total_duration, resolved_prompt

    def transcribe(
        self,
        source: str,
        on_phase: Callable[[str], None] | None = None,
        on_progress: Callable[[float], None] | None = None,
        cancel_event: threading.Event | None = None,
        subtitle_preset: str = DEFAULT_SUBTITLE_PRESET,
        speech_type: str | None = None,
        context_options: list[str] | None = None,
        register: str | None = None,
        language: str | None = None,
        proper_nouns: list[str] | None = None,
    ) -> tuple[list[Segment], str, float, str, list[Segment], float]:
        """Renvoie (sous-titres courts découpés à partir des mots, langue
        détectée, durée réelle de l'audio en secondes, initial_prompt résolu,
        cues supprimés comme échos de prompt — pour traçabilité QC —, cadence
        image réellement utilisée pour ce job — pour traçabilité QC).

        on_phase("converting" | "transcribing") signale le changement d'étape ;
        on_progress(0.0..1.0) la progression au sein de l'étape courante ;
        cancel_event, si activé, interrompt le traitement (lève Cancelled) ;
        subtitle_preset sélectionne le jeu de règles de lisibilité/QC (cf.
        SUBTITLE_PRESETS) — remplace l'ancien char_limit/output_profile
        (migration du 05/10/2026 vers les 9 presets de la fiche Master Hub).

        Le initial_prompt n'est plus reçu tout composé : il est construit ICI,
        après détection de la langue (cf. _collect_words), et DANS cette
        langue — un prompt anglais par défaut biaisait Whisper vers une
        sortie en anglais sur du contenu non anglophone (bug constaté en
        prod le 30/09/2026), y compris avec `language` forcé dans l'autre
        sens.
        """
        preset = SUBTITLE_PRESETS.get(subtitle_preset, SUBTITLE_PRESETS[DEFAULT_SUBTITLE_PRESET])
        char_limit = preset["max_chars"]
        max_lines = preset["max_lines"]
        # Cadence réelle du média (05/10/2026) plutôt qu'une hypothèse fixe à
        # 25fps — cf. _detect_frame_rate pour le détail et les replis.
        frame_rate = self._detect_frame_rate(source)
        min_duration = preset["min_duration_frames"] / frame_rate
        max_duration = preset["max_duration_sec"]
        min_gap = preset["min_gap_frames"] / frame_rate
        cps_target_max = preset["cps_max"]
        cps_warning = preset["cps_warning"]
        first_subtitle_min_sec = preset["first_subtitle_min_sec"]
        # Symétrique au délai avant le 1er sous-titre (décision client du
        # 05/10/2026, "les deux contrôles combinés" — pas de champ dédié dans
        # la fiche pour la fin de programme).
        last_subtitle_min_sec = preset["first_subtitle_min_sec"]

        if on_phase:
            on_phase("converting")
        audio = self._extract_audio(source, on_progress, cancel_event)
        if on_phase:
            on_phase("transcribing")
        words, detected_language, duration, resolved_prompt = self._collect_words(
            audio, on_progress, cancel_event,
            speech_type, context_options, register, proper_nouns, language,
        )
        words = _remove_disfluencies(words)
        cues = self._build_cues(words, char_limit, max_lines, min_duration, max_duration)
        cues, removed_echoes = _remove_hallucinated_cues(
            cues, resolved_prompt, char_limit, max_lines,
        )
        cues = _dedup_internal(cues, char_limit, max_lines)
        cues = _dedup_cross_segment(cues, char_limit, max_lines)
        cues = _merge_stray_connectors(cues, char_limit, max_lines, max_duration)
        cues = _merge_split_names(cues, char_limit, max_lines, max_duration)
        cues = _merge_split_contractions(cues, char_limit, max_lines)
        cues = _merge_short_cues(
            cues, char_limit,
            min_duration=min_duration, max_duration=max_duration, min_gap=min_gap,
            max_lines=max_lines,
        )
        cues = _enforce_min_gap(cues, min_gap)
        cues = _normalize_proper_nouns(cues, proper_nouns, char_limit, max_lines)
        _compute_confidence(cues, words)
        _compute_warnings(
            cues, char_limit, max_lines, min_duration, max_duration, min_gap,
            cps_target_max, cps_warning, first_subtitle_min_sec, last_subtitle_min_sec,
            duration,
        )
        return cues, detected_language, duration, resolved_prompt, removed_echoes, frame_rate

    @staticmethod
    def _build_cues(
        words: list[Word],
        char_limit: int = WRAP_WIDTH,
        max_lines: int = 2,
        min_duration: float = MIN_CUE_DURATION,
        max_duration: float = MAX_CUE_DURATION,
    ) -> list[Segment]:
        max_cue_chars = max_lines * char_limit
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
                if idx is not None and (buf[idx - 1].end - buf[0].start) >= min_duration:
                    held = buf[idx:]
                    del buf[idx:]
            # NB : la protection des connecteurs isolés et des noms propres
            # coupés se fait en post-traitement (_merge_stray_connectors,
            # _merge_split_names) plutôt qu'ici — un mot problématique se
            # retrouve trop souvent seul en tête de tampon après un flush
            # précédent, où "len(buf) > 1" ne peut pas s'appliquer.
            text = "".join(w.text for w in buf).strip()
            cues.append(Segment(buf[0].start, buf[-1].end, _wrap(text, char_limit, max_lines)))
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
                    prospective_dur > max_duration
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
                        prospective_dur <= max_duration + PROPER_NOUN_SLACK_SEC
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


def _best_two_line_split(words: list[str]) -> tuple[int, str, str]:
    """(longueur de la ligne la plus longue, ligne1, ligne2) — algorithme
    historique de _wrap, inchangé depuis sa version 1-2 lignes."""
    best = None
    for i in range(1, len(words)):
        l1, l2 = " ".join(words[:i]), " ".join(words[i:])
        longest = max(len(l1), len(l2))
        if best is None or longest < best[0]:
            best = (longest, l1, l2)
    return best


def _best_three_line_split(words: list[str]) -> tuple[int, str, str, str] | None:
    """Même principe que _best_two_line_split pour 3 lignes : minimise la
    longueur de la ligne la plus longue parmi tous les découpages en 3
    segments contigus. None si moins de 3 mots (3 lignes non pertinentes)."""
    if len(words) < 3:
        return None
    best = None
    for i in range(1, len(words) - 1):
        for j in range(i + 1, len(words)):
            l1 = " ".join(words[:i])
            l2 = " ".join(words[i:j])
            l3 = " ".join(words[j:])
            longest = max(len(l1), len(l2), len(l3))
            if best is None or longest < best[0]:
                best = (longest, l1, l2, l3)
    return best


def _wrap(text: str, width: int = WRAP_WIDTH, max_lines: int = 2) -> str:
    """Répartit le texte sur 1 à max_lines lignes équilibrées pour la
    lisibilité. Comportement à 2 lignes strictement inchangé (algorithme
    historique, cf. _best_two_line_split) ; extension à 3 lignes (05/10/2026,
    presets fiche Master Hub : dcp_smpte_bv21_technical, dcp_sme_fr_ktt,
    transcription_tolerance_max_ktt) seulement si max_lines >= 3 ET qu'aucun
    découpage à 2 lignes ne tient dans `width` — toujours préférer le moins
    de lignes possible, conforme aux conventions de sous-titrage."""
    if len(text) <= width:
        return text
    words = text.split()
    if len(words) <= 1:
        return text
    two = _best_two_line_split(words)
    if max_lines < 3 or two[0] <= width:
        return f"{two[1]}\n{two[2]}"
    three = _best_three_line_split(words)
    if three is not None:
        return f"{three[1]}\n{three[2]}\n{three[3]}"
    return f"{two[1]}\n{two[2]}"


PROMPT_ECHO_OVERLAP_MIN = 0.6       # part minimale des mots du cue retrouvée dans le prompt
PROMPT_ECHO_ISOLATION_SEC = 20.0    # silence minimal d'un côté pour suspecter une hallucination
# Fragment orphelin d'une hallucination coupée en plusieurs cues par
# _build_cues (ex. "501" seul, ou "-Canada" commençant par un tiret orphelin
# — jamais un vrai mot de dialogue ne commence ainsi). Combiné à l'isolement
# (silence des deux côtés), signal étroit et sûr.
_SUSPICIOUS_FRAGMENT_RE = re.compile(r"^-?\s*\d+[.,]?$|^-[a-zà-ÿ]", re.IGNORECASE)
_WORD_ONLY_RE = re.compile(r"[a-zà-öø-ÿ']+")

# Hallucinations génériques connues de Whisper sur du silence — INDÉPENDANTES
# de l'initial_prompt, un phénomène documenté du modèle lui-même (entraîné
# sur beaucoup de contenu YouTube sous-titré) : crédits de sous-titrage
# fictifs, remerciements de fin de vidéo. Observées en prod le 30/09 puis le
# 01/10/2026 ("Sous-titrage ST' 501", "Gracias por ver el video.",
# "Transcription by CastingWords") — un premier correctif ne couvrant que
# l'écho du prompt les avait laissées passer. Liste non exhaustive,
# multilingue (EN/FR/ES/DE/IT/PT), étendue au fil des cas rencontrés.
_KNOWN_HALLUCINATION_RE = re.compile(
    "subscribe|thanks? for watching|like and subscribe|see you (?:in the )?next (?:video|time)|transcri(?:bed|ption) by|subtitles? by|captions? by|amara\\.org|abonnez[ -]vous|merci d'avoir regard|sous-?titr(?:age|é|es?)(?:\\s+(?:fr|en|vf|vo|vostfr|es|de|it|pt))?\\s*[?!]?|-titr(?:age|é|es?)|\\bst'? ?\\d+\\b|société radio[ -]canada|traduit par|suscr[ií]bete|gracias por ver|subt[ií]tulos? (?:por|de)|abonnier|vielen dank f[uü]rs zuschauen|untertitel(?:t)? (?:von|durch)|iscriviti|grazie per aver guardato|sottotitol|inscreva-se|obrigado por assistir|legendado",
    re.IGNORECASE,
)


def _remove_hallucinated_cues(
    cues: list[Segment], initial_prompt: str | None, char_limit: int = WRAP_WIDTH,
    max_lines: int = 2,
) -> tuple[list[Segment], list[Segment]]:
    """Supprime un cue qui n'est manifestement pas une vraie transcription
    mais une hallucination de Whisper sur du silence/non-parole — observée
    en prod sous trois formes (30/09 puis 01/10/2026) : (1) écho du
    initial_prompt fourni ; (2) formule générique connue du modèle,
    INDÉPENDANTE du prompt (cf. _KNOWN_HALLUCINATION_RE) ; (3) fragment
    orphelin d'une hallucination coupée par _build_cues (ex. un cue réduit à
    un simple nombre isolé, ou commençant par un tiret orphelin).

    Vérifie CHAQUE CUE INDIVIDUELLEMENT (jamais un texte concaténé sur
    plusieurs cues) : une première version groupait les cues isolés entre
    eux en "îlots" pour recoller les fragments d'une même hallucination,
    mais une vraie scène de dialogue a elle aussi des pauses naturelles
    < 20s entre répliques — un "îlot" pouvait donc englober toute une scène
    réelle, supprimée en bloc dès qu'une hallucination isolée s'y trouvait
    quelque part (régression constatée le 01/10/2026 : 146 cues supprimés,
    dont une scène entière de dialogue et des paroles de chanson, à cause
    d'un "Gracias por ver el video." perdu au milieu). Une hallucination
    fragmentée peut donc laisser un résidu partiel (ex. "501" orphelin non
    reconnu) — préférable à supprimer du vrai contenu.

    Renvoie (cues conservés, cues supprimés) — le second élément alimente le
    log QC : une suppression de contenu doit rester traçable côté client,
    jamais silencieuse (cf. module KTT §6)."""
    prompt_words = set(_WORD_ONLY_RE.findall(initial_prompt.lower())) if initial_prompt else set()

    kept: list[Segment] = []
    removed: list[Segment] = []
    for i, seg in enumerate(cues):
        text = seg.text.replace("\n", " ").strip()
        text_words = _WORD_ONLY_RE.findall(text.lower())
        prompt_overlap = (
            sum(1 for w in text_words if w in prompt_words) / len(text_words)
            if text_words and prompt_words else 0.0
        )
        is_prompt_echo = prompt_overlap >= PROMPT_ECHO_OVERLAP_MIN
        is_known_phrase = bool(_KNOWN_HALLUCINATION_RE.search(text))
        is_bare_fragment = bool(_SUSPICIOUS_FRAGMENT_RE.match(text))
        if not (is_prompt_echo or is_known_phrase or is_bare_fragment):
            kept.append(seg)
            continue

        if is_known_phrase:
            # Observé le 01/10/2026 (test VAD désactivé) : une formule connue
            # peut se trouver ACCOLÉE à du vrai dialogue dans le MÊME cue
            # (ex. "Sous-titrage ST' Qu'est-ce que tu vas" — la suite "faire ?"
            # est dans le cue suivant, orpheline si on supprime tout). On
            # retire donc d'abord UNIQUEMENT la portion correspondant à la
            # formule connue ; s'il reste du texte avec au moins un vrai mot,
            # on garde le cue modifié (avec trace dans ses warnings) plutôt
            # que de perdre du vrai contenu. Un résidu court type "ST'" peut
            # subsister accolé (même compromis assumé que pour les fragments
            # orphelins ailleurs dans cette fonction).
            remainder = _KNOWN_HALLUCINATION_RE.sub(" ", text)
            remainder = re.sub(r"\s+", " ", remainder).strip()
            remainder_words = _WORD_ONLY_RE.findall(remainder.lower())
            if remainder and len(remainder_words) >= 1 and not _SUSPICIOUS_FRAGMENT_RE.match(remainder):
                seg.text = _wrap(remainder, char_limit, max_lines)
                seg.warnings.append("HALLUCINATION_PARTIAL_STRIPPED")
                kept.append(seg)
                continue
            # Signal lexical fort à lui seul (ces formules n'apparaissent
            # jamais dans un vrai dialogue diégétique) : pas besoin d'un
            # silence bilatéral. Utile en particulier pour une hallucination
            # fragmentée en plusieurs cues consécutifs (petits écarts entre
            # fragments, ex. observé 12,9s entre deux morceaux de "Sous-
            # titrage ST' 501") — exiger l'isolement des deux côtés aurait
            # laissé passer chaque fragment individuellement.
            isolated = True
        else:
            # Écho de prompt (signal plus faible) ou fragment nu ("501",
            # "-Canada" seuls — signal ambigu, pourrait être un vrai mot
            # isolé) : on garde l'exigence d'isolement bilatéral par
            # prudence, quitte à laisser un résidu partiel non supprimé
            # plutôt que de risquer de couper du vrai contenu.
            prev_gap = seg.start - cues[i - 1].end if i > 0 else float("inf")
            next_gap = cues[i + 1].start - seg.end if i + 1 < len(cues) else float("inf")
            isolated = min(prev_gap, next_gap) > PROMPT_ECHO_ISOLATION_SEC
        if not isolated:
            kept.append(seg)
            continue

        if is_prompt_echo:
            reason = "PROMPT_ECHO_REMOVED"
        elif is_known_phrase:
            reason = "KNOWN_HALLUCINATION_REMOVED"
        else:
            reason = "HALLUCINATION_FRAGMENT_REMOVED"
        seg.warnings.append(reason)
        removed.append(seg)
    return kept, removed


def _dedup_internal(cues: list[Segment], char_limit: int = WRAP_WIDTH, max_lines: int = 2) -> list[Segment]:
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
            new_seg = Segment(seg.start, seg.end, _wrap(" ".join(tokens), char_limit, max_lines))
            new_seg.warnings = list(seg.warnings) + ["INTERNAL_DUPLICATE_REMOVED"]
            result.append(new_seg)
        else:
            result.append(seg)
    return result


def _dedup_cross_segment(cues: list[Segment], char_limit: int = WRAP_WIDTH, max_lines: int = 2) -> list[Segment]:
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
                    new_seg = Segment(nxt.start, nxt.end, _wrap(" ".join(remaining), char_limit, max_lines))
                    new_seg.warnings = list(nxt.warnings) + ["CROSS_SEGMENT_OVERLAP_REMOVED"]
                    result[i + 1] = new_seg
                else:
                    del result[i + 1]
                    continue
        i += 1
    return result


def _compute_confidence(cues: list[Segment], words: list[Word]) -> None:
    """Score de confiance moyen par cue (0-1), calculé APRÈS tout le pipeline
    de fusion/découpage en comparant les bornes temporelles finales de
    chaque cue aux mots d'origine (Word.probability, exposée par
    faster-whisper par mot décodé) plutôt qu'en essayant de faire porter ce
    score à travers chaque passe de fusion individuellement — plus simple,
    plus sûr (aucune des ~8 fonctions de fusion/découpage n'a besoin d'être
    modifiée), et sémantiquement correct : la confiance d'un cue reflète la
    confiance de Whisper sur la parole couvrant cette plage de temps,
    indépendamment de la façon dont le texte a ensuite été redécoupé.

    Mute les cues en place (ajoute Segment.confidence). Un cue sans mot
    correspondant (ex. un fragment résiduel après _enforce_min_gap) garde
    confidence=None plutôt qu'une valeur inventée."""
    for seg in cues:
        matching = [w for w in words if w.start >= seg.start - 1e-3 and w.end <= seg.end + 1e-3]
        if not matching:
            # Repli : mots chevauchant partiellement la plage (bords d'un
            # cue retouché par une fusion ultérieure).
            matching = [w for w in words if w.end > seg.start and w.start < seg.end]
        if matching:
            seg.confidence = round(sum(w.probability for w in matching) / len(matching), 3)


def _compute_warnings(
    cues: list[Segment],
    char_limit: int = WRAP_WIDTH,
    max_lines: int = 2,
    min_duration: float = MIN_CUE_DURATION,
    max_duration: float = MAX_CUE_DURATION,
    min_gap: float = MIN_GAP,
    cps_target_max: float = 15.0,
    cps_warning: float = MAX_CPS,
    first_subtitle_min_sec: float = 0.0,
    last_subtitle_min_sec: float = 0.0,
    total_duration: float | None = None,
) -> None:
    """Calcule le log QC (module KTT) sur les cues finalisés — dernière étape
    du pipeline, après toutes les fusions/découpages, pour refléter ce qui
    est réellement livré plutôt qu'un état intermédiaire. Mute les cues en
    place (ajoute à Segment.warnings), ne modifie ni le texte ni le timing.

    Taxonomie adoptée le 05/10/2026 (décision client) depuis la fiche Master
    Hub §7 "Règles de QC minimales" : CPS_ABOVE_TARGET/CPS_ABOVE_WARNING
    remplacent CPS_HIGH ; DURATION_BELOW_MINIMUM remplace SHORT_CUE_KEPT.
    Nouveaux : DURATION_ABOVE_MAXIMUM, MAX_LINES_EXCEEDED, MAX_CPL_EXCEEDED,
    OVERLAP_DETECTED (garde-fou défensif — _enforce_min_gap l'empêche déjà
    structurellement, ne devrait donc jamais apparaître en pratique),
    FIRST_SUBTITLE_TOO_EARLY. GAP_BELOW_MINIMUM reste la responsabilité
    exclusive de _enforce_min_gap (pas de doublon ici). Deux codes propres à
    whisper-ktt sans équivalent dans la fiche (décision client du 05/10/2026,
    même traitement que le délai du 1er sous-titre pour la fin de
    programme) : LAST_SUBTITLE_TOO_LATE / LAST_SUBTITLE_EXCEEDS_MEDIA.

    SHOT_CHANGE_TOO_CLOSE n'est jamais émis (timestamps de plan indisponibles,
    pipeline audio seul — mis de côté par le client) ; INVALID_TEXT_ENCODING
    non plus (UTF-8 partout, Python 3 — ne peut structurellement pas
    survenir). Cf. analyse du 29/09/2026."""
    for seg in cues:
        dur = seg.end - seg.start
        text = seg.text.replace("\n", " ").strip()
        lines = seg.text.split("\n")

        if dur > 1e-6:
            cps = len(text) / dur
            if cps > cps_warning:
                seg.warnings.append("CPS_ABOVE_WARNING")
            elif cps > cps_target_max:
                seg.warnings.append("CPS_ABOVE_TARGET")

        if dur < min_duration - 1e-6:
            seg.warnings.append("DURATION_BELOW_MINIMUM")
        if dur > max_duration + 1e-6:
            seg.warnings.append("DURATION_ABOVE_MAXIMUM")

        if len(lines) > max_lines:
            seg.warnings.append("MAX_LINES_EXCEEDED")
        if any(len(line) > char_limit for line in lines):
            seg.warnings.append("MAX_CPL_EXCEEDED")

        words_in_cue = text.split()
        if len(words_in_cue) == 1 and not text.rstrip().endswith((".", "!", "?")):
            seg.warnings.append("ISOLATED_WORD_KEPT")

    for i in range(len(cues) - 1):
        if cues[i + 1].start < cues[i].end - 1e-6:
            cues[i].warnings.append("OVERLAP_DETECTED")

    if cues:
        if cues[0].start < first_subtitle_min_sec - 1e-6:
            cues[0].warnings.append("FIRST_SUBTITLE_TOO_EARLY")
        if total_duration is not None:
            if (total_duration - cues[-1].end) < last_subtitle_min_sec - 1e-6:
                cues[-1].warnings.append("LAST_SUBTITLE_TOO_LATE")
            if cues[-1].end > total_duration + 1e-6:
                cues[-1].warnings.append("LAST_SUBTITLE_EXCEEDS_MEDIA")


def _merge_stray_connectors(
    cues: list[Segment], char_limit: int = WRAP_WIDTH, max_lines: int = 2,
    max_duration: float = MAX_CUE_DURATION,
) -> list[Segment]:
    """Fusionne un sous-titre réduit à un seul mot SANS ponctuation de fin de
    phrase (jamais un énoncé complet — contrairement à "Oui."/"Non.") avec le
    voisin le plus proche dans le temps, texte inclus. Couvre aussi bien un
    connecteur/interjection isolé (ex. "Et", "Que") qu'un mot porteur de sens
    coupé de la suite d'une phrase par une pause détectée par Whisper (ex.
    "Hier" séparé de "soir, ..."). Opère sur des cues déjà finalisés : le
    calcul de proximité (avant/après) n'est donc pas faussé par un grand
    silence traversé pendant l'accumulation des mots."""
    max_chars = max_lines * char_limit
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
                if len(combined_text) <= max_chars and combined_dur <= max_duration:
                    result[i - 1] = Segment(prev_seg.start, seg.end, _wrap(combined_text, char_limit, max_lines))
                    del result[i]
                    merged = True
            elif next_seg is not None:
                combined_text = (seg.text.replace("\n", " ") + " " + next_seg.text.replace("\n", " ")).strip()
                combined_dur = next_seg.end - seg.start
                if len(combined_text) <= max_chars and combined_dur <= max_duration:
                    result[i] = Segment(seg.start, next_seg.end, _wrap(combined_text, char_limit, max_lines))
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


# Pas de plafond strict à MAX_CUE_DURATION pour une fusion de contraction
# coupée (cf. _merge_split_contractions) : un fragment du type "C" seul est
# toujours incorrect, un cue un peu long reste lisible. Garde-fou large
# contre un faux positif pathologique plutôt qu'une vraie limite de confort.
CONTRACTION_MERGE_MAX_DUR = 15.0
# Marge de caractères, même logique que PROPER_NOUN_SLACK_CHARS : une
# contraction coupée entre deux cues déjà proches du plafond de 2 lignes
# (ex. observé : 95 caractères pour un plafond à 84) doit quand même se
# recoller — un cue un peu dense reste plus lisible qu'une apostrophe
# orpheline en début de ligne.
CONTRACTION_MERGE_CHARS_SLACK = 30
_CONTRACTION_END_RE = re.compile(r"[a-zà-ÿ]+['\u2019]$", re.IGNORECASE)
_CONTRACTION_START_RE = re.compile(r"^['\u2019][a-zà-ÿ]", re.IGNORECASE)


def _merge_split_contractions(cues: list[Segment], char_limit: int = WRAP_WIDTH, max_lines: int = 2) -> list[Segment]:
    """Recolle deux cues quand une contraction (française notamment : d',
    l', qu', j', n', s', c', m', t', jusqu', lorsqu', puisqu', quoiqu'...) a
    été coupée entre les deux — observé en prod le 01/10/2026 : "...indienne
    d" / "'une tribu inconnue." (devrait être "indienne d'une tribu
    inconnue."), un écart de timing Whisper entre les deux moitiés d'une
    contraction ayant déclenché une coupure de cue en plein milieu. Une
    apostrophe en fin de mot signale une incomplétude grammaticale : ne
    devrait jamais survivre comme fin de cue. Jointure SANS espace entre les
    deux moitiés ("d" + "'une" -> "d'une", pas "d 'une")."""
    max_chars = max_lines * char_limit
    result = list(cues)
    i = 0
    while i < len(result) - 1:
        cur, nxt = result[i], result[i + 1]
        cur_text = cur.text.replace("\n", " ").rstrip()
        nxt_text = nxt.text.replace("\n", " ").lstrip()
        cur_words = cur_text.split()
        if not cur_words or not nxt_text:
            i += 1
            continue
        # Le mot coupé par Whisper peut garder son apostrophe côté gauche
        # ("d'" / "une") ou la perdre entièrement au profit du côté droit
        # ("d" / "'une", cas observé en prod) : les deux déclenchent la
        # fusion, le signal fort étant surtout le mot suivant qui commence
        # par une apostrophe (jamais un vrai début de mot autrement).
        if _CONTRACTION_END_RE.search(cur_words[-1]) or _CONTRACTION_START_RE.match(nxt_text):
            combined_text = f"{cur_text}{nxt_text}".strip()
            combined_dur = nxt.end - cur.start
            # Pas de plafond strict à MAX_CUE_DURATION ici (contrairement aux
            # autres fusions) : un cue réduit à "C" ou débutant par une
            # apostrophe orpheline est TOUJOURS incorrect, alors qu'un cue
            # un peu long reste lisible — observé en prod : fusions
            # nécessaires à 5,01s et 7,28s, rejetées à tort par la limite de
            # 5,0s avant ce correctif. CONTRACTION_MERGE_MAX_DUR reste un
            # garde-fou contre un cas pathologique (deux cues très éloignés
            # ressemblant par coïncidence à une contraction coupée).
            if len(combined_text) <= max_chars + CONTRACTION_MERGE_CHARS_SLACK and combined_dur <= CONTRACTION_MERGE_MAX_DUR:
                result[i] = Segment(cur.start, nxt.end, _wrap(combined_text, char_limit, max_lines))
                del result[i + 1]
                continue
        i += 1
    return result


def _merge_split_names(
    cues: list[Segment], char_limit: int = WRAP_WIDTH, max_lines: int = 2,
    max_duration: float = MAX_CUE_DURATION,
) -> list[Segment]:
    """Recolle deux cues consécutifs quand tout indique qu'un nom propre ou
    composé (ex. "Papi" / "Chouette") a été coupé entre les deux : le premier
    ne se termine pas par une ponctuation de fin de phrase et se termine par
    un mot capitalisé, le second commence par un mot capitalisé. En français,
    une majuscule en milieu de phrase signale quasi toujours un nom propre —
    ce test a donc un faible taux de faux positifs sur ce type de contenu."""
    max_chars = max_lines * char_limit
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
            if len(combined_text) <= max_chars and combined_dur <= max_duration:
                result[i] = Segment(cur.start, nxt.end, _wrap(combined_text, char_limit, max_lines))
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
    max_lines: int = 2,
) -> list[Segment]:
    """Applique le "Cas 2" du module KTT : un sous-titre trop court (< durée
    min) est d'abord étiré (sans empiéter sur le suivant, en respectant le
    gap minimal), sinon fusionné avec le suivant, sinon avec le précédent —
    dans la limite des caractères/ligne et de la durée max."""
    if min_gap is None:
        min_gap = MIN_GAP
    if not cues:
        return cues

    max_chars = max_lines * char_limit
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
                result.append(Segment(seg.start, next_seg.end, _wrap(combined_text, char_limit, max_lines)))
                i += 2
                continue

        if result:
            prev = result[-1]
            combined_text = (prev.text.replace("\n", " ") + " " + seg.text.replace("\n", " ")).strip()
            combined_dur = seg.end - prev.start
            if len(combined_text) <= max_chars and combined_dur <= max_duration:
                result[-1] = Segment(prev.start, seg.end, _wrap(combined_text, char_limit, max_lines))
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
    max_lines: int = 2,
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
            new_seg = Segment(seg.start, seg.end, _wrap(new_text, char_limit, max_lines))
            new_seg.warnings = list(seg.warnings) + ["PROPER_NOUN_NORMALIZED"]
            result.append(new_seg)
        else:
            result.append(seg)
    return result


def _enforce_min_gap(cues: list[Segment], min_gap: float = MIN_GAP) -> list[Segment]:
    """Recule la fin de chaque cue si besoin pour garantir un gap minimal (norme ATAA)
    avec le sous-titre suivant. Ne raccourcit jamais un cue sous sa durée nulle.

    Garde-fou absolu (30/09/2026) : quelle que soit la cause en amont (ex. bug
    de déduplication aux frontières de tranches, signalé en prod — "bloc 446 :
    ce créneau chevauche un autre sous-titre"), un cue ne doit JAMAIS ressortir
    de cette passe en empiétant sur le suivant. Quand reculer la fin du cue tout
    en respectant min_gap l'invaliderait (durée nulle/négative), on renonce au
    gap minimal mais on garantit au moins l'absence de chevauchement."""
    for i in range(len(cues) - 1):
        gap = cues[i + 1].start - cues[i].end
        if gap < min_gap:
            new_end = cues[i + 1].start - min_gap
            if new_end > cues[i].start:
                cues[i].end = new_end
            else:
                cues[i].end = min(cues[i].end, cues[i + 1].start)
                cues[i].warnings.append("GAP_BELOW_MINIMUM")  # gap minimal non atteint (cas limite)
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

