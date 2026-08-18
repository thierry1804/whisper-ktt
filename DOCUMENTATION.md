# Documentation — whisper-service

Service de sous-titrage automatique : on soumet une URL (mp4 ou wav) ou on envoie directement un fichier (mp4 ou wav), le service extrait l'audio (ffmpeg) et le transcrit (faster-whisper), puis fournit des sous-titres `.srt` / `.vtt`.

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Authentification / clé API](#authentification--clé-api)
- [Endpoints](#endpoints)
- [Cycle de vie d'un job](#cycle-de-vie-dun-job)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Configuration (transcriber.py)](#configuration-transcriberpy)
- [Démarrage local](#démarrage-local)
- [Démarrage avec Docker](#démarrage-avec-docker)
- [Déploiement en production](#déploiement-en-production)
- [Sécurité et bonnes pratiques](#sécurité-et-bonnes-pratiques)

---

## Vue d'ensemble

- **Stack** : FastAPI + [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (modèle `small`, CPU, quantifié `int8`), ffmpeg pour l'extraction audio.
- **Traitement asynchrone** : une file d'attente en mémoire, un seul job traité à la fois (`app.py`, worker unique).
- **Sortie** : fichiers `.srt` et `.vtt`, avec découpage des sous-titres optimisé pour la lisibilité (≤ 6s, ≤ 84 caractères, retour à la ligne équilibré).
- **Fichiers de sortie éphémères** : chaque fichier est supprimé du serveur juste après son premier téléchargement.
- **Déploiement** : conteneurs Docker (`whisper-service` + reverse proxy `Caddy` pour le TLS automatique).

Fichiers clés du dépôt :

| Fichier | Rôle |
|---|---|
| `app.py` | API FastAPI (routes, file d'attente, auth) |
| `transcriber.py` | Cœur métier : extraction audio, transcription, formatage SRT/VTT |
| `test_cli.py` | Test en ligne de commande sans passer par l'API |
| `run_local.sh` | Lance l'API en local (charge `.env`, `uvicorn --reload` sur le port 8000) |
| `Dockerfile` | Image du service (port interne 8140) |
| `docker-compose.yml` | Orchestration `whisper-service` + `caddy` |
| `Caddyfile` | Reverse proxy / TLS automatique |
| `DEPLOY.md` | Procédure de déploiement détaillée (serveur macOS) |

---

## Authentification / clé API

Toutes les routes `/jobs*` exigent un header **`X-API-Key`**. `GET /health` est public (pas d'auth).

La clé est lue depuis la variable d'environnement `API_KEY` au démarrage du service (`app.py:21`) — le service refuse de démarrer si elle est absente.

**Clé API actuellement configurée dans `.env` (environnement local de ce dépôt) :**

```
API_KEY=cle-ktt-20260810
```

> ⚠️ Ce fichier `.env` est exclu de git (`.gitignore`) et ne doit jamais être commité ni partagé en clair hors d'un canal sécurisé. Cette valeur semble être une clé de test/développement (motif `cle-ktt-<date>`) — pour la production, générer une clé forte aléatoire (voir plus bas) plutôt que de réutiliser celle-ci.

### Générer une nouvelle clé (production)

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Coller la valeur dans `.env` :

```
API_KEY=<valeur générée>
```

Cette clé doit être transmise à la plateforme cliente qui l'enverra dans le header `X-API-Key` sur chaque appel à `/jobs*`.

### Rotation de la clé

```bash
# éditer API_KEY dans .env, puis :
docker compose up -d   # ne recrée que whisper-service
```
Prévenir la plateforme cliente du changement.

---

## Endpoints

| Méthode | Route | Auth | Description |
|---|---|---|---|
| `POST` | `/jobs` | ✅ | Crée un job de transcription à partir d'une URL ou d'un fichier uploadé (mp4/wav) |
| `GET` | `/jobs/{id}` | ✅ | Statut et progression du job |
| `POST` | `/jobs/{id}/cancel` | ✅ | Annule un job en attente ou en cours |
| `GET` | `/jobs/{id}/srt` | ✅ | Télécharge le `.srt` (supprimé du serveur après téléchargement) |
| `GET` | `/jobs/{id}/vtt` | ✅ | Télécharge le `.vtt` (supprimé du serveur après téléchargement) |
| `GET` | `/health` | ❌ | Ping santé, renvoie aussi la taille de la file d'attente |

### `POST /jobs`

Corps en `multipart/form-data`, avec **exactement un** des deux champs :

| Champ | Type | Description |
|---|---|---|
| `url` | texte | URL d'un fichier mp4 ou wav |
| `file` | fichier | Upload direct d'un fichier mp4 ou wav |

Fournir les deux champs, ou aucun des deux, renvoie `400`. Une extension autre que `.mp4`/`.wav` sur `file` renvoie aussi `400`.

Exemple (URL) :
```bash
curl -X POST "$BASE_URL/jobs" -H "X-API-Key: $API_KEY" -F "url=https://exemple.com/video.mp4"
```

Exemple (fichier direct) :
```bash
curl -X POST "$BASE_URL/jobs" -H "X-API-Key: $API_KEY" -F "file=@/chemin/vers/audio.wav"
```

Réponse `200` :
```json
{
  "id": "a1b2c3d4e5f6",
  "source": "https://exemple.com/video.mp4",
  "status": "queued",
  "progress": 0.0,
  "srt_path": null,
  "vtt_path": null,
  "error": null
}
```
`source` reflète soit l'URL soumise, soit le chemin local (temporaire) du fichier uploadé — dans les deux cas c'est cette valeur qui est passée telle quelle à ffmpeg (`-i <source>`), qui détecte lui-même le format (mp4/wav/autres conteneurs audio/vidéo qu'il sait lire). Un fichier uploadé est stocké dans `UPLOAD_DIR` le temps du traitement, puis supprimé dès que le job atteint un état terminal (`done`/`error`/`cancelled`) — comme pour les sorties `.srt`/`.vtt`, rien n'est conservé au-delà du nécessaire.

### `GET /jobs/{id}`

Statuts possibles : `queued`, `converting`, `transcribing`, `cancelling`, `cancelled`, `done`, `error`.
`progress` va de `0.0` à `1.0` et est remis à zéro à chaque changement de statut (`converting` = extraction ffmpeg, `transcribing` = passage Whisper).

Si la transcription se déroule sans erreur technique mais qu'aucune parole n'est détectée dans le média (silence, musique seule, piste audio vide...), le job passe en `error` avec `error: "Aucune parole détectée dans le média"` plutôt qu'en `done` avec un fichier de sortie vide.

### `POST /jobs/{id}/cancel`

- Job encore en file (`queued`) → passe directement à `cancelled`.
- Job en cours (`converting`/`transcribing`) → passe à `cancelling`, puis `cancelled` une fois le process effectivement arrêté (jusqu'à ~0.5s) — il faut repoller `GET /jobs/{id}`.
- Renvoie `409` si le job est déjà dans un état terminal (`done`/`error`/`cancelled`).

### `GET /jobs/{id}/srt` et `/vtt`

- `409` si le job n'est pas encore `done`.
- `404` si le fichier a déjà été récupéré (téléchargement unique) ou est introuvable.
- Le fichier est supprimé du disque juste après l'envoi de la réponse.

### `GET /health`

```json
{ "status": "ok", "en_file": 0 }
```
`en_file` = nombre de jobs en attente dans la file (`QUEUE.qsize()`).

---

## Cycle de vie d'un job

```
queued → converting → transcribing → done
                                   ↘ error
   ↘ cancelling → cancelled  (annulation en cours de traitement)
   ↘ cancelled                (annulation alors qu'il était encore en file)
```

Traitement séquentiel : un seul job à la fois (pas de parallélisme), les suivants patientent en file (`QUEUE`).

---

## Exemples d'utilisation

Variables utilisées ci-dessous : `BASE_URL` (ex. `http://localhost:8000` en local, ou le domaine HTTPS en production) et `API_KEY`.

```bash
BASE_URL=http://localhost:8000
API_KEY=cle-ktt-20260810

# 1a. Créer un job à partir d'une URL (mp4 ou wav)
curl -X POST "$BASE_URL/jobs" \
  -H "X-API-Key: $API_KEY" \
  -F "url=https://exemple.com/video.mp4"
# → {"id": "a1b2c3d4e5f6", "status": "queued", ...}

# 1b. ... ou en envoyant directement un fichier (mp4 ou wav)
curl -X POST "$BASE_URL/jobs" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/chemin/vers/video.mp4"

# 2. Suivre la progression (polling)
curl "$BASE_URL/jobs/a1b2c3d4e5f6" -H "X-API-Key: $API_KEY"

# 3. Télécharger le sous-titre une fois "done"
curl "$BASE_URL/jobs/a1b2c3d4e5f6/srt" -H "X-API-Key: $API_KEY" -o sous-titres.srt
curl "$BASE_URL/jobs/a1b2c3d4e5f6/vtt" -H "X-API-Key: $API_KEY" -o sous-titres.vtt

# 4. Annuler un job en cours
curl -X POST "$BASE_URL/jobs/a1b2c3d4e5f6/cancel" -H "X-API-Key: $API_KEY"

# 5. Health check (pas d'auth requise)
curl "$BASE_URL/health"
```

### Test en ligne de commande, sans passer par l'API

```bash
python test_cli.py "https://exemple.com/video.mp4"
```
Charge le modèle, transcrit, affiche le SRT sur stdout (utile pour déboguer `transcriber.py` isolément).

---

## Configuration (`transcriber.py`)

Tous les réglages sont regroupés en tête de fichier :

| Paramètre | Valeur actuelle | Rôle |
|---|---|---|
| `MODEL_SIZE` | `"small"` | Taille du modèle Whisper |
| `DEVICE` | `"cpu"` | `"cpu"` ou `"cuda"` |
| `COMPUTE_TYPE` | `"int8"` | `"int8"` (CPU), `"float16"`/`"int8_float16"` (GPU) |
| `LANGUAGE` | `None` | Code langue forcé, ou `None` = auto-détection |
| `LANGUAGE_DETECTION_THRESHOLD` | `0.8` | Confiance mini avant d'arrêter la détection de langue |
| `LANGUAGE_DETECTION_SEGMENTS` | `3` | Nb de segments testés pour atteindre le seuil |
| `MAX_CUE_DURATION` | `6.0` s | Durée max d'un sous-titre |
| `MAX_CUE_CHARS` | `84` | Caractères max par sous-titre (~2 lignes de 42) |
| `MAX_WORD_GAP` | `0.8` s | Coupe un sous-titre si le silence entre 2 mots dépasse ce seuil |
| `WRAP_WIDTH` | `42` | Largeur d'une ligne de sous-titre |
| `VAD_ENABLED` | `True` | Détection d'activité vocale (filtre les silences) |
| `VAD_PARAMETERS` | `min_silence_duration_ms=500, speech_pad_ms=400` | Réglages VAD, volontairement peu agressifs pour ne pas rogner la parole |

D'autres variables d'environnement (`app.py`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `API_KEY` | *(obligatoire)* | Clé d'authentification des routes `/jobs*` |
| `OUTPUT_DIR` | `~/whisper-service/output` | Dossier où sont écrits temporairement les `.srt`/`.vtt` |
| `UPLOAD_DIR` | `~/whisper-service/uploads` | Dossier où sont écrits temporairement les fichiers uploadés (mp4/wav), supprimés dès la fin du job |

En Docker, `OUTPUT_DIR=/data/output`, `UPLOAD_DIR=/data/uploads` et `HF_HOME=/data/hf-cache` sont fixés dans le `Dockerfile`.

---

## Démarrage local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export API_KEY=une-cle-de-test
uvicorn app:app --reload
```

Ou via le script fourni (charge `.venv` et `.env`, écoute sur `0.0.0.0:8000`) :

```bash
./run_local.sh
```

---

## Démarrage avec Docker

```bash
cp .env.example .env   # puis y renseigner une vraie API_KEY
docker compose up -d --build
```

Architecture des conteneurs (`docker-compose.yml`) :

- **`whisper-service`** : construit depuis `Dockerfile`, écoute en interne sur le port `8140`, **non exposé directement** sur l'hôte. Volumes : `./output` (sorties, éphémères) et `hf-cache` (cache du modèle Whisper, persistant entre redémarrages).
- **`caddy`** : reverse proxy exposé sur `80`/`443`, TLS automatique via Let's Encrypt selon le domaine défini dans `Caddyfile`. Seul point d'entrée public.

---

## Déploiement en production

Voir [DEPLOY.md](DEPLOY.md) pour la procédure complète (prérequis serveur macOS, configuration DNS/domaine, build, vérification, exploitation courante, rotation de clé).

Points essentiels :
- Domaine avec enregistrement DNS **A** vers l'IP publique du serveur, ports **80**/**443** ouverts.
- Éditer `Caddyfile` avec le vrai domaine avant le build.
- `docker compose logs -f whisper-service` / `caddy` pour les logs.
- Mise à jour : `git pull && docker compose up -d --build`.
- `restart: unless-stopped` relance les conteneurs après crash/reboot du daemon Docker (penser à activer le démarrage auto de Docker Desktop sur macOS).

---

## Sécurité et bonnes pratiques

- **Clé API** : ne jamais committer `.env` (déjà exclu via `.gitignore`), ne pas la partager en clair hors d'un canal sécurisé. Comparaison en temps constant côté serveur (`secrets.compare_digest`) pour éviter les attaques par timing.
- **Fichiers éphémères** : les `.srt`/`.vtt` sont supprimés du disque dès leur premier téléchargement — aucune sauvegarde à prévoir côté serveur ; le client doit conserver le fichier après récupération.
- **Traitement séquentiel** : un seul job à la fois, pas d'isolation multi-tenant particulière au-delà de la clé API — à garder en tête si plusieurs clients partagent la même instance.
- **CPU uniquement** en configuration actuelle (`DEVICE="cpu"`) : temps de traitement proportionnel à la durée des vidéos (observé : ~35s pour ~1min45 de vidéo en test).
