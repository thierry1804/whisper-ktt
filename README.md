# whisper-service

Service de sous-titrage : soumet une URL (mp4 ou wav) ou envoie directement un fichier (mp4 ou wav), extrait l'audio via ffmpeg et transcrit avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper), renvoie des sous-titres `.srt`/`.vtt`.

## Fonctionnement

1. `POST /jobs` (`multipart/form-data`) avec soit un champ `url` (mp4 ou wav), soit un champ `file` (upload direct d'un mp4 ou wav) → renvoie un `id` de job (traitement asynchrone en file, un seul job à la fois).
2. `GET /jobs/{id}` → statut du job (`queued`, `converting`, `transcribing`, `cancelling`, `cancelled`, `done`, `error`) et `progress` (0.0 à 1.0, remis à zéro à chaque changement de statut : `converting` = extraction audio via ffmpeg, `transcribing` = transcription Whisper).
3. `POST /jobs/{id}/cancel` → arrête un job en attente ou en cours. Si le job attendait en file, il passe directement à `cancelled`. S'il était en cours de traitement, il passe à `cancelling` (l'arrêt effectif du process en cours prend jusqu'à ~0.5s) puis `cancelled` une fois confirmé — repoller `GET /jobs/{id}` pour le voir. Renvoie 409 si le job est déjà terminé (`done`/`error`/`cancelled`).
4. `GET /jobs/{id}/srt` et `GET /jobs/{id}/vtt` → téléchargent le fichier correspondant. **Chaque fichier est supprimé du serveur juste après son téléchargement** (récupérable une seule fois).

Toutes les routes `/jobs*` nécessitent un header `X-API-Key`. `GET /health` reste public.

## Démarrage local (sans Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Test en ligne de commande, sans passer par l'API :

```bash
python test_cli.py "https://exemple.com/video.mp4"
```

Lancement de l'API :

```bash
export API_KEY=une-cle-de-test
uvicorn app:app --reload
```

## Démarrage avec Docker

```bash
cp .env.example .env   # puis y renseigner une vraie API_KEY
docker compose up -d --build
```

Le service est alors joignable via Caddy (TLS automatique) sur le domaine configuré dans `Caddyfile`.

## Déploiement en production

Voir [DEPLOY.md](DEPLOY.md) pour la procédure complète (serveur macOS, DNS, Caddy, rotation de la clé API).

## Configuration

Réglages ajustables dans `transcriber.py` : taille du modèle Whisper, device (CPU/GPU), langue (auto-détection ou forcée), seuils de découpage des sous-titres, paramètres VAD.
