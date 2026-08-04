# whisper-service

Service de sous-titrage : soumet une URL de vidéo/audio, extrait l'audio via ffmpeg et transcrit avec [faster-whisper](https://github.com/SYSTRAN/faster-whisper), renvoie des sous-titres `.srt`/`.vtt`.

## Fonctionnement

1. `POST /jobs` avec une URL → renvoie un `id` de job (traitement asynchrone en file, un seul job à la fois).
2. `GET /jobs/{id}` → statut du job (`queued`, `processing`, `done`, `error`).
3. `GET /jobs/{id}/srt` et `GET /jobs/{id}/vtt` → téléchargent le fichier correspondant. **Chaque fichier est supprimé du serveur juste après son téléchargement** (récupérable une seule fois).

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
