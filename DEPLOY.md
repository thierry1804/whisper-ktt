# Déploiement sur le serveur macOS

## Prérequis

- Docker Desktop pour macOS (ou Colima/OrbStack) installé et lancé.
- Un nom de domaine dont l'enregistrement DNS **A** pointe vers l'IP publique du serveur macOS (nécessaire pour le HTTPS automatique de Caddy).
- Ports **80** et **443** ouverts sur le pare-feu / la box vers le serveur (80 sert au challenge ACME de Let's Encrypt, 443 au trafic HTTPS).

## 1. Récupérer le code

```bash
git clone <url-du-repo> whisper-service
cd whisper-service
```

## 2. Configurer le domaine

Éditer `Caddyfile` et remplacer le placeholder par le vrai domaine :

```
sous-titres.votre-domaine.tld {
    reverse_proxy whisper-service:8140
}
```

## 3. Générer la clé API

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copier la valeur générée dans `.env` :

```
API_KEY=<valeur générée>
```

Cette clé doit être transmise à la plateforme cliente — elle devra l'envoyer dans le header `X-API-Key` sur chaque appel à `/jobs*`.

## 4. Build et démarrage

```bash
docker compose up -d --build
```

- `whisper-service` n'est pas exposé directement sur l'hôte : seul Caddy (80/443) est joignable depuis l'extérieur, tout le trafic transite par le reverse proxy.
- Le modèle Whisper est téléchargé au premier démarrage puis mis en cache dans le volume `hf-cache` (pas de re-téléchargement aux redémarrages suivants).

## 5. Vérification

```bash
curl https://sous-titres.votre-domaine.tld/health
# {"status":"ok","en_file":0}
```

Test complet avec la clé API :

```bash
API_KEY=$(grep API_KEY .env | cut -d= -f2)

curl -X POST https://sous-titres.votre-domaine.tld/jobs \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://exemple.com/video.mp4"}'
# → {"id": "...", "status": "queued", ...}

curl https://sous-titres.votre-domaine.tld/jobs/<id> -H "X-API-Key: $API_KEY"
# → poll jusqu'à "status":"done"

curl https://sous-titres.votre-domaine.tld/jobs/<id>/srt -H "X-API-Key: $API_KEY" -o sous-titres.srt
# le fichier est supprimé côté serveur juste après ce téléchargement
```

## Exploitation courante

- **Logs** : `docker compose logs -f whisper-service` / `docker compose logs -f caddy`
- **Mise à jour** :
  ```bash
  git pull
  docker compose up -d --build
  ```
- **Redémarrage automatique** : `restart: unless-stopped` relance les conteneurs après un crash ou un redémarrage du daemon Docker. Sur macOS, penser à activer le démarrage automatique de Docker Desktop à la connexion si le service doit survivre à un reboot du serveur.
- **Rotation de la clé API** : changer `API_KEY` dans `.env` puis `docker compose up -d` (recrée uniquement `whisper-service`) ; prévenir la plateforme cliente du changement.

## Points d'attention

- Les fichiers `.srt`/`.vtt` sont éphémères : supprimés du disque dès leur premier téléchargement. Aucune sauvegarde à prévoir sur `./output/`.
- `.env` contient un secret : ne jamais le committer (déjà exclu via `.gitignore`), ne pas le partager en clair hors d'un canal sécurisé.
- Le modèle tourne en CPU (`faster-whisper`, pas d'accélération GPU/Metal sur macOS) — prévoir un temps de traitement proportionnel à la durée des vidéos (observé : ~35s pour ~1min45 de vidéo en environnement de test).
