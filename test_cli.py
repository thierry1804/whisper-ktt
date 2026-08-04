import sys, time, logging

# Active les logs internes de faster-whisper (chargement, VAD)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

from transcriber import Transcriber, to_srt

url = sys.argv[1]

print(" Chargement du modèle (télécharge au 1er lancement)", flush=True)
t0 = time.time()
t = Transcriber()
print(f" Modèle prêt en {time.time() - t0:.1f}s", flush=True)

print(" Extraction audio + transcription", flush=True)
t1 = time.time()
segments = t.transcribe(url)
print(f" Transcription terminée en {time.time() - t1:.1f}s "
      f"({len(segments)} segments)\n", flush=True)

print(to_srt(segments))

