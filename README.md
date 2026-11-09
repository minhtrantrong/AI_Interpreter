# Translate Realtime

A fully local real-time speech-to-text and translation app. The frontend is React +
TypeScript; the backend is FastAPI with a WebSocket streaming pipeline. Audio never
leaves the machine: Faster Whisper handles transcription and
`NLLB-200 distilled 600M` (CTranslate2, INT8/FP16) handles translation between
English, Vietnamese, Japanese, and Korean.

Vietnamese documentation: [README-vi.md](./README-vi.md)

## Features

- **Live microphone translation** — the browser resamples microphone audio to
  mono PCM16 at 16 kHz with an AudioWorklet and streams it over WebSocket.
- **Partial and final results** — low-latency partial transcripts use beam 1; each
  finished utterance is re-decoded with beam 5 before the final translation.
- **Automatic utterance detection** — an adaptive voice gate and Silero VAD skip
  silence, keep pre-roll audio, and auto-commit when the speaker stops.
- **Language handling** — source language can be `auto` or fixed; output is
  Vietnamese, English, Japanese, or Korean. Language is locked per utterance after
  a confident detection.
- **Runtime model switching** — choose `Whisper Small` (fast) or
  `Large-v3 Turbo` (more accurate) without restarting the backend.
- **Text translation** — type text instead of speaking and it is translated
  through the same pipeline.
- **GPU-friendly defaults** — one backend worker, INT8/FP16 compute types, and
  persistent model caches so VRAM and startup time stay predictable.

## Architecture

```text
Microphone
   │  AudioWorklet → PCM16 LE, mono, 16 kHz
   ▼
WebSocket `/ws/realtime`
   │
   ├─ faster-whisper-small / large-v3-turbo  → transcript
   ├─ language detection (script + lingua)   → source language
   └─ NLLB-200 distilled 600M (CTranslate2)  → translation
   │
   ▼
WebSocket events → React UI
```

Repository layout:

```text
backend/    FastAPI app, ASR/translation services, tests, model scripts
frontend/   React + Vite UI, AudioWorklet, Caddy config for production
scripts/    Remote deploy helpers (deploy.sh, deploy-status.sh, …)
compose.yaml    GPU backend + Caddy frontend stack
DEPLOYMENT.md   Production deployment guide
```

## Requirements

- Python 3.10 or newer for the backend.
- Node.js 22 and npm for the frontend.
- An NVIDIA GPU with CUDA 12 and cuDNN 9 for real model inference. The reference
  production machine is an RTX 3060 12 GB with ~5 GB VRAM free.
- About 20 GB of free disk space for CUDA images, dependencies, and model caches.
- No Ollama or any external inference service is required.

## First-time setup

Install the backend and download/convert the models:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[models,dev]'
cp .env.example .env
python scripts/prepare_models.py
python scripts/prepare_asr_models.py --model turbo
```

On Windows, create and activate the virtual environment with
`python -m venv .venv` and `.venv\Scripts\activate` instead.

`prepare_models.py` downloads `facebook/nllb-200-distilled-600M`, converts it to
CTranslate2, and writes the runtime copy to
`backend/models/nllb-200-distilled-600m-ct2`. Faster Whisper downloads its ASR
models into the Hugging Face cache on first use (or during the preparation step).

## Running locally

Terminal 1 — backend:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`, then check
`http://127.0.0.1:8000/health`.

Terminal 2 — frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, click **Bắt đầu nghe** (Start listening), and grant
microphone access. The first utterance may be slower while Whisper finishes loading
model weights.

If the backend is not on `ws://127.0.0.1:8000/ws/realtime`, set `VITE_WS_URL` in
`frontend/.env`:

```dotenv
VITE_WS_URL=ws://127.0.0.1:8000/ws/realtime
```

## Production with Docker (NVIDIA GPU)

The Compose stack runs the CUDA backend plus a Caddy frontend that proxies
WebSocket traffic and terminates HTTPS. Defaults use `large-v3-turbo` on the GPU
with a single worker so the models are not duplicated in VRAM.

```bash
cp .env.docker.example .env.docker
docker compose --env-file .env.docker up -d --build
```

On the first start the backend downloads the models, loads Turbo and NLLB-200 onto
the GPU, and only then reports healthy — this can take several minutes. Later
starts reuse the `model-data` volume.

To deploy from a workstation in one command, configure SSH once:

```bash
cp .deploy.conf.example .deploy.conf
nano .deploy.conf
./scripts/deploy.sh
```

`./scripts/deploy-status.sh` prints stack status and recent backend logs. See
[DEPLOYMENT.md](./DEPLOYMENT.md) for server preparation, domain/HTTPS setup, model
caching, and operations.

## Configuration

Backend settings come from environment variables or `backend/.env`. The most
important ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Backend bind address |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed WebSocket/CORS origins |
| `USE_FAKE_MODELS` | `false` | Run the protocol without loading real models |
| `PRELOAD_MODELS` | `false` (`true` in Docker) | Load models during startup |
| `ASR_DEFAULT_MODEL` | `small` (`turbo` in Docker) | Initial Whisper model |
| `ASR_MODEL` / `ASR_TURBO_MODEL` | `Systran/faster-whisper-small` / `large-v3-turbo` | Model IDs behind the `small` and `turbo` keys |
| `ASR_DEVICE` / `ASR_COMPUTE_TYPE` | `cuda` / `int8_float16` | Whisper device and quantization |
| `ASR_NUM_WORKERS` | `1` | Keep at `1`; each worker loads its own copy of the models |
| `ASR_FINAL_BEAM_SIZE` | `5` | Beam size used to re-decode a finished utterance |
| `SILENCE_FINALIZE_MS` | `900` | Silence needed to auto-commit an utterance |
| `MAX_UTTERANCE_SEC` | `45` | Upper bound for a single utterance |
| `TRANSLATION_MODEL_PATH` | `models/nllb-200-distilled-600m-ct2` | Converted NLLB model directory |
| `TRANSLATION_DEVICE` / `TRANSLATION_COMPUTE_TYPE` | `cuda` / `int8_float16` | Translation device and quantization |
| `TRANSLATION_BEAM_SIZE` | `3` | Translation beam size |
| `TRANSLATION_INTERVAL_MS` | `900` | Minimum interval between partial translations |

Frontend build-time variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_WS_URL` | `ws://<host>:8000/ws/realtime` in dev, same-origin `/ws/realtime` in production | WebSocket endpoint |
| `VITE_DEFAULT_ASR_MODEL` | `small` (`turbo` in Docker) | Initially selected ASR model |

For the full list of tuning options and audio/VAD behavior, see
[backend/README.md](./backend/README.md).

## WebSocket protocol

Connect with:

```text
ws://127.0.0.1:8000/ws/realtime?source=auto&target=en&asr_model=small
```

Supported language codes are `auto`, `vi`, `en`, `ja`, and `ko`. The target
language cannot be `auto`.

Send binary frames containing PCM16 little-endian, mono, 16 kHz audio
(100–200 ms per frame). Control messages are JSON:

```json
{"type":"commit"}
{"type":"reset","clear_context":true}
{"type":"set_languages","source":"vi","target":"ja"}
{"type":"set_asr_model","model":"turbo"}
{"type":"translate_text","text":"Hello world"}
{"type":"ping"}
{"type":"end"}
```

The two valid model keys are `small` and `turbo`. `small` is preloaded at startup;
`turbo` is loaded onto the GPU the first time it is selected and stays cached until
the backend stops.

The server emits `session.ready`, `session.model_loading`, `session.model`,
`transcript.partial`, `translation.partial`, `transcript.final`,
`translation.final`, and `utterance.end` events. The backend also commits
automatically when it detects silence.

Clients must resample to mono 16 kHz themselves — do not send WebM/Opus straight
from `MediaRecorder`.

## Tests and smoke scripts

```bash
cd backend
pytest
python scripts/translation_smoke.py
python scripts/ws_model_switch_smoke.py
python scripts/ws_audio_smoke.py /path/to/audio.wav --model turbo --target vi --realtime
python scripts/ws_silence_smoke.py --model turbo --seconds 5 --noise-rms 0.006
```

To exercise the WebSocket protocol without a GPU, set `USE_FAKE_MODELS=true` and run
`python scripts/ws_smoke.py`.

Frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## Performance notes

- Keep `ASR_NUM_WORKERS=1`; extra Uvicorn workers duplicate every model in VRAM.
- `ASR_DEFAULT_MODEL=turbo` favors accuracy. Switch to `small` when GPU compute is
  tight or latency matters more.
- Keep `int8_float16` for both ASR and translation on a 12 GB GPU.
- Do not set thread counts above the number of physical CPU cores; 4 threads for
  ASR and 4 for translation are safe starting points.
- If you hit CUDA out-of-memory, check `nvidia-smi` for other processes, then set
  `ASR_DEFAULT_MODEL=small` and restart the stack.

## Documentation

- [DEPLOYMENT.md](./DEPLOYMENT.md) — Docker/GPU deployment, HTTPS, and operations.
- [backend/README.md](./backend/README.md) — backend pipeline, configuration, and
  WebSocket details.
- [frontend/DESIGN.md](./frontend/DESIGN.md) — UI design notes.
- [README-vi.md](./README-vi.md) — Vietnamese version of this document.
