# Docker Deployment on an NVIDIA GPU

This configuration is intended for a Linux machine with an RTX 3060 12 GB GPU
and approximately 5 GB of free VRAM. Production defaults to `Large-v3-Turbo`,
`int8_float16`, a single backend worker, and NLLB-200 distilled 600M with
CTranslate2. `Whisper Small` can still be selected in the UI.

## 1. Prepare the server

The server needs:

- An NVIDIA driver that supports CUDA 12.3 or later.
- Docker Engine and Docker Compose v2.
- NVIDIA Container Toolkit configured for Docker.
- At least 20 GB of free disk space is recommended for the CUDA image,
  dependencies, and model cache.

Check the GPU on the host and inside a container:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

If the second command fails, fix the NVIDIA Container Toolkit setup before
starting the application. Docker does not enforce a hard VRAM limit per
container; use `nvidia-smi` to ensure other processes leave at least 5 GB
available.

## 2. Configure

From the project root:

```bash
cp .env.docker.example .env.docker
```

If you have a domain pointing to the server, update:

```dotenv
SITE_ADDRESS=translate.example.com
```

Caddy will automatically obtain and renew an HTTPS certificate when the domain
is reachable from the Internet on ports 80 and 443. HTTPS is required for the
browser to grant microphone access when users are not accessing the app through
`localhost`.

For HTTP testing on a local network, leave `SITE_ADDRESS=:80`. The UI will be
available, but the browser may block microphone access when using the server's
IP address.

## 3. Build and run

```bash
docker compose --env-file .env.docker build
docker compose --env-file .env.docker up -d
docker compose --env-file .env.docker logs -f backend
```

On the first startup, the backend will:

1. Download NLLB-200 distilled 600M and convert it to CTranslate2 format.
2. Download Whisper Small to the disk cache.
3. Download Large-v3-Turbo to the disk cache.
4. Load Turbo and NLLB-200 onto the GPU before reporting as healthy.

The initial setup may take several minutes, depending on network and CPU speed.
Subsequent startups reuse the `model-data` volume and do not download the models
again.

### Deploy to the server with one command from your local machine

Create the SSH configuration once:

```bash
cp .deploy.conf.example .deploy.conf
nano .deploy.conf
```

Make sure `.env.docker` contains the production configuration you want. For
subsequent deployments, run this from the project root:

```bash
./scripts/deploy.sh
```

The script packages the exact code state on your local machine (including
uncommitted changes), excluding models, virtual environments, `node_modules`,
secret files, and build output. It then verifies the checksum, builds on the
server, starts Compose, and waits for both services to become healthy. Each
release is stored at `SERVER_PATH/releases/<release-id>`. The
`SERVER_PATH/current` link is updated only after the health check succeeds.

The script tries to run `docker` directly first. If the SSH user is not in the
`docker` group, it falls back to `sudo docker` and prompts for the sudo password
in the terminal. The password is not written to `.deploy.conf` or uploaded to
the server. For CI without a terminal, an administrator must grant direct Docker
access or configure suitable `sudo NOPASSWD` permissions.

View the deployment status and the latest 100 lines of backend logs from your
local machine:

```bash
./scripts/deploy-status.sh
```

Check the status:

```bash
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs --tail=100 backend
curl http://127.0.0.1/health
nvidia-smi
```

When using an HTTPS domain:

```bash
curl https://translate.example.com/health
```

## 4. Operations

Update the code and rebuild:

```bash
git pull
docker compose --env-file .env.docker up -d --build
```

Restart without rebuilding:

```bash
docker compose --env-file .env.docker restart
```

Follow the logs:

```bash
docker compose --env-file .env.docker logs -f --tail=200
```

Stop the application while keeping the model cache:

```bash
docker compose --env-file .env.docker down
```

Do not add `--volumes` to the `down` command if you want to keep the downloaded
models.

## 5. Performance tuning

- Keep `ASR_NUM_WORKERS=1`: each separate Uvicorn worker loads its own copy of
  every model and increases VRAM usage.
- `ASR_DEFAULT_MODEL=turbo` prioritizes accuracy. Change it to `small` if GPU
  compute is limited or you need faster responses.
- Keep `ASR_COMPUTE_TYPE=int8_float16` and
  `TRANSLATION_COMPUTE_TYPE=int8_float16`; these are suitable for the available
  VRAM.
- `ASR_FINAL_BEAM_SIZE=5` refines the result at the end of an utterance; partial
  results use beam 1 to keep latency low.
- Do not set the thread count higher than the number of physical CPU cores. Four
  threads for ASR and four for translation are safe starting points.

If you encounter a CUDA out-of-memory error, first check for other processes
with `nvidia-smi`. Then set `ASR_DEFAULT_MODEL=small` and restart the stack.
