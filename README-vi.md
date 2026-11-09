# Translate Realtime

Ứng dụng React + FastAPI WebSocket chạy speech-to-text và dịch thuật hoàn toàn
local. Stack dùng Faster Whisper cho STT và `NLLB-200 distilled 600M`
CTranslate2 INT8 cho dịch Anh, Việt, Nhật, Hàn.

## Cài model lần đầu

```bash
cd backend
source .venv/bin/activate
pip install -e '.[models,dev]'
python scripts/prepare_models.py
```

## Khởi động

Terminal 1:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Mở `http://localhost:5173`, nhấn **Bắt đầu nghe** và cấp quyền microphone.
Không cần khởi động Ollama. Lần đầu Faster Whisper có thể cần tải model STT;
hãy chờ backend báo `Application startup complete`.

Xem [backend/README.md](backend/README.md) để biết cấu hình và cách kiểm tra chi
tiết. Frontend dùng `VITE_WS_URL` trong `frontend/.env` nếu backend không chạy ở
`ws://127.0.0.1:8000/ws/realtime`.

## Chạy production bằng Docker GPU

Dự án có sẵn `compose.yaml` gồm backend CUDA và frontend Caddy hỗ trợ WebSocket,
HTTPS. Cấu hình mặc định production dùng Large-v3-Turbo trên GPU với một worker
để không nhân đôi VRAM.

```bash
cp .env.docker.example .env.docker
docker compose --env-file .env.docker up -d --build
```

Xem hướng dẫn server, HTTPS, model cache và vận hành tại
[DEPLOYMENT.md](DEPLOYMENT.md).
