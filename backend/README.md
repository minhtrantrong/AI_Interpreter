# Translate Realtime Backend

Backend FastAPI/WebSocket chạy hoàn toàn local theo luồng:

```text
PCM16 microphone -> faster-whisper-small / large-v3-turbo -> nhận diện ngôn ngữ
                 -> NLLB-200 600M (CTranslate2 INT8/FP16) -> events WebSocket
```

Stack này không cần Ollama. Cả STT và dịch đều chạy trên GPU NVIDIA; model dịch đã
được chuyển sang CTranslate2 để giảm dung lượng và độ trễ.

## 1. Cài đặt lần đầu

Yêu cầu Python 3.12, CUDA 12 và cuDNN 9.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[models,dev]'
cp .env.example .env
python scripts/prepare_models.py
python scripts/prepare_asr_models.py --model turbo
```

`prepare_models.py` tải `facebook/nllb-200-distilled-600M`, chuyển sang
CTranslate2 INT8/FP16 và lưu bản chạy tại
`backend/models/nllb-200-distilled-600m-ct2`. Faster Whisper
tự tải model STT vào cache trong lần chạy đầu.

## 2. Khởi động

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Chờ log `Application startup complete`, sau đó kiểm tra
`http://127.0.0.1:8000/health`. Không chạy `ollama serve`.

Các cấu hình chính trong `.env`:

```dotenv
USE_FAKE_MODELS=false
PRELOAD_MODELS=true
ASR_DEFAULT_MODEL=small
ASR_MODEL=Systran/faster-whisper-small
ASR_TURBO_MODEL=large-v3-turbo
ASR_DEVICE=cuda
ASR_COMPUTE_TYPE=int8_float16
ASR_CPU_THREADS=4
ASR_NUM_WORKERS=1
ASR_BEAM_SIZE=1
ASR_FINAL_BEAM_SIZE=5
ASR_STEP_MS=600
ASR_MAX_WINDOW_SEC=12
ASR_STABILITY_MARGIN_SEC=2.5
ASR_MAX_NO_SPEECH_PROB=0.65
ASR_MIN_AVG_LOGPROB=-1.0
ASR_VAD_THRESHOLD=0.6
ASR_VAD_MIN_SPEECH_MS=250
ASR_SUPPRESS_PROMOTIONAL_HALLUCINATIONS=true
SPEECH_RMS_THRESHOLD=0.01
SPEECH_START_MS=250
SPEECH_PREROLL_MS=350
NOISE_FLOOR_MULTIPLIER=2.5
TRANSLATION_MODEL_PATH=models/nllb-200-distilled-600m-ct2
TRANSLATION_DEVICE=cuda
TRANSLATION_COMPUTE_TYPE=int8_float16
TRANSLATION_INTRA_THREADS=4
TRANSLATION_BEAM_SIZE=3
TRANSLATION_INTERVAL_MS=900
```

Partial transcript dùng beam 1 và giữ liên tục tối đa 12 giây ngữ cảnh. Local
agreement chỉ cắt phần cũ khi cửa sổ thực sự vượt giới hạn, đồng thời giữ lại 2,5
giây cuối để không làm mất từ khi nói nhanh. Khi kết thúc câu, backend giải mã lại
buffer đầy đủ bằng beam 5 để sửa các sai lệch tạm thời. Ngôn ngữ được khóa trong
phạm vi một câu sau lần phát hiện đáng tin cậy đầu tiên. Bản dịch dài được chia
theo câu và chạy theo batch để hạn chế lặp từ.
Voice gate không gửi audio sang ASR cho tới khi phát hiện ít nhất 250 ms giọng nói
liên tục. Ngưỡng tự thích nghi theo nhiễu nền, còn pre-roll 350 ms giữ lại âm đầu
câu. Partial segment có xác suất `no speech` cao hoặc log-probability thấp bị bỏ
qua để tránh Whisper tự sinh chữ khi microphone im lặng. Silero VAD dùng ngưỡng
0,6 và yêu cầu tối thiểu 250 ms giọng nói. Ở kết quả final, backend chỉ cứu một
segment yếu khi nó được kẹp giữa hai segment đáng tin cậy và liền mạch; segment
yếu ở đầu/cuối bị loại để không nhận nhầm câu sinh ra trong khoảng im lặng.
Backend còn đối chiếu timestamp từng segment với vùng có giọng nói từ một lượt
VAD không padding và loại các câu CTA đăng ký kênh/video thường gặp trong dữ liệu
phụ đề khi `ASR_SUPPRESS_PROMOTIONAL_HALLUCINATIONS=true`.

## 3. WebSocket protocol

Kết nối:

```text
ws://127.0.0.1:8000/ws/realtime?source=auto&target=en&asr_model=small
```

Mã ngôn ngữ hỗ trợ: `auto`, `vi`, `en`, `ja`, `ko`. Ngôn ngữ đích không nhận
`auto`.

Client gửi binary frame PCM16 little-endian, mono, 16 kHz và có thể gửi các lệnh:

```json
{"type":"commit"}
{"type":"reset","clear_context":true}
{"type":"set_languages","source":"vi","target":"ja"}
{"type":"set_asr_model","model":"turbo"}
{"type":"ping"}
{"type":"end"}
```

Hai khóa model hợp lệ là `small` và `turbo`. Small được tải sẵn khi backend khởi
động; Turbo được tải vào GPU ở lần chọn đầu tiên và được cache cho tới khi tắt
backend. Server trả các event `session.ready`, `session.model_loading`,
`session.model`, `transcript.partial`,
`translation.partial`, `transcript.final`, `translation.final` và
`utterance.end`. Backend cũng tự commit khi phát hiện khoảng im lặng.

Frontend phải dùng AudioWorklet để resample microphone về mono 16 kHz; không gửi
trực tiếp WebM/Opus từ `MediaRecorder`. Mỗi frame nên chứa 100–200 ms audio.

## 4. Kiểm tra

```bash
pytest
python scripts/translation_smoke.py
python scripts/ws_model_switch_smoke.py
python scripts/ws_audio_smoke.py /path/to/audio.wav --model turbo --target vi --realtime
python scripts/ws_silence_smoke.py --model turbo --seconds 5 --noise-rms 0.006
```

Muốn kiểm tra giao thức mà không load GPU, đặt `USE_FAKE_MODELS=true` rồi chạy
`python scripts/ws_smoke.py`.

## 5. Docker GPU

Docker production dùng image CUDA 12.3.2 + cuDNN 9, Uvicorn một worker và volume
model bền vững. Chạy toàn bộ stack theo [DEPLOYMENT.md](../DEPLOYMENT.md). Không
chạy nhiều backend worker vì mỗi process sẽ nạp một bản model riêng lên GPU.
