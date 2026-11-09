# Triển khai Docker trên NVIDIA GPU

Cấu hình này dành cho một máy Linux có RTX 3060 12 GB và còn khoảng 5 GB VRAM.
Production mặc định dùng `Large-v3-Turbo`, `int8_float16`, một backend worker và
NLLB-200 distilled 600M CTranslate2. `Whisper Small` vẫn có thể chọn trên giao diện.

## 1. Chuẩn bị server

Server cần có:

- NVIDIA driver hỗ trợ CUDA 12.3 trở lên.
- Docker Engine và Docker Compose v2.
- NVIDIA Container Toolkit đã được cấu hình cho Docker.
- Nên còn ít nhất 20 GB dung lượng đĩa cho CUDA image, dependency và model cache.

Kiểm tra GPU trên host và bên trong container:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

Nếu lệnh thứ hai lỗi, cần sửa NVIDIA Container Toolkit trước khi khởi động ứng
dụng. Docker không giới hạn VRAM cứng cho từng container; hãy dùng `nvidia-smi`
để bảo đảm các tiến trình khác thực sự còn ít nhất 5 GB.

## 2. Cấu hình

Tại thư mục gốc dự án:

```bash
cp .env.docker.example .env.docker
```

Nếu đã có domain trỏ về server, sửa:

```dotenv
SITE_ADDRESS=translate.example.com
```

Caddy sẽ tự xin và gia hạn chứng chỉ HTTPS khi domain truy cập được từ Internet
qua cổng 80/443. HTTPS là bắt buộc để trình duyệt cấp microphone khi người dùng
không truy cập bằng `localhost`.

Để test HTTP trong mạng local, giữ `SITE_ADDRESS=:80`. Khi đó giao diện vẫn mở
được nhưng microphone có thể bị trình duyệt chặn nếu dùng địa chỉ IP của server.

## 3. Build và chạy

```bash
docker compose --env-file .env.docker build
docker compose --env-file .env.docker up -d
docker compose --env-file .env.docker logs -f backend
```

Lần đầu backend sẽ:

1. Tải và chuyển NLLB-200 distilled 600M sang CTranslate2.
2. Tải Whisper Small vào cache đĩa.
3. Tải Large-v3-Turbo vào cache đĩa.
4. Nạp Turbo và NLLB-200 lên GPU rồi mới báo healthy.

Quá trình đầu tiên có thể kéo dài nhiều phút tùy mạng và CPU. Các lần khởi động
sau dùng volume `model-data`, không tải lại model.

### Deploy server bằng một lệnh từ máy local

Tạo cấu hình SSH một lần:

```bash
cp .deploy.conf.example .deploy.conf
nano .deploy.conf
```

Đảm bảo `.env.docker` chứa cấu hình production mong muốn. Từ những lần sau, chỉ
cần chạy tại thư mục gốc dự án:

```bash
./scripts/deploy.sh
```

Script đóng gói đúng trạng thái code đang có trên máy local (kể cả thay đổi chưa
commit), loại trừ model, virtualenv, `node_modules`, file secret và build output;
sau đó kiểm tra checksum, build trên server, chạy Compose và chờ hai service
healthy. Mỗi bản được lưu tại `SERVER_PATH/releases/<release-id>`, còn liên kết
`SERVER_PATH/current` chỉ được cập nhật sau khi health check thành công.

Script ưu tiên chạy `docker` trực tiếp. Nếu user SSH chưa thuộc nhóm `docker`, nó
tự chuyển sang `sudo docker` và hỏi mật khẩu sudo qua terminal. Mật khẩu không
được ghi vào `.deploy.conf` hay upload lên server. Với CI không có terminal, admin
cần cấp quyền Docker trực tiếp hoặc cấu hình sudo `NOPASSWD` phù hợp.

Xem trạng thái và 100 dòng log backend từ máy local:

```bash
./scripts/deploy-status.sh
```

Kiểm tra trạng thái:

```bash
docker compose --env-file .env.docker ps
docker compose --env-file .env.docker logs --tail=100 backend
curl http://127.0.0.1/health
nvidia-smi
```

Khi dùng domain HTTPS:

```bash
curl https://translate.example.com/health
```

## 4. Vận hành

Cập nhật code và dựng lại:

```bash
git pull
docker compose --env-file .env.docker up -d --build
```

Khởi động lại mà không build:

```bash
docker compose --env-file .env.docker restart
```

Xem log realtime:

```bash
docker compose --env-file .env.docker logs -f --tail=200
```

Dừng ứng dụng nhưng giữ model cache:

```bash
docker compose --env-file .env.docker down
```

Không thêm `--volumes` vào lệnh `down` nếu muốn giữ model đã tải.

## 5. Tinh chỉnh hiệu năng

- Giữ `ASR_NUM_WORKERS=1`: mỗi worker Uvicorn riêng sẽ nạp lại toàn bộ model và
  làm tăng VRAM.
- `ASR_DEFAULT_MODEL=turbo`: ưu tiên độ chính xác. Đổi sang `small` nếu server
  còn ít GPU compute hoặc cần phản hồi nhanh hơn.
- Giữ `ASR_COMPUTE_TYPE=int8_float16` và
  `TRANSLATION_COMPUTE_TYPE=int8_float16`: phù hợp mức VRAM còn lại.
- `ASR_FINAL_BEAM_SIZE=5` sửa kết quả ở cuối câu; partial dùng beam 1 để giữ độ
  trễ thấp.
- Không tăng số thread vượt quá số core CPU thực. Mặc định 4 thread cho STT và
  4 thread cho dịch là điểm bắt đầu an toàn.

Nếu gặp CUDA out-of-memory, trước tiên kiểm tra tiến trình khác bằng
`nvidia-smi`. Sau đó đổi `ASR_DEFAULT_MODEL=small` và khởi động lại stack.
