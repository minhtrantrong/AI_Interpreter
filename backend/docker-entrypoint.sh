#!/bin/sh
set -eu

translation_model_path="${TRANSLATION_MODEL_PATH:-/models/nllb-200-distilled-600m-ct2}"

if [ ! -f "${translation_model_path}/model.bin" ]; then
  echo "[models] Đang chuẩn bị NLLB-200 CTranslate2 tại ${translation_model_path}"
  python /app/scripts/prepare_models.py --output "${translation_model_path}"
fi

echo "[models] Kiểm tra Whisper Small trong cache local"
python /app/scripts/prepare_asr_models.py --model small

if [ "${PREPARE_TURBO_CACHE:-true}" = "true" ]; then
  echo "[models] Kiểm tra Large-v3-Turbo trong cache local"
  python /app/scripts/prepare_asr_models.py --model turbo
fi

exec "$@"
