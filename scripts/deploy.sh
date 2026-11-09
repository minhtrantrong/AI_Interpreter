#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
config_file="${DEPLOY_CONFIG:-$project_dir/.deploy.conf}"

fail() {
  echo "Lỗi: $*" >&2
  exit 1
}

if [[ ! -f "$config_file" ]]; then
  fail "Không tìm thấy $config_file. Hãy chạy: cp .deploy.conf.example .deploy.conf"
fi

# shellcheck disable=SC1090
source "$config_file"

: "${SERVER_HOST:?Thiếu SERVER_HOST trong .deploy.conf}"
: "${SERVER_USER:?Thiếu SERVER_USER trong .deploy.conf}"
: "${SERVER_PATH:?Thiếu SERVER_PATH trong .deploy.conf}"

SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"
DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-.env.docker}"
DEPLOY_HEALTH_TIMEOUT="${DEPLOY_HEALTH_TIMEOUT:-1800}"

[[ "$SERVER_HOST" =~ ^[a-zA-Z0-9._:-]+$ ]] \
  || fail "SERVER_HOST chứa ký tự không hợp lệ"
[[ "$SERVER_USER" =~ ^[a-zA-Z0-9._-]+$ ]] \
  || fail "SERVER_USER chứa ký tự không hợp lệ"
[[ "$SERVER_PATH" =~ ^/[a-zA-Z0-9._/-]+$ ]] \
  || fail "SERVER_PATH phải là đường dẫn tuyệt đối, không chứa khoảng trắng"
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || fail "SSH_PORT phải là số"
[[ "$DEPLOY_HEALTH_TIMEOUT" =~ ^[0-9]+$ ]] \
  || fail "DEPLOY_HEALTH_TIMEOUT phải là số giây"

if [[ "$DEPLOY_ENV_FILE" != /* ]]; then
  DEPLOY_ENV_FILE="$project_dir/$DEPLOY_ENV_FILE"
fi
[[ -f "$DEPLOY_ENV_FILE" ]] \
  || fail "Không tìm thấy file môi trường: $DEPLOY_ENV_FILE"

if [[ "$SSH_KEY" == "~/"* ]]; then
  SSH_KEY="${HOME}/${SSH_KEY:2}"
fi
if [[ -n "$SSH_KEY" && ! -f "$SSH_KEY" ]]; then
  fail "Không tìm thấy SSH key: $SSH_KEY"
fi

command -v ssh >/dev/null || fail "Máy local chưa cài ssh"
command -v scp >/dev/null || fail "Máy local chưa cài scp"
command -v tar >/dev/null || fail "Máy local chưa cài tar"
command -v sha256sum >/dev/null || fail "Máy local chưa cài sha256sum"

ssh_options=(
  -p "$SSH_PORT"
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
  -o StrictHostKeyChecking=accept-new
)
scp_options=(
  -P "$SSH_PORT"
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
  -o StrictHostKeyChecking=accept-new
)
if [[ -n "$SSH_KEY" ]]; then
  ssh_options+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
  scp_options+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
fi

git_suffix="local"
if command -v git >/dev/null && git -C "$project_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_suffix="$(git -C "$project_dir" rev-parse --short HEAD)"
fi
release_id="$(date -u +%Y%m%dT%H%M%SZ)-$git_suffix"
archive="$(mktemp "/tmp/translate-realtime-${release_id}.XXXXXX.tar.gz")"

cleanup() {
  rm -f -- "$archive"
}
trap cleanup EXIT INT TERM

echo "[1/5] Đóng gói source code: $release_id"
tar \
  --exclude='./.git' \
  --exclude='./.deploy.conf' \
  --exclude='./.env.docker' \
  --exclude='./backend/.env' \
  --exclude='./backend/.venv' \
  --exclude='./backend/models' \
  --exclude='./frontend/.env' \
  --exclude='./frontend/node_modules' \
  --exclude='./frontend/dist' \
  --exclude='*/.pytest_cache' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -C "$project_dir" \
  -czf "$archive" \
  .

checksum="$(sha256sum "$archive" | awk '{print $1}')"
remote="$SERVER_USER@$SERVER_HOST"
incoming="$SERVER_PATH/.incoming"

echo "[2/5] Kiểm tra kết nối SSH và chuẩn bị thư mục server"
ssh "${ssh_options[@]}" "$remote" \
  "mkdir -p '$incoming'"

echo "[3/5] Upload code và cấu hình Docker"
scp "${scp_options[@]}" "$archive" \
  "$remote:$incoming/$release_id.tar.gz"
scp "${scp_options[@]}" "$DEPLOY_ENV_FILE" \
  "$remote:$incoming/$release_id.env"
scp "${scp_options[@]}" "$script_dir/deploy-remote.sh" \
  "$remote:$incoming/$release_id-deploy.sh"

echo "[4/5] Build và deploy trên server"
# Remote helper được upload thành file để stdin của SSH vẫn là terminal. Nhờ đó
# sudo có thể hỏi mật khẩu an toàn nếu server chưa cấu hình NOPASSWD cho Docker.
ssh -tt "${ssh_options[@]}" "$remote" \
  "bash '$incoming/$release_id-deploy.sh' '$SERVER_PATH' '$release_id' '$checksum' '$DEPLOY_HEALTH_TIMEOUT'"

echo "[5/5] Hoàn tất deployment $release_id"
echo "Kiểm tra log bằng: ./scripts/deploy-status.sh"
