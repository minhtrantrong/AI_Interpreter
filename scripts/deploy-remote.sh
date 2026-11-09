#!/usr/bin/env bash
set -Eeuo pipefail

base_dir="${1:?Thiếu SERVER_PATH}"
release_id="${2:?Thiếu release id}"
expected_checksum="${3:?Thiếu checksum}"
health_timeout="${4:?Thiếu health timeout}"

incoming_dir="$base_dir/.incoming"
archive="$incoming_dir/$release_id.tar.gz"
uploaded_env="$incoming_dir/$release_id.env"
remote_helper="$incoming_dir/$release_id-deploy.sh"
releases_dir="$base_dir/releases"
release_dir="$releases_dir/$release_id"

fail() {
  echo "Lỗi deploy trên server: $*" >&2
  exit 1
}

cleanup_incoming() {
  rm -f -- "$archive" "$uploaded_env" "$remote_helper"
}
trap cleanup_incoming EXIT INT TERM

command -v sha256sum >/dev/null || fail "Server chưa cài sha256sum"
command -v docker >/dev/null || fail "Server chưa cài Docker"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  command -v sudo >/dev/null \
    || fail "User không có quyền Docker và server không có sudo"
  if sudo -n docker info >/dev/null 2>&1; then
    docker_cmd=(sudo -n docker)
  elif [[ -t 0 ]]; then
    echo "Docker yêu cầu sudo. Hãy nhập mật khẩu của user SSH:"
    sudo -v
    docker_cmd=(sudo docker)
  else
    fail "Docker cần sudo có mật khẩu nhưng phiên SSH không có TTY"
  fi
fi

"${docker_cmd[@]}" compose version >/dev/null 2>&1 \
  || fail "Server chưa có Docker Compose v2"

[[ -f "$archive" ]] || fail "Không tìm thấy archive đã upload"
[[ -f "$uploaded_env" ]] || fail "Không tìm thấy .env.docker đã upload"

actual_checksum="$(sha256sum "$archive" | awk '{print $1}')"
[[ "$actual_checksum" == "$expected_checksum" ]] \
  || fail "Checksum archive không khớp"

mkdir -p "$releases_dir"
[[ ! -e "$release_dir" ]] || fail "Release đã tồn tại: $release_id"
mkdir "$release_dir"

tar -xzf "$archive" -C "$release_dir"
install -m 600 "$uploaded_env" "$release_dir/.env.docker"
printf '\nDEPLOY_IMAGE_TAG=%s\n' "$release_id" >> "$release_dir/.env.docker"
[[ -f "$release_dir/compose.yaml" ]] \
  || fail "Archive không chứa compose.yaml"

compose=(
  "${docker_cmd[@]}" compose
  --env-file "$release_dir/.env.docker"
  -f "$release_dir/compose.yaml"
)

echo "Kiểm tra Docker Compose config"
"${compose[@]}" config --quiet

echo "Build image mới; container hiện tại vẫn tiếp tục phục vụ trong bước này"
"${compose[@]}" build

echo "Khởi động release mới"
"${compose[@]}" up -d --remove-orphans

service_health() {
  local service="$1"
  local container_id
  container_id="$("${compose[@]}" ps -q "$service")"
  if [[ -z "$container_id" ]]; then
    echo "missing"
    return
  fi
  "${docker_cmd[@]}" inspect \
    --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
    "$container_id"
}

echo "Chờ backend và frontend healthy (timeout ${health_timeout}s)"
started_at="$(date +%s)"
while true; do
  backend_health="$(service_health backend)"
  frontend_health="$(service_health frontend)"
  echo "  backend=$backend_health frontend=$frontend_health"

  if [[ "$backend_health" == "healthy" && "$frontend_health" == "healthy" ]]; then
    break
  fi
  if [[ "$backend_health" =~ ^(unhealthy|exited|dead|missing)$ ]] \
    || [[ "$frontend_health" =~ ^(unhealthy|exited|dead|missing)$ ]]; then
    echo "Release mới không healthy. Log gần nhất:" >&2
    "${compose[@]}" ps >&2 || true
    "${compose[@]}" logs --tail=150 >&2 || true
    fail "Container dừng hoặc unhealthy"
  fi

  now="$(date +%s)"
  if (( now - started_at >= health_timeout )); then
    echo "Hết thời gian chờ. Log gần nhất:" >&2
    "${compose[@]}" ps >&2 || true
    "${compose[@]}" logs --tail=150 >&2 || true
    fail "Health check timeout"
  fi
  sleep 5
done

ln -sfnT "$release_dir" "$base_dir/current"
echo "Release hiện tại: $release_dir"
"${compose[@]}" ps
