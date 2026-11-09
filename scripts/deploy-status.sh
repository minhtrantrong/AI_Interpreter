#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
config_file="${DEPLOY_CONFIG:-$project_dir/.deploy.conf}"

[[ -f "$config_file" ]] || {
  echo "Không tìm thấy $config_file" >&2
  exit 1
}

# shellcheck disable=SC1090
source "$config_file"
: "${SERVER_HOST:?Thiếu SERVER_HOST}"
: "${SERVER_USER:?Thiếu SERVER_USER}"
: "${SERVER_PATH:?Thiếu SERVER_PATH}"

SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"
if [[ "$SSH_KEY" == "~/"* ]]; then
  SSH_KEY="${HOME}/${SSH_KEY:2}"
fi

[[ "$SERVER_HOST" =~ ^[a-zA-Z0-9._:-]+$ ]] || {
  echo "SERVER_HOST chứa ký tự không hợp lệ" >&2
  exit 1
}
[[ "$SERVER_USER" =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "SERVER_USER chứa ký tự không hợp lệ" >&2
  exit 1
}
[[ "$SERVER_PATH" =~ ^/[a-zA-Z0-9._/-]+$ ]] || {
  echo "SERVER_PATH không hợp lệ" >&2
  exit 1
}
[[ "$SSH_PORT" =~ ^[0-9]+$ ]] || {
  echo "SSH_PORT phải là số" >&2
  exit 1
}

ssh_options=(-p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  ssh_options+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
fi

remote="$SERVER_USER@$SERVER_HOST"
ssh -tt "${ssh_options[@]}" "$remote" \
  "cd '$SERVER_PATH/current' && if docker info >/dev/null 2>&1; then docker compose --env-file .env.docker ps && docker compose --env-file .env.docker logs --tail=100 backend; else sudo docker compose --env-file .env.docker ps && sudo docker compose --env-file .env.docker logs --tail=100 backend; fi"
