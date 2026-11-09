#!/bin/sh
set -eu

repo_url="https://nvidia.github.io/libnvidia-container/stable/deb"
keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
source_list="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
temp_dir="$(mktemp -d /tmp/nvidia-toolkit.XXXXXX)"

cleanup() {
  rm -rf "$temp_dir"
}
trap cleanup EXIT INT TERM

echo "[1/5] Cài công cụ cần thiết"
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

echo "[2/5] Cài NVIDIA repository key"
curl -fsSL "https://nvidia.github.io/libnvidia-container/gpgkey" \
  -o "$temp_dir/gpgkey"
sudo gpg --dearmor --yes -o "$keyring" "$temp_dir/gpgkey"

echo "[3/5] Thêm NVIDIA Container Toolkit repository"
curl -fsSL "$repo_url/nvidia-container-toolkit.list" \
  -o "$temp_dir/nvidia-container-toolkit.list"
sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
  "$temp_dir/nvidia-container-toolkit.list" \
  > "$temp_dir/nvidia-container-toolkit-signed.list"
sudo install -m 0644 "$temp_dir/nvidia-container-toolkit-signed.list" "$source_list"

echo "[4/5] Cài và cấu hình NVIDIA Container Toolkit"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker

echo "[5/5] Khởi động lại Docker"
sudo systemctl restart docker

echo "Hoàn tất. Hãy chạy lệnh kiểm tra GPU được hướng dẫn tiếp theo."
