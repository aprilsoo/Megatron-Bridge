#!/usr/bin/env bash
# ==============================================================================
# Megatron-Bridge 打包脚本
# 功能：将目录打包成 tar.gz，文件名包含 git commit hash 和时间戳
# ==============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# 获取项目名称（目录名）
PROJECT_NAME=$(basename "$SCRIPT_DIR")

# 获取 git commit hash（短格式）
if git rev-parse --git-dir > /dev/null 2>&1; then
    GIT_HASH=$(git rev-parse --short HEAD)
else
    GIT_HASH="no-git"
fi

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 生成文件名
OUTPUT_NAME="${PROJECT_NAME}_${GIT_HASH}_${TIMESTAMP}.tar.gz"
OUTPUT_PATH="../${OUTPUT_NAME}"

echo "[INFO] Building ${PROJECT_NAME}..."
echo "[INFO] Git commit: ${GIT_HASH}"
echo "[INFO] Timestamp: ${TIMESTAMP}"
echo "[INFO] Output: ${OUTPUT_PATH}"

# 打包（排除常见无需打包的文件）
tar czf "$OUTPUT_PATH" \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='.uv-cache' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='*.tar.gz' \
    -C .. \
    "$PROJECT_NAME"

echo "[INFO] Build completed: ${OUTPUT_PATH}"
echo "[INFO] Size: $(du -h "$OUTPUT_PATH" | cut -f1)"