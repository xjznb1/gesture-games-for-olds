#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"
PATCHELF_BIN="$ROOT_DIR/venv/bin/patchelf"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "未找到 Python 虚拟环境: $PYTHON_BIN"
  echo "请先在项目根目录创建并安装依赖。"
  exit 1
fi

if [[ ! -x "$PATCHELF_BIN" ]]; then
  echo "未找到 patchelf，正在安装..."
  "$PYTHON_BIN" -m pip install cx_Freeze
fi

cd "$ROOT_DIR"

"$PYTHON_BIN" -m pip install --upgrade cx_Freeze

mkdir -p "$ROOT_DIR/fonts"
if [[ ! -f "$ROOT_DIR/fonts/wqy-zenhei.ttc" ]]; then
  if [[ -f "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc" ]]; then
    cp -a "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc" "$ROOT_DIR/fonts/wqy-zenhei.ttc"
  elif [[ -f "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf" ]]; then
    cp -a "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf" "$ROOT_DIR/fonts/wqy-zenhei.ttc"
  else
    echo "警告: 未找到可用的中文字体，将继续打包但可能出现缺字。"
  fi
fi

rm -rf "$ROOT_DIR/build/exe."* "$ROOT_DIR/dist/cognitive_training_game"

PATH="$ROOT_DIR/venv/bin:$PATH" "$PYTHON_BIN" setup_freeze.py build_exe

BUILD_EXE_DIR="$(find "$ROOT_DIR/build" -maxdepth 1 -type d -name 'exe.*' | head -n 1)"
if [[ -z "$BUILD_EXE_DIR" ]]; then
  echo "未找到 build/exe.* 目录，打包失败。"
  exit 1
fi

mkdir -p "$ROOT_DIR/dist"
mkdir -p "$ROOT_DIR/dist/cognitive_training_game"
cp -a "$BUILD_EXE_DIR"/. "$ROOT_DIR/dist/cognitive_training_game/"

echo ""
echo "打包完成。输出目录: $ROOT_DIR/dist/cognitive_training_game"
echo "运行程序: $ROOT_DIR/dist/cognitive_training_game/cognitive_training_game"
