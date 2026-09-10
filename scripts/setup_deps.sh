#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_deps.sh – Download header-only dependencies
# Run from the project root: bash scripts/setup_deps.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# nlohmann/json (single header, MIT license)
NLH_URL="https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
NLH_DIR="$ROOT/third_party/nlohmann"

mkdir -p "$NLH_DIR"
if [ ! -f "$NLH_DIR/json.hpp" ]; then
    echo "[setup] Downloading nlohmann/json …"
    curl -fsSL "$NLH_URL" -o "$NLH_DIR/json.hpp"
    echo "[setup] OK: third_party/nlohmann/json.hpp"
else
    echo "[setup] nlohmann/json already present."
fi

echo ""
echo "[setup] All dependencies ready."
echo ""
echo "Build instructions:"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  make -j\$(nproc)"
echo ""
echo "Run tests:"
echo "  ./build/test_kernels"
echo ""
echo "Download a model:"
echo "  python python/convert_weights.py --model_id Qwen/Qwen2-0.5B --output_dir ./models/qwen2-0.5b"
echo ""
echo "Run inference:"
echo "  ./build/forgert ./models/qwen2-0.5b --prompt 'Hello' --max_tokens 20"
