#!/usr/bin/env bash
# Installs OpenFold3-MLX (an external repository) at a pinned version and
# applies this project's patches. Run from the repository root.
set -euo pipefail

REPO_URL="https://github.com/latent-spacecraft/openfold-3-mlx.git"
PINNED_COMMIT="eeac37eb82dc2b80cf043eb26105a16d2493d052"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${OPENFOLD_PROJECT_DIR:-$ROOT_DIR/openfold-3-mlx}"

if [ ! -d "$TARGET_DIR/.git" ]; then
    echo "Cloning OpenFold3-MLX into $TARGET_DIR"
    git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
git checkout --quiet "$PINNED_COMMIT"

for patch in "$ROOT_DIR"/patches/*.patch; do
    if git apply --check "$patch" 2>/dev/null; then
        git apply "$patch"
        echo "Applied patch: $(basename "$patch")"
    else
        echo "Patch already applied or not applicable: $(basename "$patch")"
    fi
done

echo "Installing the OpenFold3-MLX environment (.venv)"
chmod +x ./install.sh
./install.sh

echo "OpenFold3-MLX is ready in $TARGET_DIR"
