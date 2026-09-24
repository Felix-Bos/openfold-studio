#!/usr/bin/env bash
# Installe OpenFold3-MLX (depot externe) a une version figee et applique
# les correctifs de ce projet. A lancer depuis la racine du depot.
set -euo pipefail

REPO_URL="https://github.com/latent-spacecraft/openfold-3-mlx.git"
PINNED_COMMIT="eeac37eb82dc2b80cf043eb26105a16d2493d052"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${OPENFOLD_PROJECT_DIR:-$ROOT_DIR/openfold-3-mlx}"

if [ ! -d "$TARGET_DIR/.git" ]; then
    echo "Clonage de OpenFold3-MLX dans $TARGET_DIR"
    git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"
git checkout --quiet "$PINNED_COMMIT"

for patch in "$ROOT_DIR"/patches/*.patch; do
    if git apply --check "$patch" 2>/dev/null; then
        git apply "$patch"
        echo "Correctif applique : $(basename "$patch")"
    else
        echo "Correctif deja applique ou incompatible : $(basename "$patch")"
    fi
done

echo "Installation de l'environnement OpenFold3-MLX (.venv)"
chmod +x ./install.sh
./install.sh

echo "OpenFold3-MLX pret dans $TARGET_DIR"
