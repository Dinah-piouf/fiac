#!/bin/bash
# ==================================================================================================
# FIAC -  Script de lancement automatique, donne l'environnement parfait Python pour l'application!
# Usage : bash launch.sh
# ==================================================================================================

set -e

VENV_DIR="faceid_env"
APP_FILE="face_recognition_app.py"

echo "============================================="
echo "   FIAC - Installation & Lancement"
echo "============================================="

# --- 1. Trouver une version Python compatible -----------------------------------
find_python() {
    for cmd in python3.11 python3.10 python3.12 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

echo ""
echo "Recherche d'une version Python compatible (3.10–3.12)..."
PYTHON=$(find_python || true)

if [ -z "$PYTHON" ]; then
    echo "ATTENTION Python 3.10–3.12 non trouvé. Installation via pyenv..."

    if ! command -v pyenv &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq curl git build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev libffi-dev
        curl https://pyenv.run | bash

        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"

        grep -qxF 'export PYENV_ROOT="$HOME/.pyenv"' ~/.bashrc || \
            echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        grep -qxF 'export PATH="$PYENV_ROOT/bin:$PATH"' ~/.bashrc || \
            echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        grep -qxF 'eval "$(pyenv init -)"' ~/.bashrc || \
            echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    else
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
    fi

    pyenv install -s 3.11.9
    pyenv local 3.11.9
    PYTHON=$(pyenv which python)
fi

echo "O.K. Python : $PYTHON ($($PYTHON --version))"

# ---- 2. Créer le venv si absent -----------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "---> Création de l'environnement virtuel..."
    "$PYTHON" -m venv "$VENV_DIR"
    echo "O.K. Environnement créé : $VENV_DIR/"
else
    echo "O.K. Environnement virtuel existant détecté."
fi

source "$VENV_DIR/bin/activate"
echo "O.K. Environnement activé : $(which python)"

# ---- 3. Mettre à jour pip ---------------------------------------------------
echo ""
echo "--> Mise à jour de pip..."
pip install --quiet --upgrade pip "setuptools<82" wheel

# ---- 4. Installer les dépendances ------------------------------------------
echo ""
echo " Vérification des dépendances..."

check_pkg() {
    python -c "import $1" &>/dev/null
}

install_if_missing() {
    local import_name="$1"
    shift
    local pip_pkgs=("$@")
    if ! check_pkg "$import_name" 2>/dev/null; then
        echo "  Installation de ${pip_pkgs[*]}..."
        pip install --quiet "${pip_pkgs[@]}"
    else
        echo "O.K. $import_name déjà installé"
    fi
}

install_if_missing "torch"           torch torchvision torchaudio
install_if_missing "tensorflow" tensorflow
install_if_missing "insightface"     insightface
install_if_missing "onnxruntime"     onnxruntime
install_if_missing "cv2"             opencv-python
install_if_missing "PIL"             pillow
install_if_missing "sklearn"         scikit-learn
install_if_missing "matplotlib"      matplotlib
install_if_missing "streamlit"       streamlit
install_if_missing "tqdm"            tqdm

# ---- 5. Vérifier que le fichier app existe ------------------------------------
echo ""
if [ ! -f "$APP_FILE" ]; then
    echo "NO NO Fichier $APP_FILE introuvable dans : $(pwd)"
    echo "Merci de placer face_recognition_app.py dans le même dossier que launch.sh !!!"
    exit 1
fi

# ---- 6. Lancer Streamlit ----------------------------------------------------
echo ""
echo "============================================="
echo "   Chaud Chaud devant! Lancement de l'application!"
echo "   Ouvrez http://localhost:8501"
echo "   Ctrl+C pour arrêter"
echo "============================================="
echo ""

streamlit run "$APP_FILE" \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false
