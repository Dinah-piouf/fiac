#!/bin/bash
# ===================================================================================================
# FACE ID — Script de lancement automatique, donne l'environnement parfait Python pour l'application!
# Usage : bash launch_fast.sh
# ===================================================================================================

set -e

VENV_DIR="faceid_fast_env"
APP_FILE="face_recognition_app.py"
REQUIRED_MINOR_MIN=10
REQUIRED_MINOR_MAX=12

# _____________________________________________________________________________
# 0. DIAGNOSTIC & NETTOYAGE INITIAL
# _____________________________________________________________________________
echo "============================================="
echo "    FIAC - Diagnostic de l'environnement     "
echo "============================================="
echo ""


# 0. détection et comptage des conflits


CONFLICTS=0

# 0a. Détecter les venvs actifs qui pourraient entrer en conflit
if [ -n "$VIRTUAL_ENV" ]; then
    echo "ATTENTION:  CONFLIT DÉTECTÉ : un venv est déjà actif --> $VIRTUAL_ENV"
    echo "Désactivation automatique..."
    deactivate 2>/dev/null || true
    CONFLICTS=$((CONFLICTS + 1))
fi

# 0b. Détecter conda actif
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "ATTENTION: CONFLIT DÉTECTÉ : environnement conda actif --> $CONDA_DEFAULT_ENV"
    echo "   -->  Exécute 'conda deactivate' manuellement puis relance ce script."
    CONFLICTS=$((CONFLICTS + 1))
fi

# 0c. Vérifier la version Python du venv existant (si présent)
if [ -d "$VENV_DIR" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
    if [ -f "$VENV_PYTHON" ]; then
        VENV_VER=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "inconnu")
        VENV_MINOR=$(echo "$VENV_VER" | cut -d. -f2)
        if [ "$VENV_MINOR" -gt "$REQUIRED_MINOR_MAX" ] || [ "$VENV_MINOR" -lt "$REQUIRED_MINOR_MIN" ] 2>/dev/null; then
            echo "ATTENTION:  CONFLIT DÉTECTÉ : le venv '$VENV_DIR' utilise Python $VENV_VER (incompatible)"
            echo "Suppression et recréation avec une version compatible..."
            rm -rf "$VENV_DIR"
            CONFLICTS=$((CONFLICTS + 1))
        fi
    fi
fi

# 0d. Détecter pyenv mal initialisé
if [ -d "$HOME/.pyenv" ] && ! command -v pyenv &>/dev/null; then
    echo "ATTENTION:  CONFLIT DÉTECTÉ : pyenv installé mais non initialisé dans ce shell"
    echo "   Initialisation automatique..."
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)" 2>/dev/null || true
    CONFLICTS=$((CONFLICTS + 1))
fi

# 0e. Vérifier l'espace disque (minimum 3 Go requis)
FREE_KB=$(df -k . | awk 'NR==2 {print $4}')
FREE_GB=$(echo "scale=1; $FREE_KB / 1048576" | bc 2>/dev/null || echo "?")
if [ "$FREE_KB" -lt 3145728 ] 2>/dev/null; then
    echo "ATTENTION:  AVERTISSEMENT : espace disque faible (${FREE_GB} Go libres)"
    echo "   PyTorch nécessite ~2 Go. Libère de l'espace si l'installation échoue."
    CONFLICTS=$((CONFLICTS + 1))
else
    echo "O.K.: Espace disque : ${FREE_GB} Go libres"
fi

# 0f. Résumé diagnostic
echo ""
if [ "$CONFLICTS" -eq 0 ]; then
    echo "O.K.: Aucun conflit détecté, environnement propre."
else
    echo "ATTENTION: $CONFLICTS conflit(s) détecté(s) et traité(s) (voir ci-dessus)."
fi

# Interrompre si conda est actif
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo ""
    echo "NO NO! Arrêt : désactive conda manuellement avant de continuer."
    echo "   --> conda deactivate"
    exit 1
fi

echo ""
echo "============================================="
echo "   FIAC - Installation & Lancement           "
echo "============================================="

#  1. Installer uv (installeur rapide) ---------------------------------------
echo ""
echo "--> Vérification de uv (installeur rapide)..."
if ! command -v uv &>/dev/null; then
    echo "Installation de uv... Veuillez patienter..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ajouter uv au PATH pour cette session
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
echo "O.K.: uv disponible : $(uv --version)"

#  2. Trouver une version Python compatible ------------------------------------
find_python() {
    for cmd in python3.11 python3.10 python3.12 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$minor" -ge "$REQUIRED_MINOR_MIN" ] && [ "$minor" -le "$REQUIRED_MINOR_MAX" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}



#########################################################################################################################################################
##                                                     POINT PIVOTANT - UTILISATION DE WINDOWS                                                         ##
#########################################################################################################################################################



## Début commenter ici, si Windows

echo ""
echo "--> Recherche d'une version Python compatible (3.10 à 3.12)..."
PYTHON=$(find_python || true)

if [ -z "$PYTHON" ]; then
    echo "ATTENTION:  Python 3.10–3.12 non trouvé."
    # uv peut installer Python directement
    echo "Installation de Python 3.11 via uv..."
    uv python install 3.11
    PYTHON=$(uv python find 3.11)
fi

echo "O.K.: Python : $PYTHON ($($PYTHON --version))"

## Fin commenter ici pour Windows

## Début décommenter ici pour Windows

# echo ""
# echo "--> Forçage de Python 3.10 pour s'adapter aux exigences Windows..."

# # Vérifier si python3.10 existe déjà
# if command -v python3.10 &>/dev/null; then
#     PYTHON="python3.10"
# else
#     echo "Python 3.10 non trouvé → installation via uv..."
#     uv python install 3.10
#     PYTHON=$(uv python find 3.10)
# fi

# echo "O.K.: Python : $PYTHON ($($PYTHON --version))"

## Fin décommenter ici pour Windows, entrer la commande bash launch_fast.sh dans Git Bash



#########################################################################################################################################################
##                                                    FIN POINT PIVOTANT - UTILISATION DE WINDOWS                                                      ##
#########################################################################################################################################################




#  3. Créer le venv avec uv ------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "--> Création de l'environnement virtuel..."
    uv venv "$VENV_DIR" --python "$PYTHON"
    echo "O.K.: Environnement créé : $VENV_DIR/"
else
    echo "O.K.: Environnement virtuel existant détecté."
fi

source "$VENV_DIR/bin/activate"
echo "O.K.: Environnement activé : $(which python) ($(python --version))"

# 4. Installer les dépendances avec uv ----------------------------------
echo ""
echo "--> Vérification des dépendances (uv - mode rapidos!)..."

check_pkg() {
    python -c "import $1" &>/dev/null
}

install_if_missing() {
    local import_name="$1"
    shift
    local pip_pkgs=("$@")
    if ! check_pkg "$import_name" 2>/dev/null; then
        echo "Installation de ${pip_pkgs[*]}..."
        uv pip install "${pip_pkgs[@]}"
    else
        echo "  O.K.: $import_name déjà installé"
    fi
}

# setuptools en premier pour éviter les conflits
uv pip install --quiet "setuptools<82" wheel

install_if_missing "torch"           torch torchvision torchaudio
install_if_missing "tensorflow" tensorflow
#install_if_missing "facenet_pytorch" facenet-pytorch
install_if_missing "insightface"     insightface
install_if_missing "onnxruntime"     onnxruntime
install_if_missing "cv2"             opencv-python
install_if_missing "PIL"             pillow
install_if_missing "sklearn"         scikit-learn
install_if_missing "matplotlib"      matplotlib
install_if_missing "streamlit"       streamlit
install_if_missing "tqdm"            tqdm

#  5. Vérifier que le fichier app existe ---------------------------------
echo ""
if [ ! -f "$APP_FILE" ]; then
    echo "NO NO! Fichier $APP_FILE introuvable dans : $(pwd)"
    echo "   Place face_recognition_app.py dans le même dossier que launch.sh"
    exit 1
fi

#  6. Résumé final avant lancement ------------------------------------------
echo ""
echo "============================================="
echo "   O.K.: Environnement prêt"
echo "   Python  : $(python --version)"
echo "   Venv    : $(which python)"
echo "   GPU     : $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Non détecté (CPU)')" 2>/dev/null)"
echo "============================================="

#  7. Lancer Streamlit --------------------------------------------------------
echo ""
echo "   En avant! Lancement de l'application!"
echo "   Veuillez ouvrir le lien suivant: http://localhost:8501 dans votre navigateur"
echo "   Ctrl+C pour arrêter"
echo ""

streamlit run "$APP_FILE" \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false