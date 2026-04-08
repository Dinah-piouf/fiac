"""
=============================================================================================
RECONNAISSANCE FACIALE POUR LA VÉRIFICATION D'IDENTITÉ - PROJET FIAC
============================================================================================
Voici toutes les étapes intégrées du projet :
  1. Configuration et vérification de l'environnement
  2. Préparation des données LFW
  3. Modèle ResNet50 (TensorFlow/Keras), extracteur de features
  4. Modèle ArcFace (insightface)
  5. Évaluation : FAR, FRR, ROC, AUC
  6. Interface Streamlit (permettant notamment l'upload d'image de ref + webcam temps réel)

Lancement :
  bash launch_fast.sh
===============================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS - importants
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import warnings
import time
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, auc
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TensorFlow logs

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES & CHEMINS
# ─────────────────────────────────────────────────────────────────────────────

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR     = Path("data")
RESULTS_DIR  = Path("evaluation/results")
PLOTS_DIR    = Path("evaluation/plots")
IMG_SIZE_RN  = 224   # taille entrée ResNet50
IMG_SIZE_AF  = 160   # taille entrée ArcFace
THRESHOLD_RN = 0.80  # seuil cosinus ResNet50  (à calibrer par l'utilisateur)
THRESHOLD_AF = 0.30  # seuil cosinus ArcFace   (à calibrer par l'utilisateur)

for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  1 — VÉRIFICATION ENVIRONNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def check_environment() -> dict: #checker l'environnement au cas où
    results = {}
    libs = {
        "tensorflow": lambda: __import__("tensorflow").__version__,
        "cv2":        lambda: cv2.__version__,
        "insightface":lambda: __import__("insightface").__version__,
        "sklearn":    lambda: __import__("sklearn").__version__,
        "streamlit":  lambda: st.__version__,
    }
    for lib, get_ver in libs.items():
        try:
            results[lib] = {"ok": True, "version": get_ver()}
        except Exception as e:
            results[lib] = {"ok": False, "error": str(e)}

    results["cuda"] = {
        "ok": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0)
                  if torch.cuda.is_available() else "CPU"
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  2 — PRÉPARATION DES DONNÉES LFW
# ─────────────────────────────────────────────────────────────────────────────
def load_lfw_pairs(subset: str = "test") -> tuple:
    """
    Fonction générant des paires équilibrées directement depuis les images LFW, sans utiliser les .txt.
    - Paires positives  (label=1) : 2 images de la même personne
    - Paires négatives  (label=0) : 2 images de personnes différentes
    Les images sont contenues dans data/lfw_home/lfw_funneled/
    """
    import random

    lfw_root = DATA_DIR / "lfw_home" / "lfw_funneled"
    if not lfw_root.exists():
        st.error(f"Dossier LFW introuvable : {lfw_root}")
        return np.array([]), np.array([])

    # Lister les personnes avec >=2 images (positives) et toutes (négatives)
    people_multi = {}
    people_all   = {}
    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        if len(imgs) >= 2:
            people_multi[person_dir.name] = imgs
        if imgs:
            people_all[person_dir.name] = imgs

    multi_names = list(people_multi.keys())
    all_names   = list(people_all.keys())
    random.seed(42)

    pairs_imgs, labels = [], []

    # Paires positives
    for name in random.sample(multi_names, min(250, len(multi_names))):
        imgs = people_multi[name]
        a, b = random.sample(imgs, 2)
        pairs_imgs.append((str(a), str(b)))
        labels.append(1)

    # Paires négatives (même nombre)
    for _ in range(len(pairs_imgs)):
        n1, n2 = random.sample(all_names, 2)
        a = random.choice(people_all[n1])
        b = random.choice(people_all[n2])
        pairs_imgs.append((str(a), str(b)))
        labels.append(0)

    # Mélange
    combined = list(zip(pairs_imgs, labels))
    random.shuffle(combined)
    pairs_imgs, labels = zip(*combined)

    # Chargement des images en numpy float [0,1]
    def load_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    st.info(f"Chargement de {len(labels)} paires pour l'évaluation...")
    pairs_np  = np.array([[load_img(a), load_img(b)] for a, b in pairs_imgs])
    labels_np = np.array(labels)

    #Comptage des paires
    pos = int(labels_np.sum())
    neg = int((1 - labels_np).sum())
    st.success(f"O.K.: {len(labels_np)} paires — {pos} positives, {neg} négatives")
    return pairs_np, labels_np


def preprocess_for_resnet(img_array: np.ndarray) -> np.ndarray:
    """
    img_array : float [0,1], shape (H, W, 3), issu de LFW
    float32 224×224×3 prétraité pour ResNet50, pour que ça soit plus facile
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = (img_array * 255).astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE_RN, IMG_SIZE_RN))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img


def preprocess_for_arcface(img_array: np.ndarray) -> np.ndarray:
    """
    img_array : float [0,1], shape (H, W, 3)
    uint8 BGR 160×160, pour que ça soit plus facile aussi
    """
    img = (img_array * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE_AF, IMG_SIZE_AF))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 3a — MODÈLE RESNET50 (TensorFlow / Keras)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement ResNet50…")
def load_resnet50():
    """
    Charge ResNet50 pré-entraîné sur ImageNet.
    Ajoute un GlobalAveragePooling2D pour obtenir un embedding 2048-D.
    J'ai repris les grands pas de ce que l'on avait fait en cours.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras import Model

    #Appel au modèle ResNet50
    base = ResNet50(include_top=False, weights="imagenet",
                    input_shape=(IMG_SIZE_RN, IMG_SIZE_RN, 3))
    output = base.output
    output = GlobalAveragePooling2D()(output)
    extractor = Model(inputs=base.input, outputs=output)
    return extractor


def get_resnet_embedding(img_bgr: np.ndarray, extractor) -> np.ndarray:
    """
    img_bgr : uint8 BGR (capture webcam ou upload)
    embedding 2048-D normalisé (L2)
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img     = cv2.resize(img_rgb, (IMG_SIZE_RN, IMG_SIZE_RN))
    img     = img.astype(np.float32)
    img     = preprocess_input(img)              # normalisation via ImageNet
    img     = np.expand_dims(img, axis=0)        # taille attendue (1, 224, 224, 3)

    #Prédiction du modèle
    emb = extractor.predict(img, verbose=0)      # taille attendue (1, 2048)
    emb = emb[0]
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm                         # normalisation L2!
    return emb

#Calcul de la similarité
def resnet_similarity(img1: np.ndarray, img2: np.ndarray,
                       extractor) -> float:
    """Similarité cosinus entre deux images BGR."""
    e1 = get_resnet_embedding(img1, extractor)
    e2 = get_resnet_embedding(img2, extractor)
    return float(np.dot(e1, e2))                 # cosinus, donc des vecteurs normalisés


# ─────────────────────────────────────────────────────────────────────────────
#  3b — MODÈLE ARCFACE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement ArcFace…")
def load_arcface():
    import insightface
    from insightface.app import FaceAnalysis
    #Va load ArcFace
    app = FaceAnalysis(name="buffalo_sc",
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(IMG_SIZE_AF, IMG_SIZE_AF))
    return app


def get_arcface_embedding(img_bgr: np.ndarray, app) -> np.ndarray | None:
    #notamment là où joue ArcFace Loss
    faces = app.get(img_bgr)
    if not faces:
        return None
    return faces[0].normed_embedding


def arcface_similarity(img1: np.ndarray, img2: np.ndarray, app) -> float:
    e1 = get_arcface_embedding(img1, app)
    e2 = get_arcface_embedding(img2, app)
    if e1 is None or e2 is None:
        return 0.0
    return float(np.dot(e1, e2))


# ─────────────────────────────────────────────────────────────────────────────
#  4 — ÉVALUATION : FAR / FRR / ROC / AUC
# ─────────────────────────────────────────────────────────────────────────────
def compute_far_frr(scores: np.ndarray, labels: np.ndarray,
                    threshold: float) -> tuple[float, float]:
    preds    = (scores >= threshold).astype(int)
    neg_mask = labels == 0
    pos_mask = labels == 1
    #fait les calculs nécessaires pour FAR et FRR
    far = np.sum((preds == 1) & neg_mask) / max(np.sum(neg_mask), 1)
    frr = np.sum((preds == 0) & pos_mask) / max(np.sum(pos_mask), 1)
    return float(far), float(frr)


def evaluate_model(pairs: np.ndarray, labels: np.ndarray,
                   model_fn, model_name: str,
                   preprocess_fn,
                   max_pairs: int = 300) -> dict:
    """
    Permet d'évaluer un modèle sur nb=max_pairs de paires LFW.
    preprocess_fn, c'est donc la fonction de prétraitement adaptée au modèle.
    """
    n      = min(len(labels), max_pairs)
    scores = []
    progress = st.progress(0, text=f"Évaluation {model_name}…")

    for i in range(n):
        img1 = preprocess_fn(pairs[i, 0])
        img2 = preprocess_fn(pairs[i, 1])
        s    = model_fn(img1, img2)
        scores.append(s)
        if i % 20 == 0:
            progress.progress((i + 1) / n,
                               text=f"Évaluation {model_name} — {i+1}/{n}")
    progress.empty()

    scores = np.array(scores)
    labs   = labels[:n]

    fpr, tpr, thresholds = roc_curve(labs, scores)
    roc_auc              = auc(fpr, tpr)

    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_thr = thresholds[eer_idx]
    far, frr = compute_far_frr(scores, labs, eer_thr)

    result = {
        "model":     model_name,
        "auc":       round(roc_auc, 4),
        "eer":       round((far + frr) / 2, 4),
        "far":       round(far, 4),
        "frr":       round(frr, 4),
        "threshold": round(float(eer_thr), 4),
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
        "scores":    scores.tolist(),
        "labels":    labs.tolist(),
    }

    out_file = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(result, f)

    return result

#Va permettre de montrer les courbes ROC
def plot_roc_curves(results: list[dict]) -> plt.Figure:
    colors = ["#00C9FF", "#FF6B6B"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    for res, color in zip(results, colors):
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
                label=f"{res['model']} (AUC = {res['auc']:.3f})")
    ax.set_xlabel("Taux de fausse acceptation (FAR)", color="white")
    ax.set_ylabel("Taux de vraie acceptation (TAR = 1 − FRR)", color="white")
    ax.set_title("Courbes ROC — Comparaison des modèles", color="white")
    ax.legend(loc="lower right", facecolor="#1A1D27", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def plot_far_frr(results: list[dict]) -> plt.Figure:
    models = [r["model"] for r in results]
    far_v  = [r["far"]   for r in results]
    frr_v  = [r["frr"]   for r in results]
    x      = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    w = 0.3
    ax.bar(x - w/2, far_v, w, label="FAR", color="#FF6B6B")
    ax.bar(x + w/2, frr_v, w, label="FRR", color="#00C9FF")
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white")
    ax.set_ylabel("Taux d'erreur", color="white")
    ax.set_title("FAR & FRR au seuil EER", color="white")
    ax.legend(facecolor="#1A1D27", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  5 — VÉRIFICATION D'IDENTITÉ (logique métier)
# ─────────────────────────────────────────────────────────────────────────────
def verify_identity(img1_bgr: np.ndarray, img2_bgr: np.ndarray,
                    model_choice: str, threshold: float,
                    resnet=None, arcface=None) -> dict:
    t0 = time.time()
    if model_choice == "ResNet50":
        score = resnet_similarity(img1_bgr, img2_bgr, resnet)
    else:
        score = arcface_similarity(img1_bgr, img2_bgr, arcface)
    elapsed = (time.time() - t0) * 1000
    return {
        "match":   score >= threshold,
        "score":   round(score, 4),
        "model":   model_choice,
        "time_ms": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6 — INTERFACE STREAMLIT FINAL
# ─────────────────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_result_overlay(img_bgr: np.ndarray, match: bool,
                         score: float) -> np.ndarray:
    color = (0, 220, 0) if match else (0, 0, 220)
    label = f"{'It is a big match!! :D' if match else 'No no match... :c'}  {score:.3f}"
    h, w  = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, h - 38), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.65, img_bgr, 0.35, 0, img_bgr)
    cv2.putText(img_bgr, label, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img_bgr


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; background: #080A10; color: #D0D6E8; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; letter-spacing: .08em; color: #7A859E; }
.stTabs [aria-selected="true"] { color: #00C9FF !important; border-bottom: 2px solid #00C9FF; }
.result-card { border-radius: 12px; padding: 1.2rem 1.5rem; margin: .8rem 0; font-family: 'DM Mono', monospace; font-size: .9rem; letter-spacing: .03em; }
.match-ok   { background: rgba(0,180,80,.12);  border: 1px solid #00B450; }
.match-fail { background: rgba(220,50,50,.12); border: 1px solid #DC3232; }
.metric-box { background: #10131F; border: 1px solid #1E2435; border-radius: 10px; padding: .9rem 1rem; text-align: center; }
.metric-label { font-size: .7rem; color: #7A859E; margin-bottom: .3rem; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #00C9FF; }
div[data-testid="stSidebar"] { background: #0B0E18 !important; }
.stButton > button { background: linear-gradient(135deg, #00C9FF, #0077FF); color: white; border: none; border-radius: 8px; font-family: 'DM Mono', monospace; font-weight: 500; letter-spacing: .05em; padding: .5rem 1.3rem; transition: opacity .2s; }
.stButton > button:hover { opacity: .85; }
</style>
"""


def render_metric(label: str, value: str, col):
    col.markdown(
        f'<div class="metric-box">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="FIAC System", page_icon="🧩",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # Sidebar avec le choix des modèles + seuil de détection
    with st.sidebar:
        st.markdown("## FIAC System")
        st.markdown("---")
        model_choice = st.radio(
            "Modèle",
            ["ResNet50", "ArcFace"],
            help="ResNet50 = backbone ImageNet + GlobalAvgPool (TensorFlow)\n"
                 "ArcFace  = insightface buffalo_sc"
        )
        threshold = st.slider(
            "Seuil de décision", 0.0, 1.0,
            THRESHOLD_RN if model_choice == "ResNet50" else THRESHOLD_AF,
            0.01,
            help="Similarité cosinus minimale pour valider l'identité"
        )
        st.markdown("---")
        st.markdown(f"**Dispositif :** `{DEVICE.upper()}`")
        st.markdown("---")
        env_check = st.button("Vérifier l'environnement")

    #  Chargement des modèles 
    resnet_model, arcface_app = None, None
    try:
        resnet_model = load_resnet50()
    except Exception as e:
        st.sidebar.warning(f"ResNet50 non disponible : {e}")
    try:
        arcface_app = load_arcface()
    except Exception as e:
        st.sidebar.warning(f"ArcFace non disponible : {e}")

    # Les différentes Tabs pour la vérification d'image, la webcam, etc
    tab_verify, tab_webcam, tab_eval, tab_env = st.tabs([
        "Vérification image",
        "Webcam temps réel",
        "Évaluation LFW",
        "Environnement",
    ])

    # ── TAB 1 : VÉRIFICATION PAR IMAGE ───────────────────────────────────────
    with tab_verify:
        st.markdown("### Vérification par images")
        st.markdown("Chargez une **image de référence** et une **image à vérifier**.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Image de référence**")
            ref_file = st.file_uploader("Référence", type=["jpg","jpeg","png"],
                                         key="ref", label_visibility="collapsed")
            if ref_file:
                st.image(Image.open(ref_file), use_column_width=True)
        with col2:
            st.markdown("**Image à vérifier**")
            probe_file = st.file_uploader("Probe", type=["jpg","jpeg","png"],
                                           key="probe", label_visibility="collapsed")
            if probe_file:
                st.image(Image.open(probe_file), use_column_width=True)

        if ref_file and probe_file:
            if st.button("Vérifier l'identité"):
                ref_bgr   = pil_to_bgr(Image.open(ref_file))
                probe_bgr = pil_to_bgr(Image.open(probe_file))
                with st.spinner("Calcul des embeddings…"):
                    res = verify_identity(ref_bgr, probe_bgr, model_choice,
                                          threshold, resnet=resnet_model,
                                          arcface=arcface_app)
                status = "match-ok" if res["match"] else "match-fail"
                icon   = "O.K.: IDENTITÉ CONFIRMÉE" if res["match"] else "NO NO: IDENTITÉ REJETÉE"
                st.markdown(
                    f'<div class="result-card {status}"><strong>{icon}</strong><br>'
                    f'Score : <strong>{res["score"]}</strong> '
                    f'(seuil : {threshold}) — {res["time_ms"]} ms</div>',
                    unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                render_metric("Modèle",     res["model"],           c1)
                render_metric("Similarité", str(res["score"]),      c2)
                render_metric("Temps",      f'{res["time_ms"]} ms', c3)

    # ── TAB 2 : WEBCAM ────────────────────────────────────────────────────────
    with tab_webcam:
        st.markdown("### Vérification en temps réel (webcam)")
        ref_webcam = st.file_uploader("Image de référence",
                                       type=["jpg","jpeg","png"], key="ref_webcam")
        ref_bgr_wc = None
        if ref_webcam:
            ref_bgr_wc = pil_to_bgr(Image.open(ref_webcam))
            st.image(cv2.cvtColor(ref_bgr_wc, cv2.COLOR_BGR2RGB),
                     caption="Référence", width=200)

        start_cam = st.button("Démarrer la webcam!")
        stop_cam  = st.button("Arrêter la caméra")
        frame_ph  = st.empty()
        result_ph = st.empty()

        if start_cam and ref_bgr_wc is not None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Impossible d'accéder à la webcam.")
            else:
                st.session_state["cam_running"] = True

        if st.session_state.get("cam_running", False) and ref_bgr_wc is not None:
            cap = cv2.VideoCapture(0)
            try:
                while not stop_cam:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    res = verify_identity(ref_bgr_wc, frame, model_choice,
                                          threshold, resnet=resnet_model,
                                          arcface=arcface_app)
                    annotated = draw_result_overlay(frame.copy(),
                                                    res["match"], res["score"])
                    frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   channels="RGB", use_column_width=True)
                    status = "match-ok" if res["match"] else "match-fail"
                    icon   = "O.K.: CONFIRMÉE" if res["match"] else "NO NO: REJETÉE"
                    result_ph.markdown(
                        f'<div class="result-card {status}" style="padding:.6rem 1rem">'
                        f'{icon} — score : <strong>{res["score"]}</strong></div>',
                        unsafe_allow_html=True)
                    if stop_cam:
                        break
            finally:
                cap.release()
                st.session_state["cam_running"] = False
        elif start_cam and ref_bgr_wc is None:
            st.warning("Veuillez d'abord charger une image de référence.")

    # ── TAB 3 : ÉVALUATION LFW ───────────────────────────────────────────────
    with tab_eval:
        st.markdown("### Évaluation sur LFW")
        n_pairs  = st.slider("Nombre de paires à évaluer", 50, 500, 150, 50)
        run_eval = st.button("Lancer l'évaluation")

        if run_eval:
            pairs, labels = load_lfw_pairs("test") #va donc lire les paires pour permettre les futurs calculs de reconnaissance
            results = []

            if resnet_model is not None:
                res_rn = evaluate_model(
                    pairs, labels,
                    lambda i1, i2: resnet_similarity(i1, i2, resnet_model),
                    "ResNet50",
                    preprocess_fn=preprocess_for_resnet,
                    max_pairs=n_pairs
                )
                results.append(res_rn)

            if arcface_app is not None:
                res_af = evaluate_model(
                    pairs, labels,
                    lambda i1, i2: arcface_similarity(i1, i2, arcface_app),
                    "ArcFace",
                    preprocess_fn=preprocess_for_arcface,
                    max_pairs=n_pairs
                )
                results.append(res_af)

            if not results:
                st.error("Aucun modèle disponible pour l'évaluation.")
            else:
                st.markdown("#### Résultats")
                rows = [[r["model"], r["auc"], r["eer"],
                          r["far"], r["frr"], r["threshold"]] for r in results]
                st.table({"Modèle":  [r[0] for r in rows],
                           "AUC":     [r[1] for r in rows],
                           "EER":     [r[2] for r in rows],
                           "FAR":     [r[3] for r in rows],
                           "FRR":     [r[4] for r in rows],
                           "Seuil":   [r[5] for r in rows]})

                col_roc, col_bar = st.columns(2)
                with col_roc:
                    st.markdown("##### Courbes ROC")
                    fig_roc = plot_roc_curves(results)
                    st.pyplot(fig_roc, use_container_width=True)
                    buf = io.BytesIO()
                    fig_roc.savefig(buf, format="png", dpi=150,
                                    facecolor="#0E1117")
                    st.download_button("Télécharger ROC",
                                       buf.getvalue(), "roc_curves.png",
                                       "image/png")
                with col_bar:
                    st.markdown("##### FAR / FRR")
                    fig_bar = plot_far_frr(results)
                    st.pyplot(fig_bar, use_container_width=True)

                st.success(f"O.K.: Évaluation terminée. Résultats dans `{RESULTS_DIR}/`.")

    # ── TAB 4 : ENVIRONNEMENT ────────────────────────────────────────────────
    with tab_env:
        st.markdown("### Vérification de l'environnement")
        env  = check_environment()
        cols = st.columns(3)
        for i, (lib, info) in enumerate(env.items()):
            col = cols[i % 3]
            if lib == "cuda":
                col.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-label">CUDA / GPU</div>'
                    f'<div class="metric-value">{"O.K." if info["ok"] else "NO NO"}</div>'
                    f'<div style="font-size:.75rem;color:#7A859E">{info["device"]}</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                ok  = info.get("ok", False)
                ver = info.get("version", info.get("error", "?"))
                col.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-label">{lib}</div>'
                    f'<div class="metric-value">{"O.K." if ok else "NO NO"}</div>'
                    f'<div style="font-size:.75rem;color:#7A859E">{ver}</div>'
                    f'</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Lancement rapide")
        st.code("bash launch_fast.sh", language="bash")
        st.markdown("---")
        st.markdown("#### Lancement classique")
        st.code("bash launch.sh", language="bash")


# ───────────────────────────────────────────────────────────────────────────── le main quoi
if __name__ == "__main__":
    main()