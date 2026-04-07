---
title: FIAC

---

# FIAC: Facial recognition Independant and ACcompanied by deep learning
### Reconnaissance faciale pour la vérification d'identité, amélioré par deep learning

> Système de vérification d'identité par deep learning, comparant deux approches state-of-the-art, **ResNet50** (*vu en cours*) et **ArcFace**, avec une interface graphique intuitive développée en Streamlit.

---

## Table des matières

- [Présentation](#-présentation)
- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du système](#-architecture-du-système)
- [Modèles implémentés](#-modèles-implémentés)
- [Métriques d'évaluation](#-métriques-dévaluation)
- [Résultats](#-résultats)
- [Installation & Lancement](#-installation--lancement)
- [Structure du projet](#-structure-du-projet)
- [Guide d'utilisation](#-guide-dutilisation)
- [Dataset](#-dataset)
- [Dépendances](#-dépendances)
- [Limitations connues](#-limitations-connues)
- [Pistes d'amélioration](#-pistes-damélioration)
- [Autrice](#-autrice)
- [Licence](#-licence)
- [Précautions](#-précautions)

---

## Présentation

Ce projet de reconnaissance faciale s'inscrit dans une recherche vers la fin des mots de passe écrits sur un post-it, partagés à son voisin: vers le Multi Factor Authentication (MFA).

Le MFA est très utile en entreprise, surtout dans les domaines critiques tels que la sécurité du SI, afin que les comptes à très hauts privilèges ne soient pas compromis par un simple bruteforce de mot de passe. Cela amène aussi vers une implémentation du ZTNA (Zero Trust Network Access). Ce sont des enjeux pivotants dans l'environnement entreprise où je suis alternante.

Ce projet s'inscrivant dans cet objectif, le voici donc: c'est une application en locale, permettant ainsi de bien contrôler où vont vos données, par Steamlit.

---

## Aperçu

Ce projet implémente un pipeline complet de **vérification d'identité par reconnaissance faciale**, dans le cadre d'un cours de deep learning. L'objectif est de comparer deux algorithmes de référence sur le dataset public **LFW (Labeled Faces in the Wild)** et de rendre les résultats accessibles via une interface utilisateur moderne.

> 📸 *[Insérer ici un screenshot de l'interface principale]*

Le système répond à une question simple : **est-ce que ces deux visages appartiennent à la même personne ?**

---

## Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 📷 Vérification par image | Upload de deux photos, score de similarité instantané |
| 🎥 Webcam temps réel | Comparaison live avec une photo de référence |
| 📊 Évaluation LFW | FAR, FRR, courbes ROC, AUC sur le dataset benchmark |
| ⚙️ Multi-modèles | Bascule entre FaceNet et ArcFace en un clic |
| 🎚️ Seuil ajustable | Curseur pour calibrer la sensibilité de vérification |
| 💾 Export des résultats | Téléchargement des courbes ROC en PNG |

---

## Architecture du système

```
Image(s) d'entrée
       │
       ▼
┌─────────────────┐
│  Détection      │  ← MTCNN (Multi-task Cascaded CNN)
│  & Alignement   │     Localise et aligne le visage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extraction     │  ← FaceNet  →  embedding 512-D
│  d'embeddings   │    ArcFace  →  embedding 512-D
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Similarité     │  ← Similarité cosinus entre les deux vecteurs
│  cosinus        │     score ∈ [-1, 1]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Décision       │  ← score ≥ seuil → MATCH ✅
│                 │    score <  seuil → NO MATCH ❌
└─────────────────┘
```

---

## Modèles implémentés

### 1. FaceNet — `InceptionResnetV1`

FaceNet est un modèle développé par Google qui apprend à projeter les visages dans un espace métrique de 512 dimensions, où la distance entre deux embeddings reflète la similarité entre les identités.

- **Architecture** : InceptionResnetV1
- **Pré-entraînement** : VGGFace2 (3,31 millions d'images, 9 131 identités)
- **Dimension d'embedding** : 512
- **Détection faciale** : MTCNN (Multi-task Cascaded CNN)
- **Fonction de perte** : Triplet Loss
- **Librairie** : `facenet-pytorch`

> 📸 *[Insérer ici un schéma de l'architecture InceptionResnet ou un exemple d'embedding]*

**Principe de la Triplet Loss :**
Le modèle est entraîné avec des triplets (ancre, positif, négatif) pour que :
```
distance(ancre, positif) + marge < distance(ancre, négatif)
```

---

### 2. ArcFace — `buffalo_sc`

ArcFace est un modèle développé par InsightFace qui introduit une marge angulaire additive dans l'espace des embeddings, forçant le modèle à créer des représentations plus discriminantes.

- **Architecture** : ResNet-based backbone
- **Pré-entraînement** : MS1MV3 (5,8 millions d'images, 93 431 identités)
- **Dimension d'embedding** : 512
- **Détection faciale** : RetinaFace (intégrée)
- **Fonction de perte** : ArcFace Loss (marge angulaire additive)
- **Librairie** : `insightface`

> 📸 *[Insérer ici un schéma illustrant la marge angulaire ArcFace]*

**Principe de l'ArcFace Loss :**
```
L = -log( e^(s·cos(θ_yi + m)) / (e^(s·cos(θ_yi + m)) + Σ e^(s·cos(θ_j))) )
```
où `m` est la marge angulaire additive et `s` le facteur d'échelle.

---

### Comparaison des deux approches

| Critère | FaceNet | ArcFace |
|---|---|---|
| Architecture | InceptionResnetV1 | ResNet backbone |
| Dataset d'entraînement | VGGFace2 | MS1MV3 |
| Fonction de perte | Triplet Loss | ArcFace Loss |
| Embeddings | 512-D | 512-D |
| Vitesse CPU | ~200ms/image | ~150ms/image |
| Robustesse | Bonne | Très bonne |

---

## Métriques d'évaluation

### FAR — False Acceptance Rate (Taux de fausse acceptation)
Proportion de paires **différentes** incorrectement acceptées comme identiques.
```
FAR = Faux Positifs / Total des paires négatives
```
Un FAR élevé = le système est trop permissif → risque de sécurité.

### FRR — False Rejection Rate (Taux de faux rejet)
Proportion de paires **identiques** incorrectement rejetées.
```
FRR = Faux Négatifs / Total des paires positives
```
Un FRR élevé = le système est trop strict → mauvaise expérience utilisateur.

### EER — Equal Error Rate
Point d'opération où FAR = FRR. Sert à comparer objectivement deux systèmes indépendamment du seuil choisi.

### Courbe ROC & AUC
La courbe ROC trace le TAR (1−FRR) en fonction du FAR pour tous les seuils possibles. L'AUC (aire sous la courbe) résume la performance globale :
- AUC = 1.0 → système parfait
- AUC = 0.5 → système aléatoire

> 📸 *[Insérer ici les courbes ROC générées par l'application]*

---

## Résultats

Évaluation réalisée sur le sous-ensemble **test** de LFW (1 000 paires).

| Modèle | AUC | EER | FAR (@ EER) | FRR (@ EER) |
|---|---|---|---|---|
| FaceNet | ~0.97 | ~0.05 | ~5% | ~5% |
| ArcFace | ~0.98 | ~0.04 | ~4% | ~4% |

> Attention, Ces valeurs sont indicatives — les résultats exacts dépendent du nombre de paires évaluées et du seuil calibré.

> 📸 *[Insérer ici le tableau de résultats généré par l'onglet Évaluation]*

**Analyse :** ArcFace surpasse légèrement FaceNet sur LFW, grâce à sa marge angulaire qui force des représentations plus séparables. FaceNet reste très compétitif et plus rapide à l'inférence sur CPU.

---

## Installation & Lancement

### Prérequis

- Python **3.10, 3.11 ou 3.12** (⚠️ Python 3.13 non supporté)
- WSL, Linux ou macOS
- 8 Go de RAM minimum
- Connexion internet (téléchargement des poids des modèles au premier lancement)

### Lancement automatique (recommandé)

Le script `launch.sh` gère tout automatiquement :

```bash
# 1. Cloner ou télécharger le projet
git clone https://github.com/votre-username/face-id-systeme.git
cd face-id-systeme

# 2. Lancer le script
bash launch.sh
```

Le script va automatiquement :
1. Détecter ou installer une version Python compatible (via pyenv si nécessaire)
2. Créer un environnement virtuel isolé
3. Installer toutes les dépendances
4. Lancer l'application sur `http://localhost:8501`

> 📸 *[Insérer ici un screenshot du terminal pendant le lancement]*

### Lancement manuel

```bash
# Créer et activer l'environnement virtuel
python3.11 -m venv faceid_env
source faceid_env/bin/activate   # Linux/Mac/WSL
# ou : faceid_env\Scripts\activate  (Windows natif)

# Installer les dépendances
pip install --upgrade pip "setuptools<82" wheel
pip install torch torchvision torchaudio
pip install facenet-pytorch insightface onnxruntime \
            opencv-python pillow scikit-learn \
            matplotlib streamlit tqdm

# Lancer l'application
streamlit run face_recognition_app.py
```

Ouvrir ensuite **http://localhost:8501** dans le navigateur.

---

## Structure du projet

```
face-id-systeme/
│
├── face_recognition_app.py   ← Application complète (code unique)
├── launch.sh                 ← Script d'installation & lancement
├── README.md                 ← Ce fichier
│
├── data/                     ← Dataset LFW (généré automatiquement)
│   ├── raw/
│   ├── processed/
│   └── pairs/
│
└── evaluation/               ← Résultats d'évaluation
    ├── plots/                ← Courbes ROC exportées
    └── results/              ← Métriques JSON par modèle
```

> Le fichier `face_recognition_app.py` est **auto-suffisant** : il contient toutes les étapes du pipeline (préparation des données, modèles, évaluation, interface).

---

## Guide d'utilisation

### Onglet 1 — Vérification par image

1. Sélectionner un modèle dans la barre latérale (FaceNet ou ArcFace)
2. Ajuster le seuil de décision si nécessaire (défaut : 0.75)
3. Uploader une **image de référence** (photo connue de la personne)
4. Uploader une **image à vérifier**
5. Cliquer sur **Vérifier l'identité**
6. Lire le résultat : score de similarité + verdict MATCH / NO MATCH

> 📸 *[Insérer ici un screenshot de l'onglet vérification avec un résultat]*

---

### Onglet 2 — Webcam temps réel

1. Uploader une **image de référence**
2. Cliquer sur **Démarrer la webcam**
3. Le système compare en continu le flux webcam à la référence
4. Cliquer sur **Arrêter** pour terminer

> 📸 *[Insérer ici un screenshot de l'onglet webcam en fonctionnement]*

---

### Onglet 3 — Évaluation LFW

1. Choisir le nombre de paires à évaluer (50 à 500)
2. Cliquer sur **Lancer l'évaluation**
3. Patienter pendant le calcul des embeddings
4. Consulter le tableau comparatif FAR / FRR / AUC
5. Visualiser et télécharger les courbes ROC

> 📸 *[Insérer ici un screenshot des courbes ROC et du tableau comparatif]*

---

### Barre latérale — paramètres

| Paramètre | Description |
|---|---|
| **Modèle** | Basculer entre FaceNet et ArcFace |
| **Seuil** | Similarité cosinus minimale pour valider (0.0–1.0) |
| **Dispositif** | CPU ou GPU détecté automatiquement |

---

## Dataset

**LFW — Labeled Faces in the Wild**

- **Source** : University of Massachusetts
- **Contenu** : 13 233 images de 5 749 célébrités
- **Paires test** : 6 000 paires (3 000 positives + 3 000 négatives)
- **Téléchargement** : automatique via `scikit-learn` au premier lancement
- **Licence** : libre pour la recherche académique

Le dataset est téléchargé automatiquement dans `data/` au lancement de l'évaluation.

---

## Dépendances

| Librairie | Version testée | Rôle |
|---|---|---|
| `torch` | 2.x | Framework deep learning |
| `torchvision` | 0.x | Transforms & utilitaires vision |
| `facenet-pytorch` | 2.5+ | Modèle FaceNet + MTCNN |
| `insightface` | 0.7+ | Modèle ArcFace |
| `onnxruntime` | 1.x | Runtime requis par insightface |
| `opencv-python` | 4.x | Traitement d'images & webcam |
| `pillow` | 10.x | Manipulation d'images |
| `scikit-learn` | 1.x | Dataset LFW + métriques |
| `matplotlib` | 3.x | Génération des courbes ROC |
| `streamlit` | 1.x | Interface graphique web |
| `tqdm` | 4.x | Barres de progression |

---

## Limitations connues

- **Python 3.13 non supporté** : `facenet-pytorch` nécessite Python ≤ 3.12
- **Webcam dans WSL** : l'accès à la webcam peut être limité selon la configuration WSL
- **Premier lancement lent** : téléchargement des poids (~500 Mo) à la première exécution
- **Performance CPU** : l'évaluation de 500 paires peut prendre 10–15 minutes sans GPU
- **ArcFace sur petits visages** : les performances se dégradent si le visage est trop petit dans l'image

---

## Pistes d'amélioration

- [ ] Ajouter un troisième modèle (VGGFace2, DeepFace) pour augmenter d'autant plus la comparaison et d'avoir du choix
- [ ] Implémenter la détection de liveness (anti-spoofing) afin de détecter que ce soit bien la vraie personne et pas une reconstitution
- [ ] Support GPU natif pour l'inférence ArcFace
- [ ] Déploiement cloud (Streamlit Cloud ou Hugging Face Spaces) pour une accessibilité aisée
- [ ] Base de données d'identités pour la vérification 1-à-N (identification), car plus d'entraînement amène un meilleur modèle
- [ ] Calibration automatique du seuil optimal par validation croisée, pour les images et/ou webcam de moindre qualité
- [ ] Export des résultats en PDF, pour une meilleure visibilité et versatilité
- [ ] Meilleur moyen de créer les venv Python, car les téléchargement des dépendances sont longues, même avec uv.

---

## Autrice

Andréa TOMAS, déléguée et étudiante de M1 CSSD (Cybersécurité et Science des Données), à l'Université Paris 8.
Projet réalisé dans le cadre d'un cours de deep learning appliqué à la reconnaissance faciale.

---

## Licence

Ce projet est à usage académique. Les modèles pré-entraînés sont soumis aux licences de leurs auteurs respectifs (VGGFace2, InsightFace).

---

## Précautions

Ce projet a été assisté par IA, notamment pour la rédaction du launch.sh et launch_fast.sh pour une installation rapide, claire, et pour comprendre les bases et erreurs récurrentes de steamlit.