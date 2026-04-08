---
title: FIAC

---

# FIAC: Facial recognition Independant and ACcompanied by deep learning
### Reconnaissance faciale pour la vérification d'identité, améliorée par deep learning

> Système de vérification d'identité par deep learning, comparant deux approches, **ResNet50** (*vu en cours*) et **ArcFace**, avec une interface graphique intuitive développée en Streamlit.

---

## Table des matières

- [Présentation](#-présentation)
- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du système](#-architecture-du-système)
- [Modèles implémentés](#-modèles-implémentés)
- [Métriques d'évaluation](#-métriques-dévaluation)
- [Installation & Lancement](#-installation--lancement)
- [Structure du projet](#-structure-du-projet)
- [Guide d'utilisation](#-guide-dutilisation)
- [Dataset](#-dataset)
- [Dépendances](#-dépendances)
- [Autrice](#-autrice)
- [Licence](#-licence)
- [Précautions](#-précautions)

---

## Présentation

Ce projet de reconnaissance faciale s'inscrit dans une recherche vers la fin des mots de passe écrits sur un post-it, partagés à son voisin: vers le Multi Factor Authentication (MFA).

Le MFA est très utile en entreprise, surtout dans les domaines critiques tels que la sécurité du SI, afin que les comptes à très hauts privilèges ne soient pas compromis par un simple bruteforce de mot de passe. Cela amène aussi vers une implémentation du ZTNA (Zero Trust Network Access). Ce sont des enjeux pivotants dans l'environnement entreprise où je suis alternante.

Ce projet s'inscrivant dans cet objectif, le voici donc: c'est une application en locale, permettant ainsi de bien contrôler où vont vos données, par Steamlit.

Nous parlons donc ici de reconnaisance faciale alimentée par deep learning, en effet, nous utilisons deux modèles de transfer learning augmentés par deep learning. Nous allons tout du long faire un comparatif de ces modèles.

---

## Aperçu

Ce projet implémente une pipeline complète de **vérification d'identité par reconnaissance faciale**, dans le cadre de l'UE "Programmation carte à puce". L'objectif est de comparer au moins deux algorithmes de référence sur le dataset public choisi **LFW (Labeled Faces in the Wild)** et de rendre les résultats accessibles via une interface utilisateur moderne et accessible.

https://github.com/user-attachments/assets/cc3fc8ac-b400-4677-acdc-218a37597e2c

>Page principale de l'application

Le système répond à une question simple : **est-ce que ces deux visages appartiennent à la même personne ?** Le système le fait alors par deux moyens: deux photos, ou une photo de référence et la webcam.

---

## Fonctionnalités

| Fonctionnalité | Description |
|---|---|
|Vérification par image | Upload de deux photos, score de similarité instantané |
|Webcam temps réel | Comparaison live avec une photo de référence |
|Évaluation LFW | FAR, FRR, courbes ROC, AUC sur le dataset benchmark |
|Multi-modèles | Bascule entre ResNet50 et ArcFace en un clic |
|Seuil ajustable | Curseur pour calibrer la sensibilité de vérification |
|Export des résultats | Téléchargement des courbes ROC en PNG |

---

## Architecture du système

```
Image(s) d'entrée
       │
       ▼
┌─────────────────┐
│  Prétraitement  │  ← Redimensionnement + normalisation ImageNet
│  & Resize       │     224×224 (ResNet50) / 160×160 (ArcFace)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extraction     │  ← ResNet50  →  embedding 2048-D
│  d'embeddings   │    ArcFace   →  embedding 512-D
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
│  Décision       │  ← score ≥ seuil -->  YES YES! MATCH 
│                 │    score <  seuil --> NO NO! NO MATCH
└─────────────────┘
```

---

## Modèles implémentés

### 1. ResNet50 — `GlobalAveragePooling2D`

ResNet50 est un réseau de neurones convolutif profond développé par Microsoft, pré-entraîné sur ImageNet. Utilisé ici comme extracteur de features : on retire la tête de classification et on ajoute un `GlobalAveragePooling2D` pour obtenir un vecteur d'embedding dense, exactement comme défini dans le notebook de référence.

- **Architecture** : ResNet50 (50 couches, blocs résiduels)
- **Pré-entraînement** : ImageNet (1,2 million d'images, 1 000 classes)
- **Dimension d'embedding** : 2048-D
- **Détection faciale** : aucune — traitement direct de l'image
- **Normalisation** : `preprocess_input` Keras (normalisation ImageNet)
- **Librairie** : `tensorflow` / `keras`


https://github.com/user-attachments/assets/493691f0-839c-4a85-be9e-fc26e7d48028

> Schématisation du fonctionnement de ResNet50, Source https://www.ultralytics.com/fr/blog/what-is-resnet-50-and-what-is-its-relevance-in-computer-vision

**Spécificité :** Principe des blocs résiduels
ResNet introduit des connexions de court-circuit (*skip connections*) qui permettent au gradient de se propager sans dégradation à travers les couches profondes.

---

### 2. ArcFace — `buffalo_sc`

ArcFace est un modèle développé par InsightFace qui introduit une marge angulaire additive dans l'espace des embeddings, forçant le modèle à créer des représentations plus discriminantes.

- **Architecture** : ResNet-based backbone
- **Pré-entraînement** : MS1MV3 (5,8 millions d'images, 93 431 identités)
- **Dimension d'embedding** : 512-D
- **Détection faciale** : RetinaFace (intégrée)
- **Fonction de perte** : ArcFace Loss (marge angulaire additive)
- **Librairie** : `insightface`
  
https://github.com/user-attachments/assets/f1e215a1-38a6-4017-8b1a-651a36aa6f0c
> Schématisation du fonctionnement de ArcFace, Source https://www.ultralytics.com/fr/blog/what-is-resnet-50-and-what-is-its-relevance-in-computer-vision

**Spécificité :** Principe de l'ArcFace Loss

L'ArcFace Loss est une fonction sur laquelle se repose le modèle ArcFace, et permet notamment d'optimiser la marge de distance géodésique (*généralisation d'une ligne droite du plan ou de l'espace euclidien en courbe*) parmi les features, permettant ainsi de donner des résultats plus précis et robustes.


---

### Comparaison des deux approches

| Critère | ResNet50 | ArcFace |
|---|---|---|
| Architecture | ResNet50 + GlobalAvgPool | ResNet backbone |
| Dataset d'entraînement | ImageNet | MS1MV3 |
| Spécialisation | Généraliste (features visuelles) | Spécialisé visages |
| Fonction de perte | Cross-entropy (classification) | ArcFace Loss |
| Embeddings | 2048-D | 512-D |
| Détection faciale | Non (image entière) | Oui (RetinaFace) |
| Vitesse CPU | ~300ms/image | ~150ms/image |
| Robustesse visages | Moyenne | Très bonne |

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

![roc_courbe](https://hackmd.io/_uploads/r1xlI3m2bg.png)

>Page des résultats pour un lancement de l'évaluation LFW.

---

## Installation & Lancement

### Prérequis

- Python **3.10, 3.11 ou 3.12** (Attention, Python 3.13 non supporté)
- WSL, Windows, Linux ou macOS
- 8 Go de RAM minimum
- Connexion internet (téléchargement des poids des modèles au premier lancement)

### Lancement rapide automatique (recommandé)

Pour un lancement rapide, j'ai choisi `uv`, qui est beaucoup plus rapide que pip pour fetcher et installer les packages. Pour mon système sans GPU, ça a beaucoup aidé.

Le script `launch_fast.sh` gère tout rapidement et automatiquement :

```bash
# 1. Cloner le projet via Github
git clone https://github.com/Dinah-piouf/fiac.git
cd fiac

# 2. Lancer le script
bash launch_fast.sh
```

Le script va automatiquement :
1. Détecter ou installer une version Python compatible (via pyenv si nécessaire)
2. Créer un environnement virtuel isolé
3. Installer toutes les dépendances
4. Lancer l'application sur `http://localhost:8501`

> Attention, le processus peut être très long au premier lancement, jusqu'à 10 min selon la connexion (TensorFlow et PyTorch pèsent plus de 1,5 Go au total).

> 📸 *[Insérer ici un screenshot du terminal pendant le lancement]*

### Lancement classique — via pip

```bash
# 1. Cloner le projet
git clone https://github.com/Dinah-piouf/fiac.git
cd fiac

# 2. Lancer le script
bash launch.sh
```

On peut ensuite ouvrir **http://localhost:8501** dans le navigateur.

---

## Structure du projet

```
face-id-systeme/
│
├── face_recognition_app.py   ← Application complète (code unique)
├── launch.sh                 ← Script d'installation & lancement classique (pip)
├── launch_fast.sh            ← Script d'installation & lancement rapide (uv)
├── README.md                 ← Ce fichier
│
├── data/                     ← Dataset LFW (généré automatiquement via l'application)
│   ├── raw/
│   ├── processed/
│   └── pairs/
│
└── evaluation/               ← Résultats d'évaluation
    ├── plots/                ← Courbes ROC exportées
    └── results/              ← Métriques JSON par modèle
```

> Le fichier `face_recognition_app.py` est **auto-suffisant** : il contient toutes les étapes de la pipeline (préparation des données, modèles, évaluation, interface Streamlit). Il est aussi très malléable — on peut interchanger les modèles de reconnaissance comme on le souhaite, ou en rajouter d'autres à sa guise.

---

## Guide d'utilisation

### Onglet 1 — Vérification par image

1. Sélectionner un modèle dans la barre latérale (**ResNet50** ou **ArcFace**)
2. Ajuster le seuil de décision si nécessaire (défaut : 0.80 pour ResNet50, 0.30 pour ArcFace)
3. Uploader une **image de référence** (photo connue de la personne) à l'upload de gauche
4. Uploader une **image à vérifier** à l'upload de droite
5. Cliquer sur **Vérifier l'identité**
6. Lire le résultat : score de similarité + dit s'il y a match ou pas

https://github.com/user-attachments/assets/cc3fc8ac-b400-4677-acdc-218a37597e2c

>Onglet de vérification / comparaison de deux images

---

### Onglet 2 — Webcam temps réel

1. Uploader une **image de référence**
2. Cliquer sur **Démarrer la webcam**
3. Le système compare en continu le flux webcam à la référence
4. Cliquer sur **Arrêter** pour terminer

https://github.com/user-attachments/assets/6d7da608-ecc1-4c3a-b018-221c0228f6f2

>Onglet de vérificaction webcam en temps réel.

---

### Onglet 3 — Évaluation LFW

1. Choisir le nombre de paires à évaluer (50 à 500)
2. Cliquer sur **Lancer l'évaluation**
3. Patienter pendant le calcul des embeddings
4. Consulter le tableau comparatif FAR / FRR / AUC
5. Visualiser et télécharger les courbes ROC

https://github.com/user-attachments/assets/a3ecde0e-b699-436c-975b-84cd9b678f52

>Onglet d'évaluation LFW


### Onglet 4 - Environnement

Cet onglet est à disposition afin de voir en temps réel l'état des dépendances, et un récap de comment lancer l'application.

https://github.com/user-attachments/assets/9106674c-2894-4c2d-91b2-5afae6143fa8

>Onglet environnement

---

### Barre latérale — paramètres

| Paramètre | Description |
|---|---|
| **Modèle** | Basculer entre ResNet50 et ArcFace |
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

## Autrice

Andréa TOMAS, déléguée et étudiante de M1 CSSD (Cybersécurité et Science des Données), à l'Université Paris 8.
Projet réalisé dans le cadre d'un cours de deep learning appliqué à la reconnaissance faciale.

---

## Licence

Ce projet est à usage académique. Les modèles pré-entraînés sont soumis aux licences de leurs auteurs respectifs (VGGFace2, InsightFace).

---

## Précautions

Ce projet a été assisté par IA, notamment pour la rédaction du launch.sh et launch_fast.sh pour une installation rapide, claire, et pour comprendre les bases et erreurs récurrentes de steamlit.
