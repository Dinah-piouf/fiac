#!/bin/bash
# ================================================================================================================================
# FIAC — Script de correction de l'évaluation de LFW en courbe ROC
# Usage : bash lfw_helper.sh
# Permet de réparer les erreurs faites par fetch_lfw_people(), si jamais le téléchargement se fait mal. (Problème PC peu puissant)
# ================================================================================================================================

#Kaggle va nous permettre de chercher la base de données LFW de manière sûre (si les dépôts officiels ne sont pas joignables)
pip install kaggle
kaggle datasets download -d jessicali9530/lfw-dataset -p data/lfw_home/

#On va dézipper afin d'avoir toutes les images
unzip data/lfw_home/lfw-dataset.zip -d data/lfw_home/


#On va rechercher les fichiers .txt qui vont nous aider pour l'évaluation LFW
#curl -L "https://raw.githubusercontent.com/grib0ed0v/face_recognition.pytorch/master/data/pairs.txt" -o data/lfw_home/pairs.txt
#curl -L "https://raw.githubusercontent.com/grib0ed0v/face_recognition.pytorch/master/data/pairsDevTrain.txt" -o data/lfw_home/pairsDevTrain.txt
#curl -L "https://raw.githubusercontent.com/grib0ed0v/face_recognition.pytorch/master/data/pairsDevTest.txt" -o data/lfw_home/pairsDevTest.txt