# Biométrie & Tatouage - ING-4 SSIRF

Bienvenue sur le dépôt officiel regroupant l'ensemble des travaux pratiques (TP) réalisés dans le cadre du module **Biométrie & Tatouage** (ING-4-SSIRF). Ce projet explore différentes techniques de l'analyse d'image et de la reconnaissance biométrique, allant du simple traitement d'image aux modèles de Deep Learning récents.

## Contenu du Dépôt

1. **[TP01 : Traitement des images avec PIL](./TP01_Traitement_Images_PIL)** 
   Introduction aux fondamentaux de la manipulation et la transformation d'images en Python en utilisant Pillow (Lecture, Redimensionnement, Filtres, Flous, et Histogrammes).

2. **[TP02 : Reconnaissance d’Empreinte Digitale (SSIM)](./TP02_Reconnaissance_Empreinte)** 
   Vérification et alignement d'empreintes digitales par l'utilisation de l'Index de Similarité Structurelle (SSIM) basé sur le pré-traitement intensif via l'espace de fréquence et scikit-image.

3. **[TP03 : Reconnaissance Faciale par LBP et Viola-Jones](./TP03_Reconnaissance_Faciale)** 
   Détection de visage avec les cascades de Haar (Viola-Jones) et extraction des caractéristiques locales (Local Binary Patterns) pour un premier modèle de comparaison paramétrique.

4. **[TP04 : Reconnaissance Faciale par Analyse en Composantes Principales (PCA)](./TP04_Reconnaissance_Faciale_PCA)** 
   Mesure biométrique par la méthode des Eigenfaces. Implémentation manuelle de l'ACP via algorithme SVD (Singular Value Decomposition). Modélisation des courbes de performance **FAR/FRR**.

5. **[TP05 : Reconnaissance Faciale par Deep Learning](./TP05_Reconnaissance_Faciale_DL)** 
   Atteinte de l'état de l'art de la précision via des convolutions (Dlib). L'extraction de signatures faciales ultra résilientes en vecteurs purs (128 Dimensions) et mapping mathématique des distances par Cosinus ou matrice Euclidienne.

6. **[TP06 : Tatouage Numérique par LSB (Spatial)](./TP06_Tatouage_Numerique_LSB)** 
   Implémentation du tatouage numérique LSB (Least Significant Bit) pour la dissimulation de messages invisibles dans des images en niveaux de gris et RGB. Introduction d'une version sécurisée par accès pseudo-aléatoire via une clé secrète.

7. **[TP06.2 : Tatouage Numérique par Patchwork](./TP06.2_Tatouage_Numerique_Patchwork)** 
   Implémentation du tatouage statistique Patchwork. Algorithme additif robuste, simulations d'attaques (Bruit, JPEG, Flou) sur image RGB et étude de la robustesse comparée à la méthode LSB.

## Prérequis et Installation
Les dépendances complètes du projet sont :
- `Pillow`
- `numpy`, `scipy`, `matplotlib`
- `opencv-python`
- `scikit-image`
- `face_recognition` (ou de manière équivalente `dlib`)

---
*Auteur : Yosri Hamdouni*
