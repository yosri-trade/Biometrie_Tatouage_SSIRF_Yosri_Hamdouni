# Rapport TP02 : Reconnaissance d'Empreinte Digitale (Partie D)

## Approche SSIM (Structural Similarity Index)
L'approche **SSIM** est une méthode permettant de mesurer la similitude structurelle entre deux images, s'appuyant sur la luminance, le contraste et la structure des motifs. Contrairement à une simple comparaison pixel par pixel, le SSIM évalue de manière globale la géométrie, offrant ainsi une métrique performante pour comparer la cohérence des minuties (crêtes, sillons) entre deux prises d'empreinte.

## Utilité des étapes de prétraitement
L'objectif du prétraitement strict via Pillow est d'isoler au maximum l'information utile et pure (les arêtes de l'empreinte) de facteurs très variables (ombres, colorimétrie, éclairage du scanner).
1. **Niveaux de gris et Égalisation** : Normalise la dynamique lumineuse et maximise le contraste pour que toutes les stries ressortent uniformément.
2. **Binarisation** : Force les pixels au sein de valeurs binaires opposées, éliminant le bruit de fond et rendant la séparation crête/creux très franche.
3. **Extraction de contours** : Souligne géométriquement les frontières claires ou tranchantes (les rebords des crêtes), favorisant la comparaison topographique pure par l'algorithme SSIM.

## Résultat de l'expérience (Score SSIM)
Lors de l'analyse conjointe sur nos deux images sources, la similarité de structure fine de nos deux empreintes est calculée par l'algorithme de skimage. Si les deux traces prétraitées possèdent une conformation géométrique se superposant à plus de 75% (soit `SSIM >= 0.75`), le système biométrique décide l'admission de l'utilisateur (`ACCEPTÉE`). Sinon, elle est logiquement `REJETÉE`. 
La décision finale rendue de cette exécution est illustrée visuellement sur le fichier de rendu direct : `results/comparaison_ssim.png`.
