# Rapport TP01 : Traitement des images avec PIL

## Partie 1 : Lecture et sauvegarde
**Objectif technique :** Charger l'image en mémoire à l'aide de la bibliothèque Pillow et la convertir dans un format de travail standard (RGB) avant de l'enregistrer.
**Effet visuel :** L'image originale de notre personnage est conservée à l'identique, avec l'ensemble de ses couleurs et de ses détails intacts.

## Partie 2 : Redimensionnement
**Objectif technique :** Modifier la résolution spatiale de l'image (ex. 300x150 pixels) à l'aide de la méthode `resize()` de PIL, afin de standardiser ou réduire son format.
**Effet visuel :** L'image est étirée ou compressée selon les proportions imposées. Les traits du personnage apparaissent déformés puisqu'on ne conserve pas le ratio d'aspect initial.

## Partie 3 : Augmentation de la luminosité
**Objectif technique :** Appliquer un facteur multiplicatif (x1.5) aux valeurs des pixels via le module `ImageEnhance.Brightness` pour rendre l'image globalement plus claire.
**Effet visuel :** Les couleurs de la tenue du personnage s'éclaircissent et l'image parait plus exposée à la lumière, révélant certains détails sombres avec potentiellement un léger délavé sur les zones les plus claires.

## Partie 4 : Conversion en niveaux de gris
**Objectif technique :** Utiliser la méthode `convert("L")` pour passer l'image dans un espace colorimétrique codé sur un seul canal 8 bits (valeurs de 0 à 255).
**Effet visuel :** Les informations chromatiques disparaissent. Le personnage n'est plus représenté que par des variations d'intensité lumineuse allant du noir pur au blanc pur.

## Partie 5 : Binarisation
**Objectif technique :** Appliquer un seuillage manuel (fixé à 128) sur l'image en niveaux de gris via la méthode `point()`. Les pixels plus sombres deviennent 0 et les plus clairs 255 (mode "1").
**Effet visuel :** L'image devient purement monochrome en noir et blanc très contrasté. Les ombres et dégradés disparaissent au profit de contours pleins simplifiant la représentation globale.

## Partie 6 : Détection de contours
**Objectif technique :** Appliquer un filtre convolutionnel (via `ImageFilter.FIND_EDGES` de PIL) pour surligner les fortes variations d'intensité lumineuse spatiales.
**Effet visuel :** Le fond de l'image devient majoritairement sombre tandis que les bords marquants des cheveux, des vêtements et des traits du visage du personnage se détachent par de fins tracés clairs, comme esquissés.

## Partie 7 : Flou Gaussien
**Objectif technique :** Appliquer un filtre passe-bas bidimensionnel `ImageFilter.GaussianBlur(rayon=R)` pour atténuer les hautes fréquences de l'image de façon progressive selon l'augmentation du rayon (1, 2 et 3).
**Effet visuel :** Les traits et les détails se brouillent proportionnellement au rayon choisi. On observe un adoucissement global à rayon=1 jusqu'à une perte significative de netteté empêchant de distinguer finement les textures à rayon=3.

## Partie 8 : Tracé de l'histogramme
**Objectif technique :** Calculer la distribution des fréquences d'apparition de chaque intensité lumineuse dans l'image en niveaux de gris via la méthode `histogram()` et tracer la courbe via Matplotlib.
**Effet visuel :** Représentation graphique avec une courbe qui révèle la répartition précise des tons (majorité de tons sombres, moyens ou clairs) de l'image grise initiale du personnage.

## Partie 9 : Égalisation de l'histogramme
**Objectif technique :** Repartir uniformément les contrastes de l'image via `ImageOps.equalize()` afin d'optimiser l'échelle des intensités sur tout le spectre [0, 255].
**Effet visuel :** L'image en niveaux de gris gagne considérablement en contraste de façon homogène. Les zones anciennement ternes se révèlent avec beaucoup plus de précision, rendant certains détails peu contrastés de la tenue plus évidents.
