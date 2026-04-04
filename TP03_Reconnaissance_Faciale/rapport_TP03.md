# Rapport TP03 : Reconnaissance Faciale (Viola-Jones et LBP)

## Méthodologie combinée : Viola-Jones + LBP
L'approche technique déployée dans ce TP repose sur une chaîne d'analyse rapide en deux étapes successives garantissant l'identification précise d'un visage :
1. **Viola-Jones (Cascade de Haar)** : Ce filtre algorithmique très performant analyse des blocs de contraste rectangulaires pour localiser géométriquement la boîte englobante (Bounding Box) d'un visage. Dans notre pipeline, c'est cette étape qui permet de croper pertinemment la zone d'étude.
2. **Local Binary Patterns (LBP)** : L'extracteur LBP travaille intimement avec la texture. Chaque pixel du visage découpé et redimensionné est comparé à son voisinage (les 8 pixels adjacents). On forme ainsi un schéma binaire révélant les microstructures locales, qu'on rassemble ensuite dans un histogramme normé résilient aux problématiques d'illuminance du scanner ou de la caméra.

## Conclusion de l'évaluation
Cette association d'un détecteur global (Viola-Jones) pour la topologie complété par un outil macro/micro-analytique en temps-réel (LBP) garantit une méthode de vérification à la fois rapide et robuste. 
Lors de l'application sur notre sujet généré, la distance euclidienne extraite de l'histogramme des deux images (`reference` vs `test` incluant un angle/éclairage altéré) a déterminé avec exactitude si la signature texturelle correspondait logiquement. Les performances de détection prouvent que la méthode LBP tolère remarquablement la variation d'angle légère sur le visage.
