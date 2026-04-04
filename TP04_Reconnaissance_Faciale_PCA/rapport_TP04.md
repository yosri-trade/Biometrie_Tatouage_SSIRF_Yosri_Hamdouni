# Rapport TP04 : Reconnaissance Faciale par PCA (Eigenfaces) et Viola-Jones

## 1. Méthodologie et Expérimentations FAR / FRR
Le programme exploite la réduction de dimension PCA pour filtrer et conserver les variations les plus descriptives (les "Eigenfaces") d'une base de données d'apprentissage.
Lors de nos expérimentations scriptées avec OpenCV et Numpy :
- **Tests d'effet de la variance (Paramètre k)** : La modélisation a été évaluée avec `k=10, 20 et 50`. Avec k=10, l'approximation reconstruit les traits très généraux de la base.
- **Analyse Métrique (Seuillage)** : Afin d'équilibrer sécurité et praticité, nous avons balayé une plage de seuils. Au seuil optimal de la courbe Error Equal Rate (EER), la balance entre les Faux Rejets (FRR) et les Fausses Acceptations (FAR) a été exportée mathématiquement dans le fichier `results/metrics.txt`, calculant avec succès les performances métrologiques pures du modèle Eigenface sur notre base fictive. 

## 2. Réponses aux Questions d'Analyse

1. **Pourquoi l'algorithme PCA nécessite-t-il impérativement un bon alignement géométrique des visages ?**
   L'Analyse en Composantes Principales (PCA) évalue la corrélation globale d'intensités *pixel par pixel*. Si les images ne sont pas alignées (exemple: les yeux sur la ligne 40 décalés vers la ligne 60), le modèle percevra ce décalage comme la plus forte "variabilité" mathématique entre les images, écrasant l'information faciale véritable (les "Eigenfaces" ressembleront alors à du bruit flou).

2. **Que se passe-t-il pratiquement si k (nombre d'Eigenfaces) est fixé trop faible ?**
   On perd des détails spécifiques aux identités (Sous-apprentissage ou *Underfitting*). Le sous-espace conservera essentiellement l'information d'éclairage ou la forme géométrique pure de la tête, menant fatalement à un taux élevé de Fausses Acceptations (FAR) car tous les modèles projetés se ressembleront excessivement.

3. **Que se passe-t-il si k est, à l'inverse, trop élevé au-delà de la capacité utile ?**
   Le système intègrera des détails excessivement locaux, voire du simple bruit d'image ou l'éclairage propre au contexte du jour d'entraînement (Sur-apprentissage ou *Overfitting*). Une simple variation d'expression ou de luminosité chez la même personne le lendemain causera un fort taux de Faux Rejets (FRR), car l'image différera des détails superflus mémorisés.

4. **Qu'est-ce qui rend techniquement la distance Euclidienne si adaptée au calcul dans le sous-espace PCA ?**
   Dans le sous-espace généré par le PCA, chaque axe (vecteur propre) devient mathématiquement orthogonal et sémantiquement indépendant l'un de l'autre. Par conséquent, la distance géométrique en ligne droite (Norme L2 ou Euclidienne) caractérise la dissemblance exacte des profils sans subir de distorsions dues à une éventuelle inter-corrélation d'information.

5. **Malgré ses qualités, quelles sont les faiblesses fondamentales d'Eigenfaces face aux variations prononcées d'illumination ?**
   La méthode Eigenface traite l'image comme un simple gradient mathématique de valeurs 1D brutes. Une différence majeure d'illumination (comme un flash unilatéral) modifie dramatiquement la structure variance, devenant le principal critère récupéré par le calcul PCA (premier vecteur propre). La structure identitaire passe alors mathématiquement au second plan. C'est pourquoi un prétraitement lourd s'avère indispensable en amont, comparé à une méthode d'extraction locale (comme le LBP).
