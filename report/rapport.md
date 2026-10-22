# Classification d'images CIFAR-10

**Projet individuel — Mastère IA, Hexagone**  
**Auteur :** Thomas Gassmann · **Date :** 17/10/2026

## 1. Contexte et objectifs

La classification d'images consiste à attribuer une catégorie à une image. CIFAR-10 est un petit jeu d'images couleur
souvent utilisé pour comparer des classifieurs. Le projet suit la démarche demandée : établir des références, augmenter
progressivement la capacité des modèles, ajuster les paramètres sur une validation, puis analyser les compromis entre
qualité et coût. La performance du test ne participe jamais au choix des hyperparamètres.

## 2. Présentation des données

CIFAR-10 contient 60 000 images RGB de 32 × 32 pixels : 50 000 images d'entraînement et 10 000 images de test, en dix
classes équilibrées. La source et la définition sont décrites par Krizhevsky, Nair et Hinton [1]. Les statistiques
exactes sont produites par `dataset_statistics` après téléchargement.

| Partition                  |                Taille | Utilisation                         |
|----------------------------|----------------------:|-------------------------------------|
| Entraînement officiel      |                50 000 | Ajustement des poids                |
| Validation                 | 5 000 (10 % du train) | Sélection des modèles et paramètres |
| Entraînement après découpe |                45 000 | Entraînement du benchmark complet   |
| Test officiel              |                10 000 | Évaluation finale uniquement        |

Le découpage validation est stratifié et déterministe (seed 42 par défaut). L'exploration emploie 4 000 exemples tirés
stratifiéement parmi les 45 000 exemples d'entraînement ; la validation reste inchangée.

## 3. Préparation des données

Les pixels sont convertis en tenseurs et normalisés avec les moyennes (0,4914 ; 0,4822 ; 0,4465) et écarts-types (
0,2470 ; 0,2435 ; 0,2616) usuels de CIFAR-10. L'augmentation de référence applique un recadrage aléatoire avec padding
de quatre pixels et un retournement horizontal. ColorJitter, RandomErasing, MixUp et CutMix sont options de comparaison,
pas une pile d'augmentations appliquées systématiquement. Le test et la validation utilisent uniquement la conversion et
la normalisation.

## 4. Modèles implémentés

Le MLP sert de référence qui ignore la structure spatiale locale. Le CNN personnalisé empile trois blocs convolutifs
avec Batch Normalization, ReLU et réduction spatiale, puis un classifieur avec dropout. Le ResNet-18 reprend les blocs
résiduels et remplace le stem ImageNet par une convolution 3 × 3 sans max-pooling initial, mieux adaptée aux images 32 ×
32 [2]. WideResNet-16-4 teste l'effet d'une architecture résiduelle plus large [3]. Tous les modèles apprennent depuis
zéro ; le transfert ImageNet n'est pas retenu car il changerait le protocole et ajouterait un facteur de comparaison.

## 5. Hyperparamètres et protocole expérimental

Les graines Python, NumPy et PyTorch sont fixées. Le code choisit CUDA, MPS, puis CPU si `device=auto`. La phase
exploratoire compare quatre architectures sur un sous-ensemble et cinq époques par défaut. La phase complète compare le
CNN, ResNet et WideResNet sur le train disponible, avec SGD, momentum 0,9, weight decay 5×10⁻⁴ et décroissance cosinus.
L'arrêt anticipé surveille la perte de validation. La phase d'optimisation réalise trois configurations ResNet
contrôlées : référence SGD, MixUp avec label smoothing, puis AdamW. Les paramètres détaillés sont conservés avec les
checkpoints.

**À compléter après entraînement :** tableau des durées, configuration machine, versions logicielles et liste effective
des commandes/expériences exécutées.

## 6. Résultats individuels

Aucun résultat n'est encore disponible. Pour chaque modèle, les scripts enregistrent les pertes et exactitudes par
époque, le meilleur epoch sélectionné par perte de validation, le nombre de paramètres, le temps d'entraînement et le
temps d'inférence.

**À compléter après entraînement :** courbes train/validation, matrices de confusion, précision/rappel/F1 macro,
accuracy par classe, exemples de bonnes prédictions et erreurs. Ne reporter les métriques test qu'après choix final du
modèle.

## 7. Comparaison des modèles

**Phase d'exploration** (Apple MPS, 4 000 images d'entraînement, 5 époques, AdamW, validation de 5 000 images ; source :
`results/metrics/exploration_runs.json`).

| Modèle          | Paramètres | Accuracy validation | Temps entraînement | Inférence / image |
|-----------------|-----------:|--------------------:|-------------------:|------------------:|
| MLP             |  1 707 274 |              36,3 % |               45 s |           0,08 ms |
| CNN             |    305 258 |              38,1 % |              125 s |           1,00 ms |
| WideResNet-16-4 |  2 748 890 |              48,2 % |              391 s |           4,89 ms |
| ResNet-18 CIFAR | 11 173 962 |              55,1 % |              511 s |           6,03 ms |

Pour les quatre modèles le meilleur epoch est le dernier et la validation augmente encore : les réseaux sont sous-entraînés.
Ce tableau ne permet donc pas de conclure sur la capacité finale des architectures ; seul l'ordre MLP/CNN contre
réseaux résiduels est net. Les temps incluent le démarrage de MPS (première époque nettement plus longue) et ont été
mesurés avec les algorithmes déterministes activés, ×2,5 plus lents que le réglage par défaut actuel.

**Phase complète** (45 000 images, 30 époques, SGD Nesterov, lr 0,1, cosinus ; validation de 5 000 images).

| Modèle          | Paramètres | Accuracy validation | Temps entraînement | Inférence / image |
|-----------------|-----------:|--------------------:|-------------------:|------------------:|
| CNN             |    305 258 |              87,4 % |             22 min |    0,15 ms |
| ResNet-18 CIFAR | 11 173 962 |              93,2 % |    ≈ 3 h (estimé*) |           0,63 ms |

\* L'entraînement de ResNet-18 a été interrompu à l'époque 29 sur 30, puis terminé par une 30e époque au learning rate
cosinus correspondant (poids rechargés, optimiseur réinitialisé). L'historique 1-29 provient du journal d'exécution (sans
`train_loss`), et le temps total est extrapolé de la dernière époque. WideResNet-16-4 n'a pas été entraîné en phase
complète, faute de temps de calcul.

**Scores test et F1 macro : à compléter** après `evaluate` sur le seul modèle retenu. Le benchmark automatique n'inclut un score test que si une évaluation finale existe.

## 8. Analyse critique

**À compléter après entraînement.** Comparer les confusions principales et les écarts entre entraînement et validation (
surapprentissage), relier le gain de performance au nombre de paramètres et au temps. Décrire les classes les plus
difficiles avec leurs précisions/rappels. Une amélioration sur validation n'est convaincante que si le test, consulté en
fin de protocole, confirme la tendance. Une seed unique limite l'estimation de la variabilité ; une répétition sur
plusieurs seeds est souhaitable si le calcul le permet.

## 9. Conclusion et perspectives

**À compléter après les expériences.** Retenir le meilleur modèle selon la validation et vérifier une fois sur le test.
Discuter le compromis performance/temps/mémoire. Les prolongements possibles sont les répétitions multi-seeds, une
recherche plus large mais préenregistrée, un test-time augmentation justifié par validation, et l'étude de modèles
pré-entraînés en présentant cette approche séparément.

## 10. Références

1. A. Krizhevsky, V. Nair, G. Hinton, *The CIFAR-10 dataset*, page officielle du jeu de données, University of
   Toronto, https://www.cs.toronto.edu/~kriz/cifar.html.
2. K. He, X. Zhang, S. Ren, J. Sun, “Deep Residual Learning for Image Recognition,” CVPR
   2016, https://arxiv.org/abs/1512.03385.
3. S. Zagoruyko, N. Komodakis, “Wide Residual Networks,” BMVC 2016, https://arxiv.org/abs/1605.07146.
4. H. Zhang et al., “mixup: Beyond Empirical Risk Minimization,” ICLR 2018, https://arxiv.org/abs/1710.09412.
5. S. Yun et al., “CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,” ICCV
   2019, https://arxiv.org/abs/1905.04899.
6. PyTorch, documentation d'installation et API, https://pytorch.org/docs/stable/index.html ; scikit-learn,
   `StratifiedShuffleSplit`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html.
7. OpenAI ChatGPT (modèle et version à préciser), utilisé comme assistant à la rédaction et à l'implémentation ; toutes
   les expériences et métriques sont produites et vérifiées par les scripts du projet.
