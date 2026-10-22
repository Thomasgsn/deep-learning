# Classification CIFAR-10

Projet de Mastère IA : comparer des réseaux de complexité croissante, choisir les réglages sur une validation séparée, puis mesurer une seule fois le modèle retenu sur le test officiel. Les expériences et leurs limites sont décrites dans `report/rapport.md`.

## Données

CIFAR-10 contient 60 000 images couleur de 32 × 32 pixels réparties en 10 classes : 50 000 images d'entraînement et 10 000 images de test. Le code réserve 10 % du train officiel à une validation stratifiée, reproductible avec la seed 42. Le test officiel n'est chargé que par la commande d'évaluation finale.

## Installation

Python 3.11 ou plus récent est recommandé. Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les commandes suivantes téléchargent automatiquement CIFAR-10 au premier lancement dans `data/`.

## Utilisation

### Vérifier le pipeline sans télécharger les données

```bash
PYTHONPATH=src python scripts/smoke_test.py
PYTHONPATH=src python -m pytest
```

### Exploration rapide (4 000 images d'entraînement, 5 époques par défaut)

```bash
PYTHONPATH=src python -m cifar10_project.experiments exploration --device auto
```

### Benchmark complet, puis optimisation limitée

```bash
PYTHONPATH=src python -m cifar10_project.experiments complete --device auto --epochs 40
PYTHONPATH=src python -m cifar10_project.experiments optimisation --device auto --epochs 50
```

`auto` sélectionne CUDA, Apple MPS, puis CPU. Les checkpoints, historiques et résumés sont écrits dans `checkpoints/` et `results/metrics/`. Les phases complètes peuvent durer longtemps sur CPU ; `--epochs` permet de régler le budget. Pour un essai personnalisé, copier `configs/cnn_smoke.json`, modifier ses paramètres, puis lancer :

```bash
PYTHONPATH=src python -m cifar10_project.train configs/cnn_smoke.json
```

### Évaluation finale et graphiques

Choisir le checkpoint après consultation de la validation, puis seulement lancer l'évaluation test :

```bash
PYTHONPATH=src python -m cifar10_project.evaluate checkpoints/complete_resnet18/best.pt
PYTHONPATH=src python -m cifar10_project.cli figures --checkpoint checkpoints/complete_resnet18/best.pt --history checkpoints/complete_resnet18/history.csv
PYTHONPATH=src python -m cifar10_project.cli compare
```

Les figures exploratoires et les tableaux sont générés sous `results/`. Les images d'erreurs/correctes sont extraites du test ; elles sont donc à produire après verrouillage du modèle. Pour les statistiques et graphiques de classes :

```bash
PYTHONPATH=src python -m cifar10_project.cli stats
PYTHONPATH=src python -m cifar10_project.cli figures
```

## Modèles comparés

- **MLP** : référence sans traitement spatial explicite.
- **CNN personnalisé** : convolutions, normalisation, ReLU, pooling et dropout.
- **ResNet-18 CIFAR** : connexions résiduelles, avec une première convolution adaptée aux petites images.
- **WideResNet-16-4** : comparaison d'un réseau résiduel moins profond et plus large.

Le transfert depuis ImageNet n'est pas utilisé : les images CIFAR-10 sont très petites et le protocole cible l'apprentissage depuis zéro, dans un budget maîtrisable. Les scripts permettent de comparer augmentation standard, ColorJitter ou RandomErasing dans une configuration personnalisée, ainsi que MixUp/CutMix en entraînement.

## Résultats

Seule la phase d'**exploration** a été exécutée (Apple MPS, 4 000 images d'entraînement, 5 époques, AdamW, validation de 5 000 images, aucun test consulté) :

| Modèle          | Paramètres | Accuracy validation | Entraînement | Inférence / image |
|-----------------|-----------:|--------------------:|-------------:|------------------:|
| MLP             |  1 707 274 |              36,3 % |        45 s  |          0,08 ms  |
| CNN             |    305 258 |              38,1 % |       125 s  |          1,00 ms  |
| WideResNet-16-4 |  2 748 890 |              48,2 % |       391 s  |          4,89 ms  |
| ResNet-18       | 11 173 962 |              55,1 % |       511 s  |          6,03 ms  |

Phase **complète** (45 000 images, 30 époques, SGD Nesterov lr 0,1, cosinus ; validation de 5 000 images, test non consulté) :

| Modèle          | Paramètres | Accuracy validation | Entraînement | Inférence / image |
|-----------------|-----------:|--------------------:|-------------:|------------------:|
| CNN             |    305 258 |              87,4 % |      22 min  |          0,15 ms |
| ResNet-18       | 11 173 962 |              93,2 % |  ≈ 3 h (estimé) |        0,63 ms |

L'entraînement de ResNet-18 a été interrompu à l'époque 29 sur 30, puis terminé par `scripts/finish_resnet18.py` (poids rechargés, optimiseur réinitialisé, 30e époque au learning rate cosinus correspondant). L'historique 1-29 a été reconstruit depuis `logs/complete.log` : pas de `train_loss` ni de durées pour ces époques, et le temps d'entraînement est une extrapolation (`training_seconds_estimated`). WideResNet-16-4 et la phase `optimisation` n'ont pas été exécutées en phase complète.

Les poids (`*.pt`) ne sont pas versionnés : régénérer avec `experiments complete`. `summary.json`, `history.csv` et `logs/` le sont pour garder la trace des expériences.

Ces scores d'exploration sont ceux de modèles encore loin de la convergence (la validation progresse à chaque époque, le meilleur epoch est toujours le dernier). Ils servent à écarter les choix aberrants, pas à classer définitivement les architectures. Les phases `complete` et `optimisation`, l'évaluation test et les figures restent à lancer ; `python -m cifar10_project.cli compare --prefix complete_` produit ensuite le tableau à partir des résumés enregistrés.

`deterministic` (clé de configuration, `false` par défaut) active les algorithmes déterministes de PyTorch : ×2,5 plus lent sur MPS (ResNet-18, 1,95 s contre 0,77 s par pas de 128 images), à réserver à une vérification de reproductibilité exacte. Les graines restent fixées dans tous les cas.

## Limites

Pas de transfert ImageNet, pas de répétitions multi-seeds, budget de calcul limité à un Mac (MPS) : l'objectif de la phase complète est d'approcher 93–95 % avec ResNet-18, pas de viser l'état de l'art.
