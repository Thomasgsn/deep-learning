# Challenge Computer Vision CIFAR-10


J'ai construit un benchmark reproductible de modèles de complexité croissante sur CIFAR-10. Mon protocole sépare strictement entraînement, validation et test : je sélectionne architectures et hyperparamètres sur 5 000 images de validation, puis j'évalue uniquement le modèle verrouillé sur les 10 000 images du test officiel.

## Accès rapide pour la correction

Le rapport et les résultats sont consultables sans installer Python ni relancer les entraînements.

| Document ou résultat | Emplacement |
|---|---|
| **Rapport final PDF** | **[report/rapport_cifar10.pdf](report/rapport_cifar10.pdf)** |
| Comparaison finale des modèles | [results/comparisons/final_primary_comparison.csv](results/comparisons/final_primary_comparison.csv) |
| Résultats test du modèle retenu, avec scores par classe | [results/metrics/complete_cuda_wide_resnet_test.json](results/metrics/complete_cuda_wide_resnet_test.json) |
| Synthèse des trois graines | [results/metrics/multiseed_summary.json](results/metrics/multiseed_summary.json) |
| Matrice de confusion sur le test | [results/figures/complete_cuda_wide_resnet_test_confusion.png](results/figures/complete_cuda_wide_resnet_test_confusion.png) |
| Exemples de prédictions | [Succès](results/figures/correct.png) · [Erreurs](results/figures/errors.png) |
| Guide de tous les résultats | [results/README.md](results/README.md) |
| Historiques d'entraînement et résumés des expériences | [checkpoints/README.md](checkpoints/README.md) |
| Poids du modèle final et provenance | [artifacts/final_model.pt](artifacts/final_model.pt) · [artifacts/final_model.json](artifacts/final_model.json) |

## Résultat principal

J'ai retenu WideResNet-16-4 après comparaison multi-seeds sur validation. Le checkpoint principal utilise la seed 42 préenregistrée ; je n'ai pas choisi a posteriori la seed la plus favorable.

| Mesure finale | Valeur |
|---|---:|
| Accuracy test | **93,16 %** |
| Macro-F1 test | **93,16 %** |
| Précision macro | 93,17 % |
| Rappel macro | 93,16 % |
| NLL | 0,2269 |
| ECE, 15 bins | 2,65 % |
| Paramètres entraînables | 2 748 890 |
| Latence modèle seul, batch 1 | 2,016 ms/image |
| Latence modèle seul, batch 128 | 0,1045 ms/image |

Le checkpoint évaluable est `artifacts/final_model.pt` (10,5 Mio), SHA-256 `a9f98c2d9394ab450a2c1a97b776bc5641fa07ce78071f9572e71d05c28bf933`. Son manifeste se trouve dans `artifacts/final_model.json`.

## Protocole expérimental

- L'exploration historique a été exécutée sur Apple MPS. Le benchmark final et l'optimisation documentés ci-dessous utilisent CUDA sur une NVIDIA GeForce RTX 4060. Les temps mesurés sur ces matériels ne sont pas directement comparables.
- L'exploration couvre MLP, CNN, ResNet-18 et WideResNet-16-4 sur cinq époques avec AdamW. Le benchmark complet CUDA porte sur CNN, ResNet-18 et WideResNet-16-4 ; aucun résultat complet du MLP n'est fourni.
- CIFAR-10 contient 50 000 images d'entraînement et 10 000 images test, réparties équitablement entre dix classes.
- Je réserve 10 % du train officiel à une validation stratifiée : 45 000 images servent à apprendre et 5 000 à sélectionner.
- L'exploration et l'optimisation limitée utilisent 4 000 images tirées uniquement du train ; la validation reste fixe.
- Le train reçoit crop aléatoire avec padding 4 et retournement horizontal. Validation et test n'utilisent aucune augmentation stochastique.
- Les modèles complets utilisent SGD Nesterov, momentum 0,9, weight decay 5×10⁻⁴, learning rate initial 0,1 et annealing cosinus pendant 30 époques.
- Le meilleur checkpoint d'un run est celui qui minimise la perte de validation.
- Le test n'a participé à aucune décision. Une répétition technique après verrouillage a confirmé exactement la même matrice et les mêmes métriques.

## Résultats de validation et coût

Les moyennes et écarts-types portent sur les seeds 42, 123 et 2026 pour WideResNet et ResNet. Le CNN n'a été exécuté qu'avec la seed 42.

| Modèle | Paramètres | Accuracy validation | Temps d'entraînement | Inférence batch 128 | Pic GPU |
|---|---:|---:|---:|---:|---:|
| CNN | 305 258 | 87,14 % | 3 min 37 s | 0,0138 ms/image | 364 Mo |
| WideResNet-16-4 | 2 748 890 | **93,40 ± 0,11 %** | 8 min 47 s ± 9 s | 0,1040 ± 0,0017 ms/image | 1 036 Mo |
| ResNet-18 CIFAR | 11 173 962 | 93,07 ± 0,31 % | 11 min 44 s ± 10 s | 0,1350 ± 0,0009 ms/image | 2 269 Mo |

WideResNet présente ici le meilleur compromis : +0,33 point de validation en moyenne face à ResNet-18, environ quatre fois moins de paramètres, 25 % de temps d'entraînement en moins et 54 % de mémoire GPU en moins. La différence d'accuracy reste modeste et propre à ce protocole ; je ne la généralise pas au-delà de ces trois seeds.

## Optimisation contrôlée

J'ai comparé trois configurations ResNet-18 sur le même sous-ensemble de 4 000 images, la même validation, la seed 42 et dix époques. Un seul facteur change à la fois.

| Variante | Accuracy validation | Perte validation | Durée |
|---|---:|---:|---:|
| Référence | **31,12 %** | **1,8436** | 65,7 s |
| MixUp seul, α=0,2 | 30,42 % | 1,8755 | 52,6 s |
| Label smoothing seul, ε=0,1 | 29,14 % | 1,9904 | 54,8 s |

Ces modèles sont volontairement sous-entraînés : ce tableau sert uniquement à comparer des réglages sous budget identique. Ni MixUp ni le label smoothing n'améliorent la référence dans ce cadre ; j'ai donc confirmé la référence sur les données complètes au lieu de fabriquer un gain.

## Installation

Python 3.11 ou plus récent est requis. L'environnement réellement validé utilise Python 3.12.13, PyTorch 2.14.1+cu132, torchvision 0.29.1+cu132, Windows 11 et une NVIDIA GeForce RTX 4060.

### Environnement CUDA reproduit

PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:PYTHONPATH = "src"
```

Linux/macOS ou environnement CPU :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements-cuda.txt` pointe vers l'index officiel PyTorch CUDA 13.2. `requirements-lock.txt` fige les autres versions réellement testées. `requirements.txt` reste l'installation portable avec plages de versions.

## Vérifications rapides

PowerShell :

```powershell
$env:PYTHONPATH = "src"
$env:MPLCONFIGDIR = ".tmp\matplotlib"
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

Vérification de la remise le 26 octobre 2026 : **17 tests réussis** ; smoke test réussi pour MLP, CNN, ResNet-18 et WideResNet, y compris entraînement, métriques et sauvegarde/restauration.

## Reproduire les expériences

Les commandes suivantes sont données pour PowerShell depuis la racine du dépôt. CIFAR-10 est téléchargé automatiquement dans `data/` au premier lancement.

### Optimisation limitée

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -u -m cifar10_project.experiments optimisation --device cuda --epochs 10
```

### Entraînements principaux

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -u -m cifar10_project.train configs\cnn_complete_cuda.json
.\.venv\Scripts\python.exe -u -m cifar10_project.train configs\wide_resnet_complete_cuda.json
.\.venv\Scripts\python.exe -u -m cifar10_project.train configs\resnet18_complete_ref.json
```

Les configurations `*_seed123.json` et `*_seed2026.json` reproduisent les réplications. Chaque `summary.json` conserve la configuration résolue, les versions, le matériel, la commande, le commit, l'état Git et l'empreinte SHA-256 du code Python.

### Évaluation du checkpoint verrouillé

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m cifar10_project.evaluate artifacts\final_model.pt --device cuda
.\.venv\Scripts\python.exe scripts\benchmark_checkpoint.py artifacts\final_model.pt --device cuda --batches 1 128
```

### Figures et tableaux

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m cifar10_project.cli stats
.\.venv\Scripts\python.exe -m cifar10_project.cli figures --dataset
.\.venv\Scripts\python.exe -m cifar10_project.cli figures --checkpoint artifacts\final_model.pt --metrics results\metrics\complete_cuda_wide_resnet_test.json
.\.venv\Scripts\python.exe -m cifar10_project.cli compare --prefix complete_cuda_
```

Cette dernière commande reconstruit le tableau de tous les runs CUDA, avec une ligne par graine. La synthèse finale agrégée et sa règle de sélection sont conservées dans `results/comparisons/final_primary_comparison.csv` et `results/metrics/multiseed_summary.json`.

### Reconstruire le rapport PDF

```powershell
.\.venv\Scripts\python.exe scripts\build_report_pdf.py
```

Le script lit [report/rapport.md](report/rapport.md), utilise les figures de `results/figures/` et écrit **`report/rapport_cifar10.pdf`**. Les dépendances PDF figurent dans `requirements.txt` et `requirements-lock.txt`.

## Organisation

```text
deep-learning/
├── README.md                 # Point d'entrée de la remise
├── report/                   # Rapport PDF final et source Markdown
├── src/cifar10_project/      # Code Python du projet
├── configs/                  # Hyperparamètres des entraînements
├── checkpoints/              # history.csv et summary.json de chaque run
├── results/
│   ├── comparisons/          # Tableaux CSV et graphiques comparatifs
│   ├── metrics/              # Métriques JSON et matrice de confusion brute
│   └── figures/              # Courbes, confusion, exemples et scores par classe
├── artifacts/                # Modèle final évaluable et manifeste
├── scripts/                  # Vérification, export, benchmark et génération PDF
├── tests/                    # Tests automatisés
├── docs/                     # Audit technique complémentaire
└── requirements*.txt         # Dépendances portables et environnement reproduit
```

### À quoi sert chaque partie du code ?

| Fichier | Rôle |
|---|---|
| [dataset.py](src/cifar10_project/dataset.py) | Charge CIFAR-10, sépare entraînement et validation, applique normalisation et augmentations, construit les chargeurs de données et calcule les statistiques. |
| [models.py](src/cifar10_project/models.py) | Définit le MLP, le CNN personnalisé, ResNet-18 adapté CIFAR et WideResNet-16-4. |
| [train.py](src/cifar10_project/train.py) | Entraîne les réseaux, calcule les métriques à chaque époque, sélectionne le point de contrôle sur la perte de validation et sauvegarde historique et résumé. |
| [experiments.py](src/cifar10_project/experiments.py) | Lance les phases exploration, complète et optimisation avec leurs configurations communes. |
| [evaluate.py](src/cifar10_project/evaluate.py) | Évalue un modèle retenu sur le test : accuracy, métriques par classe, F1 macro, confusion et calibration. |
| [visualization.py](src/cifar10_project/visualization.py) | Produit les courbes d'apprentissage, tableaux comparatifs, matrices de confusion et exemples de prédictions. |
| [cli.py](src/cifar10_project/cli.py) | Expose les commandes `stats`, `figures` et `compare`. |
| [utils.py](src/cifar10_project/utils.py) | Gère graines, choix du matériel, sauvegardes JSON et traçabilité de l'environnement et du code. |

### Configurations et scripts

Les fichiers JSON de [configs/](configs/) définissent les hyperparamètres sans modifier le code. `cnn_complete_cuda.json`, `resnet18_complete_ref.json` et `wide_resnet_complete_cuda.json` correspondent aux runs principaux. Les suffixes `seed123` et `seed2026` identifient les réplications ; `optim_subset_*` identifie les essais courts de régularisation.

| Script | Utilisation |
|---|---|
| [smoke_test.py](scripts/smoke_test.py) | Vérifie les quatre architectures sur des données synthétiques, sans télécharger CIFAR-10. |
| [benchmark_checkpoint.py](scripts/benchmark_checkpoint.py) | Mesure l'inférence du modèle final pour plusieurs tailles de lot. |
| [export_final_checkpoint.py](scripts/export_final_checkpoint.py) | Exporte les poids finaux sans les états d'optimiseur et produit leur manifeste SHA-256. |
| [build_report_pdf.py](scripts/build_report_pdf.py) | Convertit le rapport Markdown et ses figures en PDF. |
| [finish_resnet18.py](scripts/finish_resnet18.py) | Script historique de reprise du ResNet MPS interrompu ; il ne fait pas partie du benchmark final CUDA. |

[tests/test_core.py](tests/test_core.py) contient les tests automatisés ; [tests/conftest.py](tests/conftest.py) prépare leur environnement. L'[audit technique](docs/AUDIT_DEEP_LEARNING.md) est un document complémentaire, distinct du rapport remis.

### Fichiers conservés dans Git

Le dépôt inclut le rapport PDF, les configurations, les résultats, les historiques CSV/JSON et les poids du modèle final. Les données CIFAR-10, environnements Python, caches, fichiers TensorBoard et poids intermédiaires sont ignorés par `.gitignore`. Les deux journaux MPS historiques restent conservés pour documenter la reprise ancienne. Le téléchargement des données est automatique au premier lancement.

## Limites

Mon étude reste limitée à CIFAR-10, trois seeds pour les deux finalistes, 30 époques et un seul type de GPU. `deterministic=false` conserve les graines mais autorise certains algorithmes CUDA non déterministes ; l'étude multi-seeds quantifie partiellement cette variabilité. Les exemples qualitatifs sont les premiers exemples corrects ou incorrects rencontrés dans l'ordre du test, sans sélection esthétique. L'ECE de 2,65 % et une confiance moyenne de 95,76 % indiquent une légère surconfiance ; aucune calibration post-hoc n'a été ajustée sur le test.

## Usage de l'intelligence artificielle

J'ai utilisé OpenAI Codex comme assistant pour auditer le dépôt, améliorer la robustesse du code, rédiger et interpréter les artefacts expérimentaux. Je l'ai laissé suivre les journaux pendant l'exécution des époques afin de relever les tendances et incidents éventuels. Les métriques du rapport proviennent exclusivement des JSON/CSV et checkpoints produits par le projet ; je les ai vérifiées par des tests et contrôles de cohérence. Le nom exact du modèle OpenAI doit être repris depuis l'interface utilisée au moment de la remise si l'établissement l'exige.
