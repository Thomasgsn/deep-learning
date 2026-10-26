# Historiques des expériences

Chaque dossier de run contient :

- `history.csv` : métriques d'entraînement et de validation, taux d'apprentissage et durée par époque lorsqu'ils ont été conservés ;
- `summary.json` : paramètres, métriques du point de contrôle choisi sur la perte de validation, coût, configuration et provenance disponible.

Les poids intermédiaires `best.pt` restent locaux et sont ignorés par Git. Le **modèle final livré** est [../artifacts/final_model.pt](../artifacts/final_model.pt), avec son [manifeste](../artifacts/final_model.json).

## Benchmark final CUDA

| Dossier | Modèle et graine |
|---|---|
| [complete_cuda_cnn/](complete_cuda_cnn/) | CNN, 42 |
| [complete_cuda_resnet18_ref/](complete_cuda_resnet18_ref/) | ResNet-18, 42 |
| [complete_cuda_resnet18_seed123/](complete_cuda_resnet18_seed123/) | ResNet-18, 123 |
| [complete_cuda_resnet18_seed2026/](complete_cuda_resnet18_seed2026/) | ResNet-18, 2026 |
| [complete_cuda_wide_resnet/](complete_cuda_wide_resnet/) | WideResNet-16-4, 42 ; modèle final retenu |
| [complete_cuda_wide_resnet_seed123/](complete_cuda_wide_resnet_seed123/) | WideResNet-16-4, 123 |
| [complete_cuda_wide_resnet_seed2026/](complete_cuda_wide_resnet_seed2026/) | WideResNet-16-4, 2026 |

Ces expériences utilisent 45 000 images d'entraînement et 5 000 images de validation, pendant 30 époques. Le MLP n'a pas de run complet CUDA dans cette remise.

## Optimisation CUDA sur sous-ensemble

- [optim_subset_ref/](optim_subset_ref/) : référence SGD.
- [optim_subset_mixup/](optim_subset_mixup/) : MixUp seul.
- [optim_subset_label_smoothing/](optim_subset_label_smoothing/) : label smoothing seul.

Ces runs utilisent 4 000 images pendant dix époques. Ils sont distincts des entraînements complets.

## Expériences historiques MPS

- `exploration_mlp/`, `exploration_cnn/`, `exploration_resnet18/` et `exploration_wide_resnet/` : exploration sur 4 000 images et cinq époques.
- `complete_cnn/` et `complete_resnet18/` : anciens entraînements MPS sur le train complet.

Le ResNet MPS a été repris après interruption avec réinitialisation de l'optimiseur. Son temps total est **estimé** et certaines colonnes de l'historique reconstruit sont absentes. Son résumé signale ces limites. Il ne doit pas être confondu avec `complete_cuda_resnet18_ref/`.

Les dossiers conservent leurs noms d'origine pour préserver les chemins enregistrés dans les résumés et manifestes. Pour la comparaison finale, consulter [../results/README.md](../results/README.md).
