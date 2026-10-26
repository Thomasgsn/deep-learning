# Guide des résultats

Le rapport final est dans [../report/rapport_cifar10.pdf](../report/rapport_cifar10.pdf). Les résultats sont déjà calculés ; leur consultation ne nécessite pas de relancer les expériences.

## Résultats principaux

| Fichier | Contenu |
|---|---|
| [comparisons/final_primary_comparison.csv](comparisons/final_primary_comparison.csv) | Synthèse du benchmark final CUDA : paramètres, validation, temps, mémoire et test du seul modèle retenu. Moyennes sur trois graines pour ResNet-18 et WideResNet ; une graine pour le CNN. |
| [metrics/multiseed_summary.json](metrics/multiseed_summary.json) | Valeurs par graine, moyennes, écarts-types d'échantillon et règle de sélection de l'architecture. |
| [metrics/complete_cuda_wide_resnet_test.json](metrics/complete_cuda_wide_resnet_test.json) | Évaluation finale du WideResNet, graine 42 : scores globaux, scores par classe, matrice brute et provenance du modèle évalué. |
| [metrics/final_inference_benchmark.json](metrics/final_inference_benchmark.json) | Benchmark dédié d'inférence, tailles de lot, échauffement, dispersion et périmètre mesuré. |
| [comparisons/complete_cuda_comparison.csv](comparisons/complete_cuda_comparison.csv) | Résultats individuels de tous les entraînements complets CUDA, avec une ligne par run. |

Les noms de champs `best_validation_accuracy` des résumés désignent l'accuracy du point de contrôle choisi sur la **perte** de validation, pas nécessairement le maximum d'accuracy de l'historique. La synthèse multi-graines indique ensuite une sélection d'architecture selon l'accuracy moyenne de validation. Le test est réservé au modèle final.

## Figures à consulter

- [Comparaison finale](figures/final_primary_comparison.png).
- Courbes d'apprentissage : [CNN](figures/complete_cuda_cnn_learning.png), [ResNet-18](figures/complete_cuda_resnet18_ref_learning.png), [WideResNet-16-4](figures/complete_cuda_wide_resnet_learning.png).
- [Matrice de confusion test](figures/complete_cuda_wide_resnet_test_confusion.png) et [métriques par classe](figures/complete_cuda_wide_resnet_test_per_class.png).
- [Prédictions correctes](figures/correct.png) et [erreurs](figures/errors.png) : premiers exemples correspondants rencontrés dans l'ordre du test.
- [Exemples du jeu d'entraînement](figures/dataset_examples.png) et [répartition des classes](figures/class_distribution.png).

## Exploration et optimisation

- [comparisons/exploration_comparison.csv](comparisons/exploration_comparison.csv) et [metrics/exploration_runs.json](metrics/exploration_runs.json) : exploration historique **MPS**, 4 000 images, cinq époques, quatre modèles.
- [comparisons/optim_subset_comparison.csv](comparisons/optim_subset_comparison.csv) et [metrics/optimisation_runs.json](metrics/optimisation_runs.json) : optimisation **CUDA**, 4 000 images, dix époques, ResNet-18 de référence, MixUp seul et label smoothing seul.
- [metrics/dataset_statistics.json](metrics/dataset_statistics.json) : statistiques descriptives de CIFAR-10.

Les essais courts ne mesurent pas la performance de modèles convergés. Leurs scores ne sont pas directement comparables au benchmark sur 45 000 images et 30 époques. Les temps MPS et CUDA doivent également rester séparés. Le temps par image mesuré par lots ne correspond pas à la latence d'une requête isolée ; les périmètres modèle seul et bout en bout sont distincts.

Les historiques détaillés sont décrits dans [../checkpoints/README.md](../checkpoints/README.md). Le fichier `.npy` de confusion conserve la matrice brute également présente dans le JSON test.
