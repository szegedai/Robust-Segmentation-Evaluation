# Robust-Segmentation-Evaluation

**Evaluating the Adversarial Robustness of Semantic Segmentation: Trying Harder Pays Off**\
*Levente Halmosi*, *Bálint Mohos*, *Márk Jelasity*\
ECCV 2024\
[[ArXiv](https://arxiv.org/abs/2407.09150)]

![Description](illustrations/ships.png "Result of PAdam-Cos attack on an example PASCAL-VOC image for various
PSPNet models and SEA-AT-Small. Top row: perturbed images; bottom row: predicted
mask on the perturbed image.")

Robust-Segmentation-Evaluation introduces a reliable framework for evaluating the adversarial robustness of segmentation models by leveraging a set of diverse attacks:

1. **No-Attack:** Evaluation without any adversarial perturbation.
2. **PAdam-CE:** A novel, step size-free variant of the Projected Adam (PAdam) algorithm optimized using the Cross-Entropy (CE) loss function.
3. **PAdam-Cos:** An innovative, step size-free adaptation of PAdam based on Cosine distance.
4. **APGD-JSD(SEA):** APGD that utilizes the Jensen-Shannon Divergence (JSD) for optimization.
5. **APGD-MCE(SEA):** APGD optimized using the Masked Cross-Entropy (MCE) loss function.
6. **APGD-MSL(SEA):** APGD utilizing the Mean Squared Logarithmic (MSL) error for robust evaluation.
7. **APGD-BCE(SEA):** APGD based on the Binary Cross-Entropy (BCE) loss.
8. **ALMAProx:** An advanced attack method employing Proximal gradient techniques for adversarial perturbation.
9. **DAG-0.001:** A Directed Adversarial Gradient (DAG) attack with a perturbation scale of 0.001.
10. **DAG-0.003:** DAG attack with a perturbation scale of 0.003.
11. **PDPGD:** A variant of the Projected Dual PGD (PDPGD) algorithm for enhanced robustness assessment.

(SEA stands for Semantic Ensemble Attack, from the works of Croce et al.)

The key advantage of Robust-Segmentation-Evaluation lies in its pre-determined hyperparameters, eliminating the need for tuning and simplifying the evaluation process for any segmentation model.

## Installation

```
pip install git+https://github.com/szegedai/Robust-Segmentation-Evaluation
```

## How to use Robust-Segmentation-Evaluation

The following simple example demonstrates the use of this package on a simple evaluation task.
```python
# Import everything that is needed for the evaluation.
from rse import composite_attack, build_attacks, ATTACKS, evaluate
from rse.metrics import NmIoU, Acc
...

# Create a model and a dataloader for the evaluation.
model = ...
data_loader = ...

# Define metrics.
iou = NmIoU('multiclass', num_classes=5)
acc = Acc('multiclass', num_classes=5)

# Build callable attacks from predefined list of attack names.
# Feel free to add any custom attacks to this list.
attacks = build_attacks(ATTACKS, 5, 8 / 255)

# Create a composite attack that is easily callable and can return the
# best perturbed images produced by the list of attacks.
comp_attack = partial(
  composite_attack,
  attacks=attacks,
  inv_score_fn=iou
)

# Run the evaluation and get the results for each metric.
results = evaluate(model, data_loader, comp_attack, [iou, acc])
```

## References
The attacks and methodologies are inspired and adapted from existing works on adversarial robustness, specifically tailored for semantic segmentation tasks. Key references include but not limited to:

- **Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models** [[ArXiv](https://arxiv.org/abs/2306.12941)] ​​[[GitHub](https://github.com/nmndeep/robust-segmentation)]​\
  by *Francesco Croce, Naman D Singh and Matthias Hein*​
- **Proximal Splitting Adversarial Attacks for Semantic Segmentation** [[ArXiv](https://arxiv.org/abs/2206.07179)]​ [[GitHub](https://github.com/jeromerony/alma_prox_segmentation)]​\
  by *Jérôme Rony, Jean-Christophe Pesquet and Ismail Ben Ayed*
- **PDPGD: Primal-Dual Proximal Gradient Descent Adversarial Attack** [[ArXiv](https://arxiv.org/abs/2106.01538)] [[GitHub](https://github.com/aam-at/cpgd)]\
  by *Alexander Matyasko and Lap-Pui Chau*

## Citation

```
@inproceedings{halmosi2024robustsegeval,
  title={Evaluating the Adversarial Robustness of Semantic Segmentation: Trying Harder Pays Off}, 
  author={Levente Halmosi and B{\'a}lint Mohos and M{\'a}rk Jelasity},
  year={2024},
  booktitle={ECCV}
}
```
