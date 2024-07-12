# Robust-Segmentation-Evaluation

"Evaluating the Adversarial Robustness of Semantic Segmentation: Trying Harder Pays Off"\
*Levente Halmosi*, *Bálint Mohos*, *Márk Jelasity*\
ECCV 2024\

![Description](illustrations/ships.png "Result of PAdam-Cos attack on an example PASCAL-VOC image for various
PSPNet models and SEA-AT-Small. Top row: perturbed images; bottom row: predicted
mask on the perturbed image.")

Robust-Segmentation-Evaluation introduces a reliable framework for evaluating the adversarial robustness of segmentation models by leveraging a set of diverse attacks:

1. **Clean:** Evaluation without any adversarial perturbation.
2. **PAdam-CE:** A novel, step size-free variant of the Projected Adam (PAdam) algorithm optimized using the Cross-Entropy (CE) loss function.
3. **PAdam-Cos:** An innovative, step size-free adaptation of PAdam based on the Cosine similarity loss.
4. **SEA-JSD:** A Self-Ensemble Attack (SEA) that utilizes the Jensen-Shannon Divergence (JSD) for optimization.
5. **SEA-MCE:** SEA optimized using the Mean Cross-Entropy (MCE) loss function.
6. **SEA-MSL:** SEA utilizing the Mean Squared Logarithmic (MSL) error for robust evaluation.
7. **SEA-BCE:** SEA based on the Binary Cross-Entropy (BCE) loss.
8. **ALMAProx:** An advanced attack method employing Proximal gradient techniques for adversarial perturbation.
9. **DAG-0.001:** A Directed Adversarial Gradient (DAG) attack with a perturbation scale of 0.001.
10. **DAG-0.003:** DAG attack with a perturbation scale of 0.003.
11. **PDPGD:** A variant of the Projected Dual PGD (PDPGD) algorithm for enhanced robustness assessment.

The key advantage of Robust-Segmentation-Evaluation lies in its pre-determined hyperparameters, eliminating the need for tuning and simplifying the evaluation process for any segmentation model.

# How to use Robust-Segmentation-Evaluation

### Installation

```
pip install git+https://github.com/szegedai/Robust-Segmentation-Evaluation
```

### PyTorch Segmentation models

# References
The attacks and methodologies are inspired and adapted from existing works on adversarial robustness, specifically tailored for semantic segmentation tasks. Key references include:

Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models by Francesco Croce, Naman D Singh, and Matthias Hein​ (ar5iv)​​ (GitHub)​.
Various loss functions and their implications for effective adversarial attacks as discussed in the literature on semantic segmentation robustness​ (ar5iv)​​ (GitHub)​.
This evaluation suite is designed to provide a thorough and reliable assessment of model robustness, ensuring that the tested models can withstand diverse adversarial conditions.
