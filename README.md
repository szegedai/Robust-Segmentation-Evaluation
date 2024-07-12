# Robust-Segmentation-Evaluation

**Authors:** Levente Halmosi, Bálint Mohos, Márk Jelasity
**Institution:** University of Szeged, Hungary; HUN-REN-SZTE Research Group on AI, Szeged, Hungary  
**Conference:** ECCV 2024  
**Paper:** ---

Robust-Segmentation-Evaluation introduces a reliable framework for evaluating the adversarial robustness of segmentation models by leveraging a set of diverse and hyperparameter-free attacks:

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
