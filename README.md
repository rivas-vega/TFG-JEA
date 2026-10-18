# Mathematical Foundations of Energy-Based Models and Joint Embedding Architectures

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Document](https://img.shields.io/badge/Thesis-PDF%20Available-brightgreen)](main.pdf)

**Author:** Pedro Rivas Vega  
**Supervisor:** Prof. Eduardo Sánchez Karhunen  
**Institution:** Universidad de Sevilla — Grado en Matemáticas

---

## Overview

This repository hosts the complete monograph and source code for the Bachelor's Thesis (*Trabajo de Fin de Grado*):

> **"Mathematical Foundations of Energy-Based Models and Joint Embedding Architectures: From Contrastive Heuristics to Isotropic Latent Regularization"**

The monograph develops an end-to-end mathematical theory of self-supervised representation learning (SSL), tracking the evolution from classical energy-based models (EBMs) and contrastive learning to non-contrastive covariance regularizers (Barlow Twins, VICReg) and modern predictive world models (JEPA, V-JEPA). It culminates in the analysis of empirical characteristic function regularizers—specifically **SIGReg** (Balestriero & LeCun, 2025)—which provably achieves Barlow's redundancy reduction principle without explicit covariance matrix regularization by exploiting the **Cramér-Wold Theorem** and the differentiable **Epps-Pulley statistic**.

---

## Contents of the Monograph

1. **Chapter 1: Introduction and Foundations of Representation Learning**  
   Discriminative vs. generative paradigms, information-theoretic bottleneck, and the case for non-reconstructive latent-variable models.
2. **Chapter 2: Energy-Based Models and Compatibility Learning**  
   Energy surfaces, Jaynes' principle of maximum entropy, conditional Boltzmann-Gibbs measures, and the role of inverse temperature $\beta$.
3. **Chapter 3: Classical Siamese Networks and Dimensional Collapse**  
   Formal analysis of positive-pair contraction, spectral decomposition of weight dynamics, and representation collapse.
4. **Chapter 4: Contrastive Learning as the First Solution to Collapse**  
   Margin-based negative loss, InfoNCE, covering bounds on hyperspheres ($\mathbb{S}^{d-1}$), and the curse of dimensionality in negative sampling.
5. **Chapter 5: The Non-Contrastive Revolution: Decorrelation and Barlow's Principle**  
   Barlow's efficient coding hypothesis, mutual information, Jaynes' bridge, Barlow Twins cross-correlation, and VICReg variance-covariance criteria.
6. **Chapter 6: Predictive Architectures and Temporal Collapse: From I-JEPA to V-JEPA**  
   Non-contrastive learning in sequence domains, temporal world models, asymmetric stop-gradient and EMA target networks.
7. **Chapter 7: Probabilistic Simplification of Barlow's Principle**  
   Equivalence of isotropic Gaussianity and Barlow independence, Cramér-Wold device for dimensionality reduction, differentiable Epps-Pulley normality testing in frequency domain, and the SIGReg regularizer.
8. **Chapter 8: Conclusion and Future Directions**  
   Synthesis of the mathematical progression and prospective research directions in hierarchical world models.
- **Appendices:**  
  A. Foundations of Deep Neural Architectures.  
  B. Mathematical Analysis, Spectral Decompositions, and Measure Theory.

---

## Document Compilation

The manuscript is written in standard LaTeX using the `book` class with `newtx` (Times-compatible) typography.

### Prerequisites

- TeX Live / MacTeX / TinyTeX with `pdflatex` and `bibtex`.
- Python 3 with `numpy`, `matplotlib`, and `scipy` (if regenerating figure assets).

### Build Commands

To compile the monograph from source:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The resulting compiled document is [`main.pdf`](main.pdf).

### Regenerating Figures

To regenerate all mathematical diagrams and publication plots:

```bash
python generate_figures.py
```

---

## License

The monograph text, figures, and associated code are released under the [MIT License](LICENSE).
