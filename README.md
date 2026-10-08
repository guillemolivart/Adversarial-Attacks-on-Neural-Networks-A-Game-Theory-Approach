# Adversarial Attacks on Neural Networks: A Game Theory Approach

This repository contains the source code, results, and final documents for my Bachelor's Thesis in Mathematics, defended at the Facultat de Matemàtiques i Estadística (FME) of the Universitat Politècnica de Catalunya (UPC).

## Project Description

Moving beyond empirical "black box" engineering, this project analyzes the vulnerability of artificial neural networks to adversarial attacks through the rigorous mathematical lens of **Game Theory**. 

The problem is modeled as a two-player zero-sum game between an **Attacker** (generating perturbations or noise) and a **Defender** (the neural classifier). By computing the **Nash Equilibrium**, we aim to find mutually optimal mixed strategies and quantify the theoretical limits of model robustness. The entire framework is implemented in PyTorch.

## Repository Structure

The code and resources are organized as follows:

* **`src/`**: Contains the main source code for the project.
  * `nn.py`: Neural network architectures.
  * `generator.py`: Adversarial attack generation logic.
  * `game.py`: Definition and implementation of the payoff matrix and game environment.
  * `nash.py`: Algorithms for computing Nash Equilibria.
  * `utils.py`: Helper functions for data loading and metrics.
* **`notebooks/`**: Interactive Jupyter environments.
  * `display/`: Notebooks for results visualization (`accuracy_matrices.ipynb`, `attacked_images.ipynb`, `nash_equilibria.ipynb`).
  * `tests/`: Testing and validation of system modules (`game_test.ipynb`, `mnist_test.ipynb`, `show_samples.ipynb`).
* **`output/`**: Output directory (populated during execution).
  * Includes subdirectories for generated figures (`figures/`), attacked images (`images/`), payoff matrices (`matrices/`), trained models (`models/`), and theoretical results (`nash_equilibria/`).
  * **Thesis Documents**: Contains the full thesis report (`TFG_Olivart_Guillem.pdf`) and the presentation slides (`TFG_Pres_GuillemOlivart.pdf`).

## Installation and Setup

The project uses `pyproject.toml` for dependency management. To set up your local environment:

1. Clone the repository:
   ```bash
   git clone [https://github.com/guillemolivart/Adversarial-Attacks-on-Neural-Networks-A-Game-Theory-Approach.git](https://github.com/guillemolivart/Adversarial-Attacks-on-Neural-Networks-A-Game-Theory-Approach.git)
   cd Adversarial-Attacks-on-Neural-Networks-A-Game-Theory-Approach