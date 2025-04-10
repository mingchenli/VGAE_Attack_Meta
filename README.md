
# Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction

## 🔍 Overview

This repository provides an demo of [**Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction**](https://arxiv.org/abs/2504.06492). The attack framework is designed to compromise the performance of VGAEs on link prediction by strategically perturbing the graph structure using meta-gradient techniques. Variational Graph Autoencoders are commonly used for representation learning on graphs. However, they are vulnerable to structural attacks. This paper explores how meta-learning can be used to optimize such attacks in a more efficient and generalized manner.

Key features:
- Meta-learning-based adjacent matrix attack algorithm
- Support for common benchmark datasets (e.g., Cora, Citeseer)

## 📁 Project Structure

```
VGAE_Attack_Meta/
│
├── AttackAdj.py        # Adversarial perturbations
├── MetaAttack.py       # Main class
├── train_meta.py       # Train and evaluate via meta-learning
├── model.py            # VGAE model definition
├── layers.py           # Graph neural network layers
├── optimizer.py        # Custom optimizer for meta-learning
├── utils.py            # Utility functions
└── data/               # Dataset loading and preprocessing
```

### 1. Installation

We recommend using a conda environment.

Requirements:
- `torch`
- `numpy`
- `scipy`
- `networkx`
- `scikit-learn`

### 2. Running the Meta Attack

## 🚀 Getting Started

Simple running example：
```bash
python MetaAttack.py --dataset cora --epochs 200
```

Arguments:
- `--dataset`: Name of dataset
- `--epochs`: Number of training epochs

## 📊 Results

You can evaluate the performance of VGAE before and after the attack using  AUC and AP. 

## 🧠 Reference

This VGAE implementation is inspired by the repository:
gae-pytorch. GitHub, from [https://github.com/zfjsail/gae-pytorch](https://github.com/zfjsail/gae-pytorch)

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
