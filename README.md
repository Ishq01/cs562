Exploring the Privacy-Utility Trade-off in Multi-Hop Question Answering

CS 562: Advanced Topics in Security, Privacy, and Machine Learning
Final Project | Fall 2025 | University of Illinois Urbana-Champaign

AUTHORS
* Ishq Gupta
* Isha Agrawal
* Joshua Johns

--------------------------------------------------------------------------------

OVERVIEW
This repository contains the implementation and experimental code for our paper, "Exploring the Privacy-Utility Trade-off in Multi-Hop Question Answering Datasets."

We systematically evaluate the impact of Differentially Private Stochastic Gradient Descent (DP-SGD) on multi-hop reasoning tasks. While prior work has shown that large language models (LLMs) can "absorb" DP noise for simple classification tasks, our research demonstrates that reasoning-heavy tasks (like HotPotQA) are far more fragile.

We fine-tuned DeBERTa-v3-base using the Opacus library to quantify the "reasoning gap"—the substantial loss in Exact Match (EM) and F1 scores when formal privacy guarantees are applied.

KEY FINDINGS
* The Reasoning Gap: Multi-hop reasoning is significantly more sensitive to gradient clipping and noise than standard NLP tasks. We observed a ~30% drop in F1 and ~45% drop in EM even under moderate privacy settings (epsilon approx. 3.25).
* The "Goldfish Effect": Under DP-SGD, models struggle to retain long-range dependencies across multiple documents, leading to a decoupling of training loss and validation utility.
* Hyperparameter Sensitivity: There is a narrow "usable region" for hyperparameters. We found that a clipping norm of C=1.0 consistently outperformed C=0.5, as aggressive clipping removes gradient signals essential for reasoning.

--------------------------------------------------------------------------------

INSTALLATION & SETUP

Prerequisites:
* Python 3.8+
* PyTorch 1.13+ (with CUDA support recommended)

1. Clone the Repository
   git clone https://github.com/Ishq01/cs562.git
   cd cs562

2. Create a Virtual Environment (Recommended)
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   source venv/bin/activate

3. Install Dependencies
   We rely on Hugging Face transformers, datasets, and opacus for privacy accounting.
   pip install -r requirements.txt

   (If you do not have a requirements file yet, run the command below)
   pip install torch transformers datasets opacus accelerate scikit-learn matplotlib

--------------------------------------------------------------------------------

PROJECT STRUCTURE

data/                   # Scripts for downloading and preprocessing HotPotQA
src/
  train_baseline.py     # Standard non-private training (DeBERTa-v3)
  train_dp.py           # DP-SGD training using Opacus
  evaluate.py           # Evaluation scripts for EM/F1 metrics
  utils.py              # Helper functions for tokenization & metrics
notebooks/              # Jupyter notebooks for visualization and analysis
results/                # Logs, plots, and saved models
README.md               # Project documentation
requirements.txt        # Python dependencies

--------------------------------------------------------------------------------

USAGE

1. Data Preparation
   We use the HotPotQA (Distractor) dataset from Hugging Face. The scripts will automatically download and cache the dataset. We filter for the extractive subset (removing yes/no questions) to focus on span prediction.

2. Training the Non-Private Baseline
   To establish an upper bound for performance:
   python src/train_baseline.py --model_name "microsoft/deberta-v3-base" --batch_size 8 --epochs 4 --lr 3e-5

3. Training with DP-SGD
   To train with Differential Privacy, specify the noise multiplier (--sigma) and clipping norm (--max_grad_norm). The script uses the RDP accountant to track the privacy budget (epsilon).

   Example: Moderate Privacy (sigma=1.0, C=1.0)
   python src/train_dp.py --model_name "microsoft/deberta-v3-base" --batch_size 8 --target_epsilon 3.0 --sigma 1.0 --max_grad_norm 1.0 --epochs 4

4. Evaluation
   Evaluate the model on the validation set to generate Exact Match (EM) and F1 scores.
   python src/evaluate.py --model_path ./results/checkpoint-final

--------------------------------------------------------------------------------

RESULTS VISUALIZATION

Detailed plots regarding the Privacy-Utility trade-off can be generated using the notebooks in the notebooks/ directory.

* notebooks/Analysis.ipynb: Generates the "Reasoning Gap" plots (EM/F1 vs. Epochs) and the Loss-Utility decoupling charts.

--------------------------------------------------------------------------------

PRIVACY & ETHICS
This project implements Differential Privacy to mathematically guarantee that the model does not memorize specific training examples. This is crucial for deploying QA systems in sensitive domains (e.g., healthcare) where training data must not be reconstructible from model weights. All experiments were conducted using public benchmark data (HotPotQA).

--------------------------------------------------------------------------------

REFERENCES
* DP-SGD: Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016).
* Opacus: https://opacus.ai/
* HotPotQA: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" (EMNLP 2018).
* Base Implementation: Adapted from https://github.com/huseyinatahaninan/Differentially-Private-Fine-tuning-of-Language-Models
