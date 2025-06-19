# 🧠 Deep Learning Applications Labs

This repository contains a series of hands-on labs exploring various deep learning paradigms, including supervised learning, reinforcement learning, and transformers. It was developed and tested locally on a high-performance workstation to ensure smooth execution and compatibility across environments.

---

## 📌 Project Summary

The labs are structured as independent Jupyter notebooks and cover the following topics:

### Lab 1 – **MLP & CNNs**
- Train MLPs and CNNs on MNIST and CIFAR-10
- Experiment with residual connections and knowledge distillation

### Lab 2 – **Reinforcement Learning**
- Implement REINFORCE and Deep Q-Network (DQN) agents
- Use OpenAI’s `gymnasium` environments for training and evaluation

### Lab 3 – **Transformers**
- Perform sentiment analysis and feature extraction with Hugging Face Transformers (DistilBERT)
- Apply Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA

---

## 🧪 System Configuration (Local Development)

All experiments were run and validated locally using the following setup:

- **GPU:** NVIDIA RTX 4070 Ti Super (16 GB VRAM)  
- **CPU:** AMD Ryzen 7 7800X3D  
- **RAM:** 32 GB DDR5 (Corsair Vengeance)  
- **Storage:** NVMe SSD Kingston

This configuration allows for fast model training and smooth interaction with large transformer models.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/marcopibbes/DLA_labs
cd LabDLA
```

### 2. Set Up the Python Environment

> ⚠️ **Note:** Due to compatibility issues between libraries (especially in Lab 2), you may need separate environments.

#### Using Conda (Recommended)

```bash

conda create -n dla_lab python=3.12.8 -y
conda activate dla_lab
conda install --file requirements.txt -c conda-forge -y


```

---

## 📁 Repository Structure

```text
.
├── Lab1-CNNs.ipynb           # Supervised learning (MLPs, CNNs)
├── Lab2-DRL.ipynb            # Reinforcement learning agents
├── Lab3-Transformers.ipynb   # Transformers & PEFT
├── BaseTrainingPipeline.py   # Core trainer class
├── SLTrainingPipeline.py     # Supervised training pipeline
├── RLTrainingPipeline.py     # REINFORCE trainer
├── QLTrainingPipeline.py     # DQN trainer
├── MLP.py                    # MLP and ResMLP definitions
├── CNN.py                    # CNN and ResCNN definitions
├── data/                     # Dataset storage (auto-downloaded)
├── checkpoints/              # Saved models
├── logs/                     # TensorBoard logs
└── requirements.txt          # Dependency list
```

---

## 📓 Running the Labs

1. Activate the appropriate Python environment.
2. Launch Jupyter:

```bash
jupyter lab  # or jupyter notebook
```

3. Open the desired notebook (`Lab1-CNNs.ipynb`, `Lab2-DRL.ipynb`, or `Lab3-Transformers.ipynb`) and start running the cells.

> 💡 **Tip:**  
> If no checkpoints are available, make sure to comment out `pipeline.load()` and use `pipeline.train()` instead.

Each notebook also defines a `clear()` function to clean temporary data (checkpoints, logs, datasets).

---

## 📊 TensorBoard Support

To visualize logs and monitor training:

```bash
tensorboard --logdir=./logs
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 🔍 Dependencies

All required packages are listed in `requirements.txt`, including:

- `torch`, `torchvision`, `transformers`, `tensorboard`
- `gymnasium`, `datasets`, `scikit-learn`, `matplotlib`, `pandas`
- Utilities for training, evaluation, and visualization

Some dependencies are version-pinned for compatibility and were installed via `conda-forge`.

---

## 🧩 Notes

- Expect first runs to download datasets and pretrained weights.
- Some visualizations (e.g., Lab 2) may require `pygame`.
- Tested across Python 3.9 and 3.12.8 environments.
