# 🧠 CentraleSupelec AI Project 1: Machine Learning Foundations

An introductory machine learning project focused on building and training neural networks from the ground up using **PyTorch**. 

This repository tracks the progression from fundamental algorithms to modern architectures, including Perceptrons, Convolutional Neural Networks (CNNs) for image classification, Recurrent Neural Networks (RNNs) for language identification, and a custom Attention mechanism for text generation.

---

## ✨ Core Features

* **Custom Neural Architectures:** Hand-built implementations of linear layers, non-linear activations (ReLU), and multi-layer perceptrons (MLPs).
* **Non-linear Regression:** Neural network approximation of the $\sin(x)$ function over a continuous domain.
* **Computer Vision (CNNs):** Manual implementation of 2D matrix convolutions and a spatial-aware model to classify handwritten digits from the MNIST dataset.
* **Sequence Modeling (RNNs):** A Recurrent Neural Network designed to process variable-length strings character-by-character to identify the language of a given word.
* **Generative AI (Transformers):** Extra credit implementation of Scaled Dot-Product Attention to power a Character-level GPT model trained on Shakespeare.
* **Automated Benchmarking:** Integrated autograder to test logical redundancy, mathematical correctness, and target accuracy thresholds (e.g., >97% for MNIST).

---

## 📂 Repository Architecture

```text
cs-ai-project1/
├── models.py                # Core logic: Perceptrons, CNNs, RNNs, and Attention blocks
├── gpt_model.py             # Transformer block and GPT architecture assembly
├── autograder.py            # Primary test suite for technical correctness
├── submission_autograder.py # Final evaluation and zip generation script
├── chargpt.py               # Training script for the Character-GPT model
├── backend.py               # Support code for ML tasks and PyTorch DataLoaders
└── data/                    # Datasets for MNIST and Language Identification
```

---

## 🛠️ Installation & Setup

It is highly recommended to use a Conda environment for this project to prevent library conflicts. 

**1. Clone the repository**
```bash
git clone https://github.com/Saadd77/cs-ai-project1.git
cd cs-ai-project1
```

**2. Create and activate a Conda environment**
```bash
conda create -n ai-course python=3.9
conda activate ai-course
```

**3. Install dependencies**
Install PyTorch (CPU version recommended for maximum stability during this project) along with NumPy and Matplotlib.
*Note: If you encounter array errors, ensure NumPy is downgraded to `< 2.0.0` (e.g., `1.24.3`).*
```bash
pip install numpy==1.24.3 matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**4. Verify the installation**
Run the dependency check. If successful, a window with a spinning line segment will appear.
```bash
python autograder.py --check-dependencies
```

---

## 🚀 Usage & Testing

The project is heavily reliant on the `autograder.py` script to validate the correctness of the models against training and validation datasets.

### Running Specific Tasks

You can test individual implementations by passing the question flag to the autograder:

**Q1: Binary Perceptron**
```bash
python autograder.py -q q1
```

**Q2 & Q3: Non-linear Regression & Digit Classification**
```bash
python autograder.py -q q2
python autograder.py -q q3
```

**Q4 & Q5: Language Identification (RNN) & Convolutions (CNN)**
```bash
python autograder.py -q q4
python autograder.py -q q5
```

### Character-GPT Training (Extra Credit)

To train the generative transformer model on the provided text file (Shakespeare), run:
```bash
python chargpt.py
```

### Final Submission

To run the complete test suite and package your code for Gradescope/Edunao submission:
```bash
python submission_autograder.py
```
This will generate a `submission-p1.zip` file containing your local evaluation results and source code.
