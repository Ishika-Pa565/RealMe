# RealMe: Deepfake Image Detection Web Application

RealMe is an AI-powered deepfake image detection web application built with **PyTorch**, **timm (EfficientNet-B0)**, and **Streamlit**. It provides users with an instant, easy-to-understand fakeness score and confidence rating for any uploaded face image, designed for fast inference on standard laptop CPUs without requiring expensive hardware.

---

## 🌟 Key Features

- **Transfer Learning with EfficientNet-B0**: Pretrained on ImageNet and fine-tuned for deepfake vs. real face classification with binary cross-entropy loss.
- **Fast CPU Inference**: Evaluates an image and returns a verdict in under 5 seconds.
- **In-Memory Privacy**: Uploaded images are strictly processed in-memory and never saved to disk.
- **Confidence Scoring**: Outputs a granular fakeness score (e.g. `85% Fake`), avoiding misleading binary black-box labels.
- **Intuitive Streamlit UI**: User-friendly web interface with color-coded alerts and confidence meters.

---

## 📁 Project Structure

```text
RealMe/
├── Code/
│   ├── dataset.py        # PyTorch Dataset, augmentation pipeline, and data loaders
│   ├── train.py          # Training loop with Cosine Annealing, ROC-AUC validation
│   ├── predict.py        # Standalone CPU prediction engine
│   └── app.py            # Streamlit web interface
├── models/               # Model weights directory (best.pth saved here)
├── requirements.txt      # Project dependencies
├── README.md             # Documentation and usage guide
├── PRD.md                # Product Requirements Document & viva guide
└── .gitignore            # Git exclusion rules
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10 – 3.12
- Git

### 2. Clone Repository & Setup Environment
```bash
git clone <your-repo-url>
cd RealMe

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Training on a full dataset (e.g., Kaggle *140k Real and Fake Faces*) is recommended on a GPU (Google Colab / Kaggle).

### Dataset Format
Organize your dataset under `data/`:
```text
data/real_vs_fake/real-vs-fake/
├── train/
│   ├── real/
│   └── fake/
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Running Training
```bash
# Full training on GPU (5-8 epochs recommended)
python Code/train.py --data data/real_vs_fake/real-vs-fake --epochs 8 --batch-size 32

# Quick local test / smoke test (500 samples, 1 epoch)
python Code/train.py --data data/real_vs_fake/real-vs-fake --epochs 1 --subset 500
```
The best checkpoint based on validation ROC-AUC will automatically be saved to `models/best.pth`.

---

## 🖥️ Running the Web App

Once you have `models/best.pth` in the `models/` folder:

```bash
streamlit run Code/app.py
```
Open `http://localhost:8501` in your browser. Upload any face image (JPG/PNG) to analyze whether it is authentic or manipulated.

---

## 🔍 Standalone Prediction CLI

You can also test predictions directly via command line:
```bash
python Code/predict.py path/to/image.jpg
```

---

## 📜 Credits & Acknowledgments

This project is an upgraded, modernized PyTorch implementation building upon the foundational concepts of the original reference project:
- **Starting Point Repository**: [Ramya-Mahi/DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI](https://github.com/Ramya-Mahi/DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI)
- **Dataset**: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) by xhlulu.
- **Backbone**: EfficientNet-B0 via `timm` (PyTorch Image Models).
