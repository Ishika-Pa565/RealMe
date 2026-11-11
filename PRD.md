# RealMe: Detailed PRD and Build Spec for Antigravity

Put this file in the project root as `PRD.md`. Paste the Task prompt (section 1) into the Antigravity agent. The agent should re-read this file at the start of every phase.

---

## 1. Task prompt (copy this)

You are helping me finish RealMe, a deepfake image detection web app, in the current workspace.
Read `PRD.md` fully first. Work phase by phase (section 9). After each phase: run its acceptance check, show me the result, and wait for my approval before starting the next phase.

### Rules:
1. This workspace is a copy (clone) of an existing public GitHub project, kept as the starting point (section 3). Read the existing files in `Code/` first, then convert and upgrade them into RealMe as described here. Keep the same file names and the same simple flow: upload image, get result.
2. Use PyTorch (not TensorFlow). Replace the old TensorFlow/Keras code completely and remove unused TensorFlow imports.
3. Label convention everywhere: `fake = 1`, `real = 0`. Model output is one logit; `sigmoid(logit) = P(fake)`.
4. Windows 10/11, Python 3.10-3.12, `pathlib` for paths.
5. Do not download datasets or start long training without asking me. I train on Colab/Kaggle GPU.
6. Uploaded images are processed in memory only and never saved.
7. Keep code short, commented, and easy to explain in a viva. No unnecessary abstractions.
8. Do not add features outside the current phase. Phases 5 and 6 are optional and only start when I say so.
9. Create/update `.gitignore` so `venv/`, `venv_ref/`, `data/`, `models/*.pth`, and `__pycache__/` are never committed.
10. In `README.md`, credit the original GitHub project as the starting point.

---

## 2. Project history and decisions (context for the agent)

This is the full background so you do not repeat discussions or undo decisions.

| Topic | What happened / what was decided |
| :--- | :--- |
| **Student goal** | Major (final-year) project: deepfake image detection web app named RealMe. |
| **Starting point** | A public GitHub repo by Ramya-Mahi (`DeepFake-Detection-and-Prevention-A-Comprehensive-approach-using-AI`): MobileNetV2 (TensorFlow/Keras) image classifier, Streamlit UI, claimed 95% accuracy. Its `Code/` folder has `app.py`, `predict.py`, `train.py`. No requirements file and no trained model file are included. |
| **Repo review** | Strengths: simple UI, saved model, works as a demo. Weaknesses: images only, accuracy claim has no shown dataset split or metrics, demo used preselected images, "prevention" is in the title but not implemented. |
| **Student's synopsis** | RealMe: user uploads an image and sees a fakeness score like "85% Fake". Deep learning with a GAN at the core (Generator + Discriminator). Datasets: Kaggle "140k Real and Fake Faces" and FaceForensics++. Metrics: accuracy, precision, recall, F1, target about 95%. Streamlit, PyCharm/VS Code, Google Colab, Git/GitHub. Future scope: video, mobile app. |
| **Technical decision 1** | A GAN discriminator alone generalizes poorly to real deepfakes. The detector is a pretrained EfficientNet-B0 classifier. The GAN is optional later (Phase 6) and only as a source of extra fake training images. |
| **Technical decision 2** | PyTorch with `timm`, not TensorFlow. |
| **Technical decision 3** | Score meaning: "85% Fake" is the model's confidence P(fake), not the percentage of the image that is fake. |
| **Scope decision** | Core app first (upload, score, label). Extras (face detection, Grad-CAM, evaluation reports, cross-dataset test) are optional phases. Student explicitly asked for no extras in the first build. |
| **Environment** | Windows, VS Code, Python 3.12, TensorFlow was installed in `venv_ref` (old). New RealMe venv uses PyTorch. Training on Colab/Kaggle because the laptop is slow. |
| **Already written** | `Code/dataset.py` and `Code/train.py` (PyTorch), `Code/predict.py`, `Code/app.py` (all provided with this PRD and syntax-checked, not yet run on real data). |
| **Student language** | The student chats in Hinglish. Code, comments, and docs must be in English. |

---

## 3. How to treat the GitHub copy

- The workspace folder came from git clone of the reference repo. Its `Code/app.py`, `Code/predict.py`, `Code/train.py` are the starting files.
- Replace their contents with the RealMe PyTorch versions provided with this PRD, but keep the same file names so the project structure matches the original.
- The old git remote points to the original author's repo. Do not push there. Before the first push, ask me for my own repo URL and run: `git remote remove origin` then `git remote add origin <my-repo-url>`.
- Put a "Credits" line in `README.md` naming the original repository as the starting point, and keep any license file the original has.

---

## 4. Problem and goals

**Problem:** Deepfakes are easy to make and hard to spot. Existing detectors are accurate only on the data they were trained on, give no explanation, and are hard for normal users to run.

**Goals:**
1. Upload any face image in a browser and get a fakeness score and a Real/Fake label in under 5 seconds on a laptop CPU.
2. About 95% accuracy as a target on the in-domain test set. Report whatever is actually measured.
3. Simple enough for students, journalists, and small organizations to use.

**Non-goals (version 1):** Video, audio, mobile app, browser extension, legal-grade proof.

---

## 5. Users

General public, journalists, small organizations (KYC, HR, marketplaces), and the examiner/researcher who reads the report.

---

## 6. Functional requirements

| ID | Requirement | Phase |
| :--- | :--- | :--- |
| FR1 | Upload one JPG/PNG image (max 10 MB) | 3 |
| FR2 | Resize to 224x224 and normalize with ImageNet statistics | 2 |
| FR3 | Return P(fake) from the saved model | 2 |
| FR4 | Show score as "NN% Fake" and a label: Fake if P >= 0.5, else Real | 3 |
| FR5 | Show a note that the score is model confidence, not proof | 3 |
| FR6 | Friendly error if `models/best.pth` is missing | 3 |
| FR7 | Never save uploaded images to disk | 3 |
| FR8 | Metrics report: accuracy, precision, recall, F1, ROC-AUC, confusion matrix | 4 (optional) |
| FR9 | Face detection and crop before prediction | 5 (optional) |
| FR10 | Grad-CAM heatmap overlay | 5 (optional) |
| FR11 | Multi-image upload with results table | 5 (optional) |

---

## 7. Data and ML specification

**Dataset:** Kaggle "140k Real and Fake Faces" (xhlulu). After extracting into `data/`, the path is:
```text
data/real_vs_fake/real-vs-fake/
train/ real/ fake/
valid/ real/ fake/
test/  real/ fake/
```
If the extracted folder names differ, adapt the path and tell me.

**Model:** `timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)`.

**Training:** `BCEWithLogitsLoss`, `AdamW` (lr 1e-4, weight decay 1e-4), cosine schedule, batch size 32, 5-8 epochs. Best checkpoint chosen by validation ROC-AUC and saved to `models/best.pth`.

**Augmentation (training only):** Horizontal flip, rotation up to 10 degrees, brightness/contrast jitter, light Gaussian blur, random JPEG compression (uploads are usually compressed).

**Evaluation (optional Phase 4):** Accuracy, precision, recall, F1, ROC-AUC, confusion matrix on the test split. For a generalization claim, also test on frames from FaceForensics++ or Celeb-DF (never used in training) and report the drop honestly.

**Known limitation to state in the report:** Works best on face images and may fail on fakes from generators not seen in training.

---

## 8. Final file layout

```text
<project root>/
├── data/              # dataset (not committed)
├── Code/
│   ├── dataset.py     # provided: loaders and augmentation
│   ├── train.py       # provided: training + metrics
│   ├── predict.py     # provided: load model, predict P(fake)
│   └── app.py         # provided: Streamlit UI
├── models/            # best.pth (not committed)
├── requirements.txt
├── README.md
├── PRD.md
└── .gitignore
```

Scripts import each other directly, so always run from the project root:
```bash
python Code/train.py --data data/real_vs_fake/real-vs-fake --epochs 1 --subset 500
streamlit run Code/app.py
```

---

## 9. Phases, tasks, and acceptance checks

### Phase 1: Setup and baseline check
- Read the existing `Code/` files, then replace `train.py`, `predict.py`, `app.py` with the provided PyTorch versions and add `dataset.py`.
- Review all four files for real bugs; fix only real bugs.
- Create `requirements.txt` (pinned: torch, torchvision, timm, scikit-learn, tqdm, pillow, streamlit; plus opencv-python, facenet-pytorch, grad-cam only if Phase 5 starts).
- Create `.gitignore` and `README.md` (install, train, run, credits).
- **Accept when:** the quick-test command runs and prints TEST RESULTS on a subset. If the dataset is not downloaded yet, run a smoke test with a few generated dummy images instead.

### Phase 2: Prediction module
- Verify `Code/predict.py`: loads EfficientNet-B0 weights, returns float in [0, 1].
- **Accept when:** with a weights file (real or randomly initialized for testing) it returns a float in [0, 1] on CPU in under 5 seconds per image.

### Phase 3: Web app
- Verify `Code/app.py`: upload, score "NN% Fake", label, progress bar, caution note, missing-model message, model cached with `st.cache_resource`.
- **Accept when:** `streamlit run Code/app.py` starts, shows the missing-model message when no weights exist, and shows a result for an uploaded image when weights exist.

### Phase 4 (optional, only if I ask): Evaluation
- Create `Code/evaluate.py`: load weights, run on any folder with `real/` and `fake/` subfolders, print all metrics, save confusion-matrix and ROC PNGs to `reports/`.
- **Accept when:** runs on a tiny sample folder and writes both PNGs.

### Phase 5 (optional, only if I ask): Explainability and usability
- Face detection and crop using `facenet-pytorch` MTCNN; clear message if no face is found.
- Grad-CAM heatmap with `pytorch-grad-cam` on the last conv block.
- Multi-image upload with a table sorted by fakeness.
- **Accept when:** each feature works on three mixed test images.

### Phase 6 (optional, only if I ask): Advanced
- Second model for comparison (Xception or ViT via `timm`).
- GAN-generated fakes as extra training data.
- Invisible watermark embed and verify (the "prevention" idea).

---

## 10. Training on Colab (instructions for me, not for the agent to run)

1. Push the code to my own GitHub repo.
2. In Colab: set runtime to GPU, `git clone my repo`, `pip install -r requirements.txt`.
3. Download the Kaggle dataset into Colab (Kaggle API key) or mount Google Drive.
4. Run `python Code/train.py --data <dataset path> --epochs 8`.
5. Download `models/best.pth` and place it in the local `models/` folder.

---

## 11. Non-functional requirements

- Under 5 seconds per image on laptop CPU.
- Weights file about 16-20 MB (under 50 MB).
- Fixed seeds where possible; commands documented in `README.md`.
- No uploaded image stored; no personal data collected.

---

## 12. Risks

| Risk | Mitigation |
| :--- | :--- |
| Poor generalization to new deepfakes | Strong augmentation, honest cross-dataset test, stated limitation |
| Laptop too slow to train | Colab/Kaggle GPU; local tests on a 500-image subset |
| Large dataset download (about 4 GB) | Start early; develop with a subset |
| Wrong score harms someone | Caution note; score is never presented as proof |
| Project looks copied | Different framework and model; credit the source; be ready to explain every file |
| Version conflicts (old TensorFlow venv) | Use a fresh venv for RealMe; do not mix with `venv_ref` |

---

## 13. Timeline (10 weeks)

- Weeks 1-2: Environment, dataset download, Phase 1
- Weeks 3-4: Train baseline on Colab, Phases 2-3
- Week 5: Phase 4 evaluation
- Weeks 6-7: Phase 5 (face crop, heatmap, multi-upload)
- Week 8: Phase 6 stretch or buffer
- Week 9: Testing, threshold and bug fixes
- Week 10: Report, slides, demo video

---

## 14. Deliverables

Working code in my own GitHub repo, trained `best.pth`, `README.md` with credits, `requirements.txt`, project report (metrics, limitations, future scope), slides, and a demo video.

---

## 15. Viva preparation (what I must be able to explain)

- Why EfficientNet-B0 with transfer learning instead of training from scratch.
- Why the GAN is not the detector, and what a discriminator can and cannot do.
- What "85% Fake" means and why it is not proof.
- Why random JPEG compression is in the augmentation.
- What accuracy, precision, recall, F1, and ROC-AUC measure, and why 95% on one dataset does not guarantee real-world accuracy.
- Limitations and future scope: video, face-less images, new generators.
