# 🩻 MURA X-Ray Abnormality Detector

A Deep Learning-based project for detecting abnormalities in musculoskeletal X-ray images using the **MURA (Musculoskeletal RAdiographs)** dataset and a **ResNet-18** Convolutional Neural Network (CNN).  
The system classifies images as **Normal** or **Abnormal**, and provides a Streamlit-based UI for easy interaction.

---

## 🌐 Live Demo

Try the deployed app here:  
👉 **https://murax-rayabnormalitydetector-gfwpuvcgyeccknfbfgjunf.streamlit.app/**

---

## 📁 Project Structure

- **`backend_train.py`**  
  Handles:
  - Automatic dataset downloading via **KaggleHub**
  - Data preprocessing and loading
  - ResNet-18 model definition (binary classifier)
  - Training loop & validation
  - Saving model weights (`mura_resnet18.pth`)
  - Generating training curves (`training_results.png`)

- **`frontend_app.py`**  
  A web UI built with **Streamlit**, allowing users to:
  - Upload X-ray images (PNG, JPG, JPEG)
  - Preview uploaded images
  - Receive predictions (Normal / Abnormal) with confidence scores

- **`requirements.txt`**  
  All Python dependencies for training and running the app.

- **`mura_resnet18.pth`**  
  Trained model weights (generated after training).

- **`training_results.png`**  
  Accuracy and loss plots (generated after training).

---

## ⚙️ Prerequisites

- Python **3.8+**
- Internet connection (dataset downloads via KaggleHub)

---

## 🚀 Installation

### 1. Clone or Download the Repository
```bash
git clone https://github.com/Mayank251125/MURA_X-Ray_Abnormality_Detector
cd <https://github.com/Mayank251125/MURA_X-Ray_Abnormality_Detector/blob/main/frontend_app.py>
## 🚀 Installation

### 2. Install Dependencies
```bash
pip install -r requirements.txt

---

```markdown
## 🛠️ Training the Model

If you do not already have **mura_resnet18.pth**, run the backend training script:

```bash
python backend_train.py

---

```markdown
## ▶️ Running the Frontend App

Once the model is trained, start the Streamlit app:

```bash
streamlit run frontend_app.py

---

```markdown
## 📊 Output Files

| File | Description |
|------|-------------|
| `mura_resnet18.pth` | Trained model weights |
| `training_results.png` | Accuracy & Loss plots |

