# MURA X-Ray Abnormality Detector 🩻

This repository contains a Deep Learning project designed to detect abnormalities in musculoskeletal X-ray images. It utilizes the **MURA (Musculoskeletal RAdiographs)** dataset and a **ResNet-18** Convolutional Neural Network (CNN) to classify images as either **Normal** or **Abnormal**.

The project consists of a training backend script and a Streamlit-based frontend application for easy interaction.

## 📂 Project Structure

* **`backend_train.py`**: The core training script. It handles:
    * Automatic dataset downloading via `kagglehub`.
    * Data preprocessing and loading (MURA dataset structure).
    * Model definition (ResNet-18 binary classifier).
    * Training loop with validation.
    * Saving the trained model (`mura_resnet18.pth`) and plotting training results (`training_results.png`).
* **`frontend_app.py`**: A user-friendly web interface built with **Streamlit**. It allows users to:
    * Upload X-ray images (PNG, JPG, JPEG).
    * View the uploaded image.
    * Get real-time predictions (Normal/Abnormal) with confidence scores.
* **`requirements.txt`**: List of Python dependencies required to run the project.
* **`mura_resnet18.pth`**: (Generated after training) The saved model weights.
* **`training_results.png`**: (Generated after training) plots showing Accuracy and Loss over epochs.

## ⚙️ Prerequisites

* Python 3.8+
* Internet connection (for downloading the dataset via KaggleHub)

## 🚀 Installation

1.  **Clone the repository** (if applicable) or download the files to your local machine.

2.  **Install Dependencies**:
    Run the following command to install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Usage

### 1. Training the Model
Before running the app, you need to train the model or ensure you have the `mura_resnet18.pth` file.

Run the backend script:
```bash
python backend_train.py