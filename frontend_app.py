#Importing the tools we use

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# Configuration

MODEL_PATH = "mura_resnet18.pth"
DEVICE = torch.device("cpu") # Use CPU for inference on frontend usually


# Loading the Model

@st.cache_resource
def load_model():
    """
    Loads the ResNet-18 model structure and weights.
    Cached to prevent reloading on every interaction.
    """
    # Define structure

    model = models.resnet18(pretrained=False) # No need to download imagenet weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    # Load weights(Safety Checks)

    if not os.path.exists(MODEL_PATH):
       return None
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None
# Processing the Image
def preprocess_image(image):
    """
    Prepares uploaded image for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Ensure RGB

    image = image.convert("RGB")
    return transform(image).unsqueeze(0) # Add batch dimension


# Streamlit UI

st.set_page_config(page_title="MURA X-Ray Classifier", page_icon="🩻")

st.title("🩻 MURA X-Ray Abnormality Detector")
st.markdown("""
This tool uses a Deep Learning model (ResNet-18) to classify musculoskeletal X-rays 
as **Normal** or **Abnormal**. **Note:** Always try to upload a horizontal images for better Accuracy
""")

# Sidebar info(Sets the browser tab title and the main header text on the page.)

st.sidebar.header("Status")

# Load Model

model = load_model()

if model is None:
    st.error("⚠️ Model file not found!")
    st.warning(f"Please run `backend_train.py` first to generate `{MODEL_PATH}`.")
    st.sidebar.error("Model Offline")
else:
    st.sidebar.success("Model Loaded")
    
    # File Uploader

    uploaded_file = st.file_uploader("Upload an X-Ray Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display Image

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        
        # Predict Button

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                input_tensor = preprocess_image(image)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    prediction = "Abnormal" if prob > 0.5 else "Normal"
                    confidence = prob if prob > 0.5 else 1 - prob
                
                # Display Results

                st.divider()
                if prediction == "Abnormal":
                    st.error(f"**Prediction: {prediction}**")
                else:
                    st.success(f"**Prediction: {prediction}**")
                
                st.info(f"Confidence Score: {confidence*100:.2f}%")
                st.progress(int(confidence * 100))