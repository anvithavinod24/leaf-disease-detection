# scripts/leaf_disease_dashboard.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Load class names (same order as training)
CLASS_NAMES = [
    'Tomato__Bacterial_spot',
    'Tomato__Early_blight',
    'Tomato__Late_blight',
    'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot',
    'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__healthy'
]

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/tomato_resnet18.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit app UI
st.set_page_config(page_title="Tomato Leaf Disease Detector", layout="centered")
st.title("🌿 Tomato Leaf Disease Detection Dashboard")
st.write("Upload a tomato leaf image and get the predicted disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted.item()]

    st.success(f"**Predicted Class:** {predicted_class}")
