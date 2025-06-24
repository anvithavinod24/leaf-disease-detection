# scripts/leaf_disease_dashboard.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 🍅 Class labels in training order
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

# 🔁 Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🧠 Load trained model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/tomato_resnet18.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 🖼️ Streamlit UI
st.set_page_config(page_title="Tomato Leaf Disease Detector", layout="centered")
st.title("🌿 Tomato Leaf Disease Detection Dashboard")
st.write("Upload a tomato leaf image and get the predicted disease with confidence scores.")

uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔎 Inference
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, 3)
        top_probs = top_probs[0].numpy()
        top_indices = top_indices[0].numpy()

    # ✅ Primary prediction
    pred_class = CLASS_NAMES[top_indices[0]]
    pred_conf = top_probs[0] * 100
    st.success(f"**Predicted Class:** {pred_class} ({pred_conf:.2f}%)")

    # 📊 Show top-3 predictions
    st.markdown("### 🔍 Top 3 Predictions")
    for i in range(3):
        cls = CLASS_NAMES[top_indices[i]]
        conf = top_probs[i] * 100
        st.write(f"- {cls}: **{conf:.2f}%**")
