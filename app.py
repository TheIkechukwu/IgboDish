# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import wikipediaapi

# Set up page config
st.set_page_config(
    page_title="Igbo Dish Classifier",
    page_icon="🍲",
    layout="wide"
)

@st.cache_resource
def load_model():
    # 1. Recreate EXACT FastAI architecture
    body = models.resnet18(pretrained=False)
    body = nn.Sequential(*list(body.children())[:-2])  # Remove original head
    
    # 2. Add FastAI's default head
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 6)  # Direct to class count
    )
    
    model = nn.Sequential(body, head)
    
    # 3. Load weights with proper mapping
    model.load_state_dict(
        torch.load('igbo_dish_model_weights.pth', 
                  map_location='cpu'),
        strict=True
    )
    model.eval()
    return model

# Load class labels
with open('classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Initialize Wikipedia client
wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="IgboDishesClassifier/1.0"
)

# Define transforms (MUST match FastAI's)
def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(img_tensor):
    model = load_model()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.numpy()[0]

# Wikipedia info function
def get_dish_info(dish):
    page = wiki.page(f"{dish} soup")
    info = {'history': '', 'ingredients': '', 'preparation': ''}
    sections = {
        'history': ['History', 'Origin'],
        'ingredients': ['Ingredients'],
        'preparation': ['Preparation', 'Recipe']
    }
    for cat in sections:
        for sec in sections[cat]:
            section = page.section_by_title(sec)
            if section:
                info[cat] += section.text + "\n\n"
    return info

# Streamlit UI
st.title("🍲 Igbo Dish Classifier")
uploaded_file = st.file_uploader("Upload dish image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    tensor = image_transform(img)
    probs = predict(tensor)
    pred_class = class_labels[np.argmax(probs)]
    confidence = np.max(probs)
    
    st.subheader(f"Prediction: {pred_class} ({confidence*100:.1f}% confidence)")
    
    info = get_dish_info(pred_class)
    with st.expander("History"):
        st.write(info['history'] or "No history found")
    with st.expander("Ingredients"):
        st.write(info['ingredients'] or "Ingredients not listed")
    with st.expander("Preparation"):
        st.write(info['preparation'] or "Preparation steps unavailable")

st.markdown("---")
st.write("Sample dishes:")
cols = st.columns(6)
for col, dish in zip(cols, class_labels):
    with col:
        st.image(f"examples/{dish}.jpg", caption=dish, use_container_width=True)