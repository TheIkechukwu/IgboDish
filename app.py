# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import wikipediaapi

# Custom FastAI components
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(size)
        self.maxpool = nn.AdaptiveMaxPool2d(size)
    
    def forward(self, x):
        return torch.cat([self.maxpool(x), self.avgpool(x)], dim=1)

@st.cache_resource
def load_model():
    # 1. ResNet18 body without final layers
    body = models.resnet18(pretrained=False)
    body = nn.Sequential(*list(body.children())[:-2])
    
    # 2. FastAI-style head with custom pooling
    head = nn.Sequential(
        AdaptiveConcatPool2d(1),
        nn.Flatten(),
        nn.BatchNorm1d(1024),  # 512*2 from concat pooling
        nn.Dropout(0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 6)       # Number of classes
    )
    
    model = nn.Sequential(body, head)
    
    # 3. Load trained weights
    model.load_state_dict(
        torch.load('igbo_dish_model_weights.pth', map_location='cpu'),
        strict=True
    )
    model.eval()
    return model

# Load class labels
with open('classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Wikipedia client setup
wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="IgboDishClassifier/1.0"
)

# Image transformations
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

# Enhanced Wikipedia info fetcher
def get_dish_info(dish_name):
    variants = [f"{dish_name} Soup", f"{dish_name} Stew", dish_name]
    info = {'history': '', 'ingredients': '', 'preparation': ''}
    
    for title in variants:
        page = wiki.page(title)
        if page.exists():
            sections = {
                'history': ['History', 'Origin'],
                'ingredients': ['Ingredients', 'Components'],
                'preparation': ['Preparation', 'Recipe']
            }
            for category in sections:
                for section in sections[category]:
                    if not info[category]:
                        content = page.section_by_title(section)
                        if content:
                            info[category] = content.text
            break
    return info

# Streamlit UI
def main():
    st.title("🍲 Igbo Cuisine Classifier")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload food image", 
            type=["jpg", "jpeg", "png"],
            help="Clear photo of prepared dish works best"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_container_width=True)
            
            with st.spinner("Analyzing culinary features..."):
                tensor = image_transform(img)
                probs = predict(tensor)
                pred_class = class_labels[np.argmax(probs)]
                confidence = np.max(probs)

    if uploaded_file:
        with col2:
            st.subheader(f"**Identification**: {pred_class}")
            st.metric("Confidence Level", f"{confidence*100:.1f}%")
            
            info = get_dish_info(pred_class)
            
            with st.expander("🌍 Cultural Significance", expanded=True):
                st.write(info['history'] or "Cultural history documentation in progress")
            
            with st.expander("🛒 Key Ingredients"):
                st.write(info['ingredients'] or "Typical ingredients being researched")
            
            with st.expander("👩🍳 Traditional Preparation"):
                st.write(info['preparation'] or "Preparation methods coming soon")
            
            st.markdown("---")
            st.write("**Explore More**")
            st.page_link("https://en.wikipedia.org/wiki/Igbo_cuisine",
                         label="Igbo Culinary Traditions on Wikipedia")

    # Example dishes gallery
    st.markdown("---")
    st.subheader("Common Igbo Dishes")
    cols = st.columns(6)
    for col, dish in zip(cols, class_labels):
        with col:
            try:
                st.image(
                    f"examples/{dish}.jpg",
                    caption=dish,
                    use_container_width=True,
                    output_format="JPEG"
                )
            except FileNotFoundError:
                st.warning(f"Example image missing for {dish}")

if __name__ == "__main__":
    main()