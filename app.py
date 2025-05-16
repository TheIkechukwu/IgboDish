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
    # 1. EXACT FastAI ResNet18 architecture
    body = models.resnet18(pretrained=False)
    body = nn.Sequential(*list(body.children())[:-2])  # Keep convolutional base
    
    # 2. Recreate FastAI's custom head
    head = nn.Sequential(
        nn.AdaptiveConcatPool2d(1),  # Critical FastAI component
        nn.Flatten(),
        nn.BatchNorm1d(1024),        # From 512*2 (concat pool)
        nn.Dropout(0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 6)            # Final output layer
    )
    
    model = nn.Sequential(body, head)
    
    # 3. Load weights with exact architecture match
    state_dict = torch.load('igbo_dish_model_weights.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
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
    page = wiki.page(f"{dish} Soup")  # Try with capitalized soup
    info = {
        'history': '',
        'ingredients': '',
        'preparation': ''
    }
    
    # Section fallback system
    section_map = {
        'history': ['History', 'Origin', 'Background'],
        'ingredients': ['Ingredients', 'Components'],
        'preparation': ['Preparation', 'Cooking', 'Recipe']
    }
    
    for category in section_map:
        for section_name in section_map[category]:
            section = page.section_by_title(section_name)
            if section and not info[category]:
                info[category] = section.text
                break
                
    return info

# Streamlit UI
def main():
    st.title("🍲 Igbo Traditional Dish Classifier")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload dish image", 
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_container_width=True)
            
            with st.spinner("Analyzing ingredients and textures..."):
                tensor = image_transform(img)
                probs = predict(tensor)
                pred_class = class_labels[np.argmax(probs)]
                confidence = np.max(probs)
    
    if uploaded_file:
        with col2:
            st.subheader(f"**Prediction**: {pred_class}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            
            dish_info = get_dish_info(pred_class)
            
            with st.expander("📜 History & Cultural Significance", expanded=True):
                st.write(dish_info['history'] or "Historical information being gathered...")
            
            with st.expander("🥕 Key Ingredients"):
                st.write(dish_info['ingredients'] or "Ingredients list coming soon!")
            
            with st.expander("👩🍳 Traditional Preparation"):
                st.write(dish_info['preparation'] or "Preparation methods documentation in progress")
            
            st.markdown("---")
            st.write("**Explore Igbo Cuisine**")
            st.page_link("https://en.wikipedia.org/wiki/Igbo_cuisine", 
                        label="Wikipedia: Igbo Culinary Traditions")
    
    # Example images carousel
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
                st.error(f"Example image missing for {dish}")

if __name__ == "__main__":
    main()