# app.py
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
import numpy as np
import wikipediaapi

# Set up page config
st.set_page_config(
    page_title="Igbo Dish Classifier",
    page_icon="🍲",
    layout="wide"
)

# Define model architecture
class IgboDishClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.base(x)

# Load model function with caching
@st.cache_resource
def load_model():
    model = IgboDishClassifier()
    model.load_state_dict(
        torch.load('igbo_dish_model_weights.pth', map_location='cpu')
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

# Define transforms
def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(img_tensor):
    model = load_model()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.numpy()[0]

# Wikipedia information function
def get_dish_info(dish_name):
    page = wiki.page(f"{dish_name} soup")
    sections = {
        'history': ['History', 'Origin'],
        'ingredients': ['Ingredients'],
        'preparation': ['Preparation', 'Recipe', 'Method']
    }
    
    info = {'history': '', 'ingredients': '', 'preparation': ''}
    
    for section_type in sections:
        for section_title in sections[section_type]:
            section = page.section_by_title(section_title)
            if section:
                info[section_type] += section.text + "\n\n"
    
    return info

# Streamlit UI
st.title("🍲 Igbo Traditional Dish Classifier")
st.write("Upload an image of an Igbo dish to identify it and learn about its history, ingredients, and preparation!")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert and transform image
        img_tensor = image_transform(image)
        
        # Make prediction
        with st.spinner("Analyzing the dish..."):
            probabilities = predict(img_tensor)
            pred_class = class_labels[np.argmax(probabilities)]
            confidence = np.max(probabilities)

with col2:
    if uploaded_file:
        st.subheader(f"Prediction: {pred_class}")
        st.progress(float(confidence), 
                   text=f"Confidence: {confidence*100:.1f}%")
        
        # Get Wikipedia information
        dish_info = get_dish_info(pred_class)
        
        with st.expander("📜 History and Origin"):
            st.write(dish_info['history'] or "Historical information not available")
        
        with st.expander("🥕 Ingredients"):
            st.write(dish_info['ingredients'] or "Ingredients list not available")
        
        with st.expander("👩🍳 Preparation Method"):
            st.write(dish_info['preparation'] or "Preparation steps not available")
        
        st.markdown("---")
        st.write("**Did you know?**")
        st.write("Explore more about Igbo cuisine:")
        st.page_link("https://en.wikipedia.org/wiki/Igbo_cuisine", 
                    label="Wikipedia: Igbo Cuisine")
    else:
        st.write("Upload an image to get started!")

# Add sample images
st.subheader("Example Dishes")
cols = st.columns(6)
for col, dish in zip(cols, class_labels):
    with col:
        try:
            st.image(f"examples/{dish}.jpg", 
                     caption=dish, 
                     use_container_width=True)
        except FileNotFoundError:
            st.write(f"Example image for {dish} not found")

# Add footer
st.markdown("---")
st.markdown("**Note**: Predictions are based on visual analysis and may not be 100% accurate. Always verify with culinary experts.")