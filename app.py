# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import wikipediaapi
import numpy as np

# Set up page config
st.set_page_config(
    page_title="Igbo Dish Classifier",
    page_icon="🍲",
    layout="wide"
)

# Load model function with caching
@st.cache_resource
def load_model():
    # Define your model architecture (must match training)
    def create_model():
        arch = resnet18  # Update with your actual architecture
        return create_cnn_model(arch, n_out=6, pretrained=False)  # 6 classes
    
    model = create_model()
    model.load_state_dict(torch.load('igbo_dish_model_weights.pth', map_location='cpu'))
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image):
    model = load_model()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.numpy()[0]

# Wikipedia information function
def get_dish_info(dish_name):
    page = wiki.page(dish_name + " soup")
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
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
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
        st.progress(float(confidence), text=f"Confidence: {confidence*100:.1f}%")
        
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
        st.page_link("https://en.wikipedia.org/wiki/Igbo_cuisine", label="Wikipedia: Igbo Cuisine")
    else:
        st.write("Upload an image to get started!")

# Add sample images
st.subheader("Example Dishes")
cols = st.columns(6)
for i, (col, dish) in enumerate(zip(cols, class_labels)):
    with col:
        st.image(f"examples/{dish}.jpg", caption=dish, use_column_width=True)