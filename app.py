# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
os.environ["FASTAI_DISABLE_PROGRESS_BAR"] = "1"
from fastai.vision.all import create_cnn_model, resnet18
import google.generativeai as genai
import os

# Set Gemini API key
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    google_api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key = google_api_key)

# Class index to dish name
idx_to_label = {
    0: "Abacha",
    1: "Egusi Soup",
    2: "Nsala Soup",
    3: "Oha Soup",
    4: "Okro Soup",
    5: "Onugbu Soup"
}

@st.cache_resource
def load_model(weights_path="igbo_dish_model_weights.pth", num_classes=6):
    model = create_cnn_model(resnet18, n_out=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# Load class labels
with open('classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Image transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image):
    image_tensor = transform_image(image.convert("RGB"))
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        idx = torch.argmax(probs).item()
        return idx_to_label[idx], probs[idx].item()

@st.cache_data(show_spinner=False)
def get_dish_info(dish_name):
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    Provide structured information in markdown format about the Igbo dish '{dish_name}'.
    
    Include:
    ## History
    (Brief cultural or historical background.)

    ## Ingredients
    (List of key ingredients in bullet points.)

    ## Preparation
    (A few sentences describing traditional preparation.)

    If the dish is unknown, state so.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"**Error fetching dish info:** {e}"

model = load_model()

# Streamlit UI
def main():
    st.title("🍲 Igbo Dish Classifier")

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
                label, confidence = predict(model, img)

    if uploaded_file:
        with col2:
            st.subheader(f"**Identification**: {label}")
            st.metric("Confidence Level", f"{confidence*100:.1f}%")

            markdown_info = get_dish_info(label)

            with st.expander("📖 Dish Information", expanded=True):
                st.markdown(markdown_info)

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
