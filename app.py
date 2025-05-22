# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from fastai.vision.all import create_cnn_model, resnet18
import google.generativeai as genai
import os # To access environment variables for API key

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

# Load class labels (ensure classes.txt exists)
with open('classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Configure Gemini API
try:
    # Attempt to get API key from Streamlit secrets or environment variable
    # Streamlit Cloud uses st.secrets, local development can use os.environ
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Gemini API Key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
    st.stop() # Stop the app if no API key

genai.configure(api_key=google_api_key)

# Initialize the Generative Model
model_genai = genai.GenerativeModel('gemini-pro')

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

# Gemini API info fetcher
@st.cache_data(show_spinner="Fetching delicious details with Gemini...")
def get_dish_info_gemini(dish_name):
    # Craft a clear and specific prompt for Gemini
    prompt = f"""
    Provide information about the Igbo dish '{dish_name}'.
    Organize the information into the following sections:
    1.  **Cultural Significance/History**: Briefly describe its origin, cultural importance, or traditional context.
    2.  **Key Ingredients**: List the primary ingredients.
    3.  **Traditional Preparation**: Briefly explain the typical method of preparation.

    Format your response clearly with bold headings for each section. If a piece of information is not readily available, state "Information not available."
    """
    try:
        response = model_genai.generate_content(prompt)
        # Access the text directly from the response
        return response.text
    except Exception as e:
        st.error(f"Error fetching info from Gemini: {e}")
        return "Could not retrieve detailed information at this time."

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

            # Fetch info using Gemini
            gemini_info_text = get_dish_info_gemini(label)

            # Parse the Gemini response into sections
            # This is a simple parsing. You might need more robust parsing
            # if Gemini's output format varies.
            history_start = gemini_info_text.find("**Cultural Significance/History**:")
            ingredients_start = gemini_info_text.find("**Key Ingredients**:")
            preparation_start = gemini_info_text.find("**Traditional Preparation**:")

            history_content = "Cultural history documentation in progress."
            ingredients_content = "Typical ingredients being researched."
            preparation_content = "Preparation methods coming soon."

            if history_start != -1:
                end_idx = min(ingredients_start if ingredients_start != -1 else len(gemini_info_text),
                              preparation_start if preparation_start != -1 else len(gemini_info_text))
                history_content = gemini_info_text[history_start + len("**Cultural Significance/History**:") : end_idx].strip()
                if not history_content: history_content = "Cultural history documentation in progress."


            if ingredients_start != -1:
                end_idx = min(preparation_start if preparation_start != -1 else len(gemini_info_text),
                              len(gemini_info_text))
                ingredients_content = gemini_info_text[ingredients_start + len("**Key Ingredients**:") : end_idx].strip()
                if not ingredients_content: ingredients_content = "Typical ingredients being researched."

            if preparation_start != -1:
                preparation_content = gemini_info_text[preparation_start + len("**Traditional Preparation**:") :].strip()
                if not preparation_content: preparation_content = "Preparation methods coming soon."

            # Display information using expanders
            with st.expander("🌍 Cultural Significance", expanded=True):
                st.write(history_content)

            with st.expander("🛒 Key Ingredients"):
                st.write(ingredients_content)

            with st.expander("👩🍳 Traditional Preparation"):
                st.write(preparation_content)

            st.markdown("---")
            st.write("**Explore More**")
            st.page_link("https://en.wikipedia.org/wiki/Igbo_cuisine",
                         label="Igbo Culinary Traditions on Wikipedia") # Keep this as a general resource

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