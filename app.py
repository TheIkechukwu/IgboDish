# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from fastai.vision.all import create_cnn_model, resnet18
import google.generativeai as genai
import os

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
google_api_key = None
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Gemini API Key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
    st.stop() # Stop the app if no API key

genai.configure(api_key=google_api_key)

# Initialize the Generative Model
# We'll try to determine a suitable model dynamically or based on availability
global model_genai
model_genai = None # Initialize as None

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
    if model_genai is None:
        return "Gemini model could not be initialized. Please check API key and model availability."

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
        return response.text
    except Exception as e:
        st.error(f"Error fetching info from Gemini: {e}")
        return "Could not retrieve detailed information at this time. (Gemini API error)"

model = load_model()

# Streamlit UI
def main():
    st.title("🍲 Igbo Dish Classifier")

    global model_genai # Access the global model_genai variable

    # --- Model Availability Check ---
    try:
        # Try to initialize gemini-pro first
        model_genai = genai.GenerativeModel('gemini-pro')
        # Test if it supports generate_content
        test_response = model_genai.generate_content("hello", stream=True)
        for chunk in test_response: # Iterate to force connection
            pass
        st.success("Successfully connected to Gemini Pro.")
    except Exception as e:
        st.warning(f"Could not initialize 'gemini-pro': {e}")
        st.info("Attempting to find an alternative Gemini model...")

        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            st.info(f"Available models supporting 'generateContent': {', '.join(available_models)}")

            # Try a common alternative
            if 'models/gemini-1.5-flash-latest' in available_models:
                model_genai = genai.GenerativeModel('gemini-1.5-flash-latest')
                st.success("Switched to 'gemini-1.5-flash-latest'.")
            elif 'models/gemini-1.0-pro' in available_models:
                model_genai = genai.GenerativeModel('gemini-1.0-pro')
                st.success("Switched to 'gemini-1.0-pro'.")
            elif available_models:
                # Pick the first available if none of the preferred are found
                model_name = available_models[0].split('/')[-1] # Get just the name
                model_genai = genai.GenerativeModel(model_name)
                st.success(f"Switched to '{model_name}'.")
            else:
                st.error("No suitable Gemini model found supporting 'generateContent'. Please check your API key region and Google AI Studio.")
                st.stop() # Stop the app if no model is found

        except Exception as list_e:
            st.error(f"Error listing available models: {list_e}")
            st.error("Cannot proceed without a working Gemini model.")
            st.stop()
    # --- End Model Availability Check ---


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
            history_content = "Cultural history documentation in progress."
            ingredients_content = "Typical ingredients being researched."
            preparation_content = "Preparation methods coming soon."

            # A more robust parsing attempt:
            if "**Cultural Significance/History**:" in gemini_info_text:
                history_content = gemini_info_text.split("**Cultural Significance/History**:")[1].split("**Key Ingredients**")[0].strip()
            if "**Key Ingredients**:" in gemini_info_text:
                ingredients_content = gemini_info_text.split("**Key Ingredients**:")[1].split("**Traditional Preparation**")[0].strip()
            if "**Traditional Preparation**:" in gemini_info_text:
                preparation_content = gemini_info_text.split("**Traditional Preparation**:")[1].strip()

            # Fallback if parsing results in empty strings
            if not history_content: history_content = "Cultural history documentation in progress."
            if not ingredients_content: ingredients_content = "Typical ingredients being researched."
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