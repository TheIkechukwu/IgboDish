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
    st.info("API Key loaded from Streamlit secrets.")
except KeyError:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        st.info("API Key loaded from environment variable.")


if not google_api_key:
    st.error("Gemini API Key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
    st.stop()

# It's crucial to configure genai ONLY if a key is found.
genai.configure(api_key=google_api_key)

global model_genai
model_genai = None

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

# Gemini API info fetcher with enhanced debugging
@st.cache_data(show_spinner="Fetching delicious details with Gemini...", ttl=3600) # Added TTL for cache
def get_dish_info_gemini(dish_name):
    st.info(f"Attempting to fetch info for: {dish_name}") # Debug print
    if model_genai is None:
        st.error("Gemini model is still None inside get_dish_info_gemini. This shouldn't happen.")
        return "Gemini model could not be initialized. Please check API key and model availability."

    prompt = f"""
    Provide information about the Igbo dish '{dish_name}'.
    Organize the information into the following sections:
    1.  **Cultural Significance/History**: Briefly describe its origin, cultural importance, or traditional context.
    2.  **Key Ingredients**: List the primary ingredients.
    3.  **Traditional Preparation**: Briefly explain the typical method of preparation.

    Format your response clearly with bold headings for each section. If a piece of information is not readily available, state "Information not available."
    """
    st.info(f"Prompt being sent to Gemini: \n```\n{prompt}\n```") # Debug print

    try:
        response = model_genai.generate_content(prompt)
        st.info(f"Raw Gemini API Response Object Type: {type(response)}") # Debug print

        if response.text:
            st.success("Gemini returned text content.")
            return response.text
        else:
            debug_info = ["Gemini returned an empty text response. Debugging information:"]
            if response.candidates:
                for i, candidate in enumerate(response.candidates):
                    debug_info.append(f"Candidate {i}:")
                    debug_info.append(f"  Finish Reason: {candidate.finish_reason}")
                    if candidate.safety_ratings:
                        safety_ratings_str = ", ".join([
                            f"{sr.category.name}: {sr.probability.name}"
                            for sr in sr.category.name, sr.probability.name in candidate.safety_ratings
                        ])
                        debug_info.append(f"  Safety Ratings: {safety_ratings_str}")
                    else:
                        debug_info.append("  No safety ratings provided for candidate.")
            else:
                debug_info.append("No candidates in response.")

            if response.prompt_feedback and response.prompt_feedback.safety_ratings:
                prompt_safety_str = ", ".join([
                    f"{sr.category.name}: {sr.probability.name}"
                    for sr in response.prompt_feedback.safety_ratings
                ])
                debug_info.append(f"Prompt Feedback Safety: {prompt_safety_str}")
            else:
                debug_info.append("No prompt feedback safety info.")

            st.warning("\n".join(debug_info)) # Display debug info as one message

            return "Could not retrieve detailed information at this time. (Gemini returned no text content)"

    except Exception as e:
        st.error(f"Error fetching info from Gemini: {type(e).__name__} - {e}")
        return "Could not retrieve detailed information at this time. (Gemini API error)"

model = load_model()

# Streamlit UI
def main():
    st.title("🍲 Igbo Dish Classifier")

    global model_genai

    st.subheader("Gemini Model Initialization Check")
    # --- Model Availability Check ---
    try:
        st.info("Attempting to connect to 'gemini-pro'...")
        model_genai = genai.GenerativeModel('gemini-pro')
        # Test if it supports generate_content
        test_response = model_genai.generate_content("hello", stream=True)
        for chunk in test_response:
            pass # Iterate to force connection and catch errors early
        st.success("Successfully connected to 'gemini-pro' model.")
    except Exception as e:
        st.warning(f"Could not initialize 'gemini-pro': {type(e).__name__} - {e}")
        st.info("Attempting to find an alternative Gemini model...")

        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and 'TEXT' in m.supported_input_formats:
                     available_models.append(m.name)

            if not available_models:
                st.error("No suitable Gemini model found supporting 'generateContent' with text input. Please check your API key region and Google AI Studio dashboard.")
                st.stop()

            preferred_models = ['models/gemini-1.5-flash-latest', 'models/gemini-1.0-pro']
            found_preferred = False
            for preferred_name in preferred_models:
                if preferred_name in available_models:
                    model_genai = genai.GenerativeModel(preferred_name)
                    st.success(f"Switched to '{preferred_name}'.")
                    found_preferred = True
                    break

            if not found_preferred:
                model_name = available_models[0].split('/')[-1]
                model_genai = genai.GenerativeModel(model_name)
                st.success(f"Switched to '{model_name}' (first available).")

            st.info(f"Currently active Gemini model for content generation: `{model_genai.model_name}`.")
            st.info(f"Full list of available models supporting generateContent (text): {', '.join(available_models)}")


        except Exception as list_e:
            st.error(f"Error listing available models: {type(list_e).__name__} - {list_e}")
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

            gemini_info_text = get_dish_info_gemini(label)

            if "Gemini returned an empty text response." in gemini_info_text or \
               "Could not retrieve detailed information at this time." in gemini_info_text:
                st.write(gemini_info_text)
                history_content = "Cultural history documentation in progress."
                ingredients_content = "Typical ingredients being researched."
                preparation_content = "Preparation methods coming soon."
            else:
                history_content = "Cultural history documentation in progress."
                ingredients_content = "Typical ingredients being researched."
                preparation_content = "Preparation methods coming soon."

                if "**Cultural Significance/History**:" in gemini_info_text:
                    history_segment_end_idx = len(gemini_info_text)
                    if "**Key Ingredients**:" in gemini_info_text:
                        history_segment_end_idx = gemini_info_text.find("**Key Ingredients**:")
                    elif "**Traditional Preparation**:" in gemini_info_text:
                        history_segment_end_idx = gemini_info_text.find("**Traditional Preparation**:")
                    history_content = gemini_info_text[gemini_info_text.find("**Cultural Significance/History**:") + len("**Cultural Significance/History**:") : history_segment_end_idx].strip()

                if "**Key Ingredients**:" in gemini_info_text:
                    ingredients_segment_end_idx = len(gemini_info_text)
                    if "**Traditional Preparation**:" in gemini_info_text:
                        ingredients_segment_end_idx = gemini_info_text.find("**Traditional Preparation**:")
                    ingredients_content = gemini_info_text[gemini_info_text.find("**Key Ingredients**:") + len("**Key Ingredients**:") : ingredients_segment_end_idx].strip()

                if "**Traditional Preparation**:" in gemini_info_text:
                    preparation_content = gemini_info_text[gemini_info_text.find("**Traditional Preparation**:") + len("**Traditional Preparation**:") :].strip()

                if not history_content: history_content = "Cultural history documentation in progress."
                if not ingredients_content: ingredients_content = "Typical ingredients being researched."
                if not preparation_content: preparation_content = "Preparation methods coming soon."

            with st.expander("🌍 Cultural Significance", expanded=True):
                st.write(history_content)

            with st.expander("🛒 Key Ingredients"):
                st.write(ingredients_content)

            with st.expander("👩🍳 Traditional Preparation"):
                st.write(preparation_content)

            st.markdown("---")
            st.write("**Explore More**")
            st.page_link("https://en.wikipedia.org/wiki/Igbo_cuisine",
                         label="Igbo Culinary Traditions on Wikipedia")

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