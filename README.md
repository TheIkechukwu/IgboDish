# 🍲 Igbo Dish Classifier

## ✨ Overview

The **Igbo Dish Classifier** is an intelligent deep learning app that classifies images of traditional Igbo dishes and provides rich cultural context in real time.

Built with PyTorch (ResNet18 + FastAI-style head) and Streamlit, the app not only predicts the dish name but also fetches historical background, ingredients, and preparation information live from Wikipedia.

---

## ⚡ Features

* 🔎 **Dish classification** using a custom ResNet18-based model
* 📈 **Confidence score** shown as a visual slider
* 🗺 **Cultural history** automatically retrieved from Wikipedia
* 🍅 **Key ingredient detection** from Wikipedia summaries
* 👩‍🍳 **Preparation methods** overview
* 🌍 Interactive Streamlit web interface

---

## 🚀 Demo

**[Launch the App](https://igbodish.streamlit.app/)**

---

## 🏗️ Architecture

* **Model Backbone:** ResNet18 (torchvision) without final pooling & FC
* **Custom Head:** AdaptiveConcatPool2d (FastAI style) + fully connected layers
* **Deployment:** Streamlit

---

## 💻 How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/TheIkechukwu/IgboDish.git
cd IgboDish
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add your data

Organize images inside `data/train/` as:

```
data/
└── train/
    ├── Oha_Soup/
    ├── Egusi_Soup/
    └── ...
```

(Each subfolder = one class)

### 4️⃣ Train the model

```bash
python train.py
```

### 5️⃣ Launch the app

```bash
streamlit run app.py
```

---

## 📁 Files and Folders

```
app.py               # Streamlit app
train.py            # Training script
utils.py            # Helper functions (if separated)
igbo_dish_model_weights.pth # Saved model weights
classes.txt         # Class names
requirements.txt
examples/           # Example images for UI
README.md
```

---

## ✅ Model Performance

* Trained on a custom dataset of popular Igbo dishes
* Uses data augmentation (resizing, random flips)
* Outputs top prediction and confidence level

---

## 💡 Future Improvements

* Support **top-3 predictions**
* Add **ingredient visualization tags**
* Integrate **user feedback** (correct/incorrect)
* Deploy on **Hugging Face Spaces** or **Streamlit Cloud**

---

## 🙌 Acknowledgments

* Inspired by traditional Igbo cuisine
* Wikipedia API for dynamic cultural content
* FastAI-style head design for better generalization

---

## ⭐ Contributing

Pull requests and suggestions are welcome! Please open an issue first to discuss proposed changes.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🤝 Connect

* 📧 [ikechukwuemeka7@gmail.com](mailto:ikechukwuemeka7@gmail.com)
* 💼 [LinkedIn](https://www.linkedin.com/in/emeka-ikechukwu-58357b193)
* 🌟 [Portfolio](https://linktr.ee/TheIkechukwu)

---

### 🎉 Enjoy exploring Igbo cuisine with AI!
