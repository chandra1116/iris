# iris_app.py
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier().fit(X, y)

species_images = {
    "setosa": "Irissetosa1.jpg",
    "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# Set up page
st.set_page_config(page_title="🌸 Iris Predictor", layout="centered")

# Custom CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: #bd1eb2;
            color: #1e6dbd;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 3em;
            color: #1e6dbd;
            margin-bottom: 0.5em;
        }
        .footer {
            text-align: center;
            color: #555;
            margin-top: 2em;
            font-size: 0.9em;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🌼🐒 Iris Species Predictor🌼🐒</div>', unsafe_allow_html=True)
st.write("Adjust the sliders below to predict the Iris species based on flower measurements:")

# Input sliders
sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
input_features = np.array([[sl, sw, pl, pw]])
prediction = model.predict(input_features)[0]
predicted_species = iris.target_names[prediction]

# Display result
st.subheader("🌺 Predicted Species:")
st.success(f"{predicted_species.capitalize()}")

# Image display with updated key and container width
image_path = species_images[predicted_species]
if image_path.endswith(".jpg"):
    st.image(image_path, caption=f"Iris {predicted_species}", use_container_width=True)
else:
    st.image(image_path, caption=f"Iris {predicted_species}", use_container_width=True)


st.markdown('<div class="footer">MADE DURING ON NDF INTERNSHIP.</div>', unsafe_allow_html=True)
