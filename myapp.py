import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

st.image("http://www.ehtp.ac.ma/images/lo.png")


# Load your saved model
model = load_model('pneumonia_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to a numpy array
    img = np.array(image)
    
    # Resize the image to match the model's input shape (150x150)
    img = cv2.resize(img, (150, 150))
    
    # If the image has 3 channels (RGB), convert it to grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image (values between 0 and 1)
    img = img / 255.0
    
    # Reshape to (1, 150, 150, 1) for the model input
    img = img.reshape(1, 150, 150, 1)
    
    return img

# Streamlit app layout
st.title("Pneumonia Detection from X-Ray Images")

st.write("Upload a chest X-ray image to predict if it shows signs of pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray image.", use_column_width=True)
    
    # Preprocess the image for the model
    st.write("Processing...")
    img = preprocess_image(image)
    
    # Make prediction using the model
    prediction = model.predict(img)

    # Apply threshold for binary classification ( normal=1, pneumia 0)
    prediction_class = (prediction > 0.4).astype(int)

    # Display the result
    if prediction_class == 0:
        st.write("Prediction: The X-ray shows signs of **Pneumonia**.")

    else:
        st.write("Prediction: The X-ray is likely **Normal**.")
else:
    st.write("Please upload an X-ray image.")
