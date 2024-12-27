import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize

# Load the model
loaded_model = torch.jit.load("colorization_model_scripted.pt")
loaded_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Model loaded successfully")

def histogram_stretching(image):
    """
    Apply histogram stretching to an image.
    Scales values to the range [0, 1] for processing.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) / (max_val - min_val)
    return stretched

def colorize_and_adjust(image, fr, fg, fb):
    """
    Adjust RGB channels by the provided factors.
    """
    adjusted = image.copy()
    adjusted[:, :, 0] *= fr  # Adjust red channel
    adjusted[:, :, 1] *= fg  # Adjust green channel
    adjusted[:, :, 2] *= fb  # Adjust blue channel
    adjusted = np.clip(adjusted, 0, 1)  # Ensure values are within [0, 1]
    return adjusted

def postprocess(image, sr, sg, sb):
    """
    Perform histogram stretching and color adjustments.
    """
    # Normalize the image
    image= np.array(image / 255.0)


    # Apply histogram stretching
    stretched = np.zeros_like(image)
    for i, s in enumerate([sr, sg, sb]):  # Apply stretching to each channel
        stretched[:, :, i] = histogram_stretching(image[:, :, i]) * s

    # Convert to LAB and back to RGB
    lab = rgb2lab(stretched)
    lab[:, :, 0] = histogram_stretching(lab[:, :, 0]) * 100  # Stretch L channel
    processed = lab2rgb(lab)

    return image, processed

# Streamlit App
st.title("Image Colorization and Postprocessing")

st.sidebar.header("Colorization Settings")
fr = st.sidebar.slider("Red Factor", 0.5, 2.0, 1.2)
fg = st.sidebar.slider("Green Factor", 0.5, 2.0, 1.15)
fb = st.sidebar.slider("Blue Factor", 0.5, 2.0, 1.35)

st.sidebar.header("Histogram Stretching")
sr = st.sidebar.slider("Stretch Red", 0.0, 1.0, 0.7)
sg = st.sidebar.slider("Stretch Green", 0.0, 1.0, 0.8)
sb = st.sidebar.slider("Stretch Blue", 0.0, 1.0, 1.0)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = plt.imread(uploaded_file)

    # Perform colorization and postprocessing
    original, processed = postprocess(image, sr, sg, sb)

    # Adjust the color channels
    processed = colorize_and_adjust(processed, fr, fg, fb)

    # Display results
    st.subheader("Original Image")
    st.image(original, caption="Uploaded Image", use_column_width=True)

    st.subheader("Colorized and Processed Image")
    st.image(processed, caption="Colorized Image", use_column_width=True)