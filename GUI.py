import streamlit as st
import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from PIL import Image

# Load the model
loaded_model = torch.jit.load("colorization_model_scripted.pt")
loaded_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Model loaded successfully")

def preprocess_image(image):
    """
    Preprocess the input image for the model.
    """
    # Resize and convert to LAB
    image_resized = resize(image, (128, 128), anti_aliasing=True)
    L_channel = rgb2lab(image_resized)[:, :, 0]
    L_tensor = torch.tensor(L_channel).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
    return L_tensor, image_resized

def colorize_image(L_tensor):
    """
    Use the model to predict AB channels and combine with L channel.
    """
    with torch.no_grad():
        L_pseudo_rgb = L_tensor.repeat(1, 3, 1, 1)  # Repeat L channel to create 3-channel input
        ab_channels = loaded_model(L_pseudo_rgb).cpu().numpy()[0].transpose(1, 2, 0) * 128
    return ab_channels

def histogram_stretching(image):
    """
    Apply histogram stretching to an image.
    Scales values to the range [0, 1].
    """
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) / (max_val - min_val)
    return stretched

def postprocess_image(L_channel, ab_channels):
    """
    Combine L and AB channels into a colorized image, adjust color channels, and apply histogram stretching.
    """
    lab_image = np.zeros((128, 128, 3))
    lab_image[:, :, 0] = L_channel
    lab_image[:, :, 1:] = ab_channels
    colorized_image = lab2rgb(lab_image)

    # Multiply each color channel by 1.3
    colorized_image = np.clip(colorized_image * 1.3, 0, 1)

    # Apply histogram stretching
    colorized_image = histogram_stretching(colorized_image)

    return colorized_image

# Streamlit App
st.title("Image Colorization")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the image
    image = np.array(Image.open(uploaded_file).convert('RGB')) / 255.0
    L_tensor, original_resized = preprocess_image(image)

    # Colorize the image using the model
    ab_channels = colorize_image(L_tensor)
    colorized_image = postprocess_image(L_tensor.cpu().numpy()[0, 0], ab_channels)

    # Display results
    st.subheader("Original Image")
    st.image(original_resized, caption="Uploaded Image (Resized)", use_column_width=True)

    st.subheader("Colorized Image")
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
