# -*- coding: utf-8 -*-
"""Batch-14-app

"""
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import os
import io

# Function to download model from Google Drive
@st.cache_resource
def download_model_from_drive():
    model_path = "unet_model.h5"
    if not os.path.exists(model_path):       
        url = "https://drive.google.com/file/d/1XObpqG8qZ7YUyiRKbpVvxX11yQSK8Y_3"  
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("Model downloaded successfully from Google Drive.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
    else:
        st.info("Model already exists locally.")
    return model_path

# Load the trained model
@st.cache_resource
def load_model():
    model_path = download_model_from_drive()
    if model_path is None or not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please check the Google Drive link.")
        return None
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"dice_loss": dice_loss, "iou_metric": iou_metric}
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Define Dice Loss
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Define IoU Metric
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)

# IoU Calculation for Individual Prediction
def calculate_iou(y_true, y_pred):
    y_true = y_true.astype(np.float32)
    y_pred = (y_pred > 0.5).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)

# Preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Main Streamlit app
def main():
    st.title("Pet Segmentation with U-Net")
    st.write("Upload an image of a pet to segment it and compute IoU (if a ground truth mask is provided).")

    # Load the model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.stop()

    uploaded_image = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png"])
    uploaded_mask = st.file_uploader("Upload ground truth mask (PNG, optional)", type=["png"])

    if uploaded_image is not None:
        try:
            # Read and reset the file pointer for the image
            image_data = uploaded_image.read()
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

        with st.spinner("Predicting mask..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0]

        st.image(prediction.squeeze(), caption="Predicted Mask", clamp=True, use_container_width=True)

        if uploaded_mask is not None:
            try:
                # Read and debug the mask file
                mask_data = uploaded_mask.read()
                st.write(f"Uploaded mask size: {len(mask_data)} bytes")
                if len(mask_data) < 100:  # Arbitrary threshold for suspiciously small files
                    st.warning("Mask file is very small and may be invalid.")
                
                # Open the mask from bytes
                mask_io = io.BytesIO(mask_data)
                mask = Image.open(mask_io).convert('L')
                
                # Process the mask
                mask = np.array(mask.resize((128, 128), Image.NEAREST))
                mask = np.where(mask == 1, 1, 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
                iou = calculate_iou(mask, prediction)
                st.write(f"Intersection over Union (IoU): {iou:.4f}")
            except Exception as e:
                st.error(f"Error loading mask: {e}")
                st.write("Please ensure the mask is a valid PNG file.")

if __name__ == "__main__":
    main()