import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import io
import gdown
from transformers import TFSegformerForSemanticSegmentation

# Set page configuration
st.set_page_config(
    page_title="Pet Segmentation with SegFormer",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for image preprocessing
IMAGE_SIZE = 512
OUTPUT_SIZE = 128
MEAN = tf.constant([0.485, 0.456, 0.406])
STD = tf.constant([0.229, 0.224, 0.225])

# Class labels
ID2LABEL = {0: "background", 1: "border", 2: "foreground/pet"}
NUM_CLASSES = len(ID2LABEL)

@st.cache_resource
def download_model_from_drive():
    """
    Download model from Google Drive
    
    Returns:
        Path to downloaded model
    """
    # Define paths
    model_dir = os.path.join("models", "saved_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "segformer_model")
    
    # Check if model already exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            try:
                # Google Drive file ID from the shared link
                file_id = "1XObpqG8qZ7YUyiRKbpVvxX11yQSK8Y_3"
                
                # Download the model file
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")
                return None
    else:
        st.info("Model already exists locally.")
    
    return model_path

@st.cache_resource
def load_model():
    """
    Load the SegFormer model
    
    Returns:
        Loaded model
    """
    try:
        # Download the model first
        model_path = download_model_from_drive()
        
        if model_path is None:
            st.warning("Using default pretrained model since download failed")
            # Fall back to pretrained model
            model = TFSegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=NUM_CLASSES,
                id2label=ID2LABEL,
                label2id={label: id for id, label in ID2LABEL.items()},
                ignore_mismatched_sizes=True
            )
        else:
            # Load downloaded model
            model = TFSegformerForSemanticSegmentation.from_pretrained(model_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Falling back to pretrained model")
        # Fall back to pretrained model as a last resort
        model = TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id={label: id for id, label in ID2LABEL.items()},
            ignore_mismatched_sizes=True
        )
        return model

def normalize_image(input_image):
    """
    Normalize the input image
    
    Args:
        input_image: Image to normalize
        
    Returns:
        Normalized image
    """
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - MEAN) / tf.maximum(STD, backend.epsilon())
    return input_image

def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed image tensor, original image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Store original image for display
    original_img = img_array.copy()
    
    # Resize to target size
    img_resized = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normalize
    img_normalized = normalize_image(img_resized)
    
    # Transpose from HWC to CHW (SegFormer expects channels first)
    img_transposed = tf.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = tf.expand_dims(img_transposed, axis=0)
    
    return img_batch, original_img

def create_mask(pred_mask):
    """
    Convert model prediction to displayable mask
    
    Args:
        pred_mask: Prediction from model
        
    Returns:
        Processed mask for visualization
    """
    # Get the class with highest probability (argmax along class dimension)
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    
    # Add channel dimension
    pred_mask = tf.expand_dims(pred_mask, -1)
    
    # Resize to original image size
    pred_mask = tf.image.resize(
        pred_mask, 
        (IMAGE_SIZE, IMAGE_SIZE), 
        method="nearest"
    )
    
    return pred_mask[0]

def colorize_mask(mask):
    """
    Apply colors to segmentation mask
    
    Args:
        mask: Segmentation mask
        
    Returns:
        Colorized mask
    """
    # Define colors for each class (RGB)
    colors = [
        [0, 0, 0],      # Background (black)
        [255, 0, 0],    # Border (red)
        [0, 0, 255]     # Foreground/pet (blue)
    ]
    
    # Create RGB mask
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        # Find pixels of this class and assign color
        class_mask = np.where(mask == i, 1, 0).astype(np.uint8)
        for c in range(3):
            rgb_mask[:, :, c] += class_mask * color[c]
    
    return rgb_mask

def create_overlay(image, mask, alpha=0.5):
    """
    Create an overlay of mask on original image
    
    Args:
        image: Original image
        mask: Segmentation mask
        alpha: Transparency level (0-1)
        
    Returns:
        Overlay image
    """
    # Ensure mask shape matches image
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create blend
    overlay = cv2.addWeighted(
        image, 
        1, 
        mask.astype(np.uint8), 
        alpha, 
        0
    )
    
    return overlay

def main():
    st.title("🐶 Pet Segmentation with SegFormer")
    st.markdown("""
        This app demonstrates semantic segmentation of pet images using a SegFormer model.
        The model segments images into three classes:
        - **Background**: Areas around the pet
        - **Border**: The boundary/outline around the pet
        - **Foreground**: The pet itself
    """)
    
    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
        **SegFormer** is a state-of-the-art semantic segmentation model based on transformers.
        
        Key features:
        - Hierarchical transformer encoder
        - Lightweight MLP decoder
        - Efficient mix of local and global attention
        
        This implementation uses the MIT-B0 variant fine-tuned on the Oxford-IIIT Pet dataset.
    """)
    
    # Advanced settings in sidebar
    st.sidebar.header("Settings")
    
    # Overlay opacity
    overlay_opacity = st.sidebar.slider(
        "Overlay Opacity", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    
    # Load model
    with st.spinner("Loading SegFormer model..."):
        model = load_model()
        
    if model is None:
        st.error("Failed to load model. Using default pretrained model instead.")
    else:
        st.sidebar.success("Model loaded successfully!")
    
    # Image upload
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Upload a pet image:", type=["jpg", "jpeg", "png"])
    
    # Sample images option
    st.markdown("### Or use a sample image:")
    sample_dir = "samples"
    
    # Check if sample directory exists and contains images
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if sample_files:
        selected_sample = st.selectbox("Select a sample image:", sample_files)
        use_sample = st.button("Use this sample")
        
        if use_sample:
            with open(os.path.join(sample_dir, selected_sample), "rb") as file:
                image_bytes = file.read()
                uploaded_image = io.BytesIO(image_bytes)
                st.success(f"Using sample image: {selected_sample}")
    
    # Process uploaded image
    if uploaded_image is not None:
        # Display original image
        image = Image.open(uploaded_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        with st.spinner("Generating segmentation mask..."):
            # Preprocess the image
            img_tensor, original_img = preprocess_image(image)
            
            # Make prediction
            prediction = model(pixel_values=img_tensor, training=False)
            logits = prediction.logits
            
            # Create visualization mask
            mask = create_mask(logits).numpy()
            
            # Colorize the mask
            colorized_mask = colorize_mask(mask)
            
            # Create overlay
            overlay = create_overlay(original_img, colorized_mask, alpha=overlay_opacity)
        
        # Display results
        with col2:
            st.subheader("Segmentation Result")
            st.image(overlay, caption="Segmentation Overlay", use_column_width=True)
        
        # Display segmentation details
        st.header("Segmentation Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Background")
            st.markdown("Areas surrounding the pet")
            mask_bg = np.where(mask == 0, 255, 0).astype(np.uint8)
            st.image(mask_bg, caption="Background", use_column_width=True)
            
        with col2:
            st.subheader("Border")
            st.markdown("Boundary around the pet")
            mask_border = np.where(mask == 1, 255, 0).astype(np.uint8)
            st.image(mask_border, caption="Border", use_column_width=True)
            
        with col3:
            st.subheader("Foreground (Pet)")
            st.markdown("The pet itself")
            mask_fg = np.where(mask == 2, 255, 0).astype(np.uint8)
            st.image(mask_fg, caption="Foreground", use_column_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert mask to PNG for download
            mask_colored = Image.fromarray(colorized_mask)
            mask_bytes = io.BytesIO()
            mask_colored.save(mask_bytes, format='PNG')
            mask_bytes = mask_bytes.getvalue()
            
            st.download_button(
                label="Download Segmentation Mask",
                data=mask_bytes,
                file_name="pet_segmentation_mask.png",
                mime="image/png"
            )
        
        with col2:
            # Convert overlay to PNG for download
            overlay_img = Image.fromarray(overlay)
            overlay_bytes = io.BytesIO()
            overlay_img.save(overlay_bytes, format='PNG')
            overlay_bytes = overlay_bytes.getvalue()
            
            st.download_button(
                label="Download Overlay Image",
                data=overlay_bytes,
                file_name="pet_segmentation_overlay.png",
                mime="image/png"
            )
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("### About the Model")
    st.markdown("""
        This segmentation model is based on the SegFormer architecture and was fine-tuned on the Oxford-IIIT Pet dataset.
        
        **Key Performance Metrics:**
        - Mean IoU (Intersection over Union): Measures overlap between predictions and ground truth
        - Dice Coefficient: Similar to F1-score, balances precision and recall
        
        The model segments pet images into three semantic classes (background, border, and pet/foreground),
        making it useful for applications like pet image editing, background removal, and object detection.
    """)

if __name__ == "__main__":
    main()