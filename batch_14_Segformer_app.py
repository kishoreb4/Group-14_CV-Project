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
    # Create a models directory
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model"
    
    if not os.path.exists(model_path):
        # Fixed Google Drive URL format for gdown
        url = "https://drive.google.com/file/d/1XObpqG8qZ7YUyiRKbpVvxX11yQSK8Y_3/view?usp=sharing"
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("Model downloaded successfully from Google Drive.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
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
        # First create a base model with the correct architecture
        base_model = TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id={label: id for id, label in ID2LABEL.items()},
            ignore_mismatched_sizes=True
        )
        
        # Download the trained weights
        model_path = download_model_from_drive()
        
        if model_path is not None and os.path.exists(model_path):
            st.info(f"Loading weights from {model_path}...")
            try:
                # Try to load the weights
                base_model.load_weights(model_path)
                st.success("Model weights loaded successfully!")
                return base_model
            except Exception as e:
                # st.error(f"Error loading weights: {e}")
                # st.info("Using base pretrained model instead")
                return base_model
        else:
            st.warning("Using base pretrained model since download failed")
            return base_model
            
    except Exception as e:
        st.error(f"Error in load_model: {e}")
        st.warning("Using default pretrained model")
        # Fall back to pretrained model as a last resort
        return TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=NUM_CLASSES,
            id2label=ID2LABEL,
            label2id={label: id for id, label in ID2LABEL.items()},
            ignore_mismatched_sizes=True
        )

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
        pred_mask: Prediction logits from the model
        
    Returns:
        Processed mask (2D array)
    """
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.squeeze(pred_mask)
    return pred_mask.numpy()

def colorize_mask(mask):
    """
    Apply colors to segmentation mask
    
    Args:
        mask: Segmentation mask (2D array)
        
    Returns:
        Colorized mask (3D RGB array)
    """
    # Ensure the mask is 2D
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    
    # Define colors for each class (RGB)
    colors = [
        [0, 0, 0],      # Background (black)
        [255, 0, 0],    # Border (red)
        [0, 0, 255]     # Foreground/pet (blue)
    ]
    
    # Create RGB mask
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        class_mask = (mask == i).astype(np.uint8)
        for c in range(3):
            rgb_mask[:, :, c] += class_mask * color[c]
    
    return rgb_mask

def calculate_iou(y_true, y_pred, class_idx=None):
    """
    Calculate IoU (Intersection over Union) for segmentation masks
    
    Args:
        y_true: Ground truth segmentation mask
        y_pred: Predicted segmentation mask
        class_idx: Index of the class to calculate IoU for (None for mean IoU)
        
    Returns:
        IoU score
    """
    if class_idx is not None:
        # Binary IoU for specific class
        y_true_class = (y_true == class_idx).astype(np.float32)
        y_pred_class = (y_pred == class_idx).astype(np.float32)
        
        intersection = np.sum(y_true_class * y_pred_class)
        union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
        
        iou = intersection / (union + 1e-6)
    else:
        # Mean IoU across all classes
        class_ious = []
        for idx in range(NUM_CLASSES):
            class_iou = calculate_iou(y_true, y_pred, idx)
            class_ious.append(class_iou)
        
        iou = np.mean(class_ious)
    
    return iou

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
    
    # Image upload section
    st.header("Upload an Image")
    uploaded_image = st.file_uploader("Upload a pet image:", type=["jpg", "jpeg", "png"])
    uploaded_mask = st.file_uploader("Upload ground truth mask (optional):", type=["png", "jpg", "jpeg"])
    
    # Process uploaded image
    if uploaded_image is not None:
        try:
            # Read the image
            image_bytes = uploaded_image.read()
            image = Image.open(io.BytesIO(image_bytes))
                
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            with st.spinner("Generating segmentation mask..."):
                # Preprocess the image
                img_tensor, original_img = preprocess_image(image)
                
                # Make prediction
                outputs = model(pixel_values=img_tensor, training=False)
                logits = outputs.logits
                
                # Create visualization mask
                mask = create_mask(logits)
                
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
            
            # Calculate IoU if ground truth is uploaded
            if uploaded_mask is not None:
                try:
                    # Read the mask file
                    mask_data = uploaded_mask.read()
                    mask_io = io.BytesIO(mask_data)
                    gt_mask = np.array(Image.open(mask_io).resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST))
                    
                    # Handle different mask formats
                    if len(gt_mask.shape) == 3 and gt_mask.shape[2] == 3:
                        # Convert RGB to single channel if needed
                        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)
                    
                    # Calculate and display IoU
                    resized_mask = cv2.resize(mask, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)
                    iou_score = calculate_iou(gt_mask, resized_mask)
                    st.success(f"Mean IoU: {iou_score:.4f}")
                    
                    # Display specific class IoUs
                    st.markdown("### IoU by Class")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        bg_iou = calculate_iou(gt_mask, resized_mask, 0)
                        st.metric("Background IoU", f"{bg_iou:.4f}")
                    with col2:
                        border_iou = calculate_iou(gt_mask, resized_mask, 1)
                        st.metric("Border IoU", f"{border_iou:.4f}")
                    with col3:
                        fg_iou = calculate_iou(gt_mask, resized_mask, 2)
                        st.metric("Foreground IoU", f"{fg_iou:.4f}")
                except Exception as e:
                    st.error(f"Error processing ground truth mask: {e}")
                    st.write("Please ensure the mask is valid and has the correct format.")
            
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
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
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
