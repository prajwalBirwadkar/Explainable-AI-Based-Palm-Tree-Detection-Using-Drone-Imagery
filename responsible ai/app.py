"""
Streamlit web application for palm tree detection with Grad-CAM explainability.
Enhanced version with more interactive visualizations.
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import io
import os
import sys
import logging
import time
import json
from typing import Tuple, List, Dict, Optional, Union, Any

# Import custom modules
from model import PalmTreeDetector

# Import all GradCAM implementations
try:
    from gradcam_enhancements import (
        generate_class_activation_map,
        create_interactive_gradcam_visualization,
        generate_comparison_grid,
        explain_model_decision
    )
except ImportError:
    st.error("GradCAM enhancements module not found. Some features may be limited.")

try:
    from gradcam_palm_detection import (
        setup_gradcam,
        compute_gradcam,
        visualize_predictions_with_gradcam
    )
except ImportError:
    st.error("GradCAM palm detection module not found. Some features may be limited.")

try:
    from gradcam_integration import compute_gradcam as compute_gradcam_integration
except ImportError:
    st.error("GradCAM integration module not found. Some features may be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add page configuration
st.set_page_config(
    page_title="Responsible AI: Palm Tree Detection",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_CHECKPOINT_PATH = "MODEL_CHECKPOINTS/model_epoch_14.pth"
MODEL_WEIGHTS_PATH = "MODEL_WEIGHTS/palm_tree_model.h5"
CONFIDENCE_THRESHOLD = 0.7
NUM_CLASSES = 2  # Background + Palm Tree

@st.cache_resource
def load_model() -> PalmTreeDetector:
    """
    Load the palm tree detection model with caching for better performance
    
    Returns:
        PalmTreeDetector instance
    """
    with st.spinner("Loading palm tree detection model... This may take a moment"):
        model = PalmTreeDetector(num_classes=NUM_CLASSES, confidence_threshold=CONFIDENCE_THRESHOLD)
        
        try:
            # First try to load from checkpoint
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_CHECKPOINT_PATH)
            if os.path.exists(model_path):
                logger.info(f"Loading model from checkpoint: {model_path}")
                model.load_model_from_checkpoint(model_path)
                return model
            
            # If checkpoint not found, try H5 weights
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_WEIGHTS_PATH)
            if os.path.exists(weights_path):
                logger.info(f"Loading model from H5 weights: {weights_path}")
                model.load_model_from_h5(weights_path)
                return model
                
            # If neither exists, show a demo mode message
            st.warning("⚠️ Model files not found. Running in DEMO MODE with limited functionality.")
            return model
            
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            logger.error(f"Error loading model: {e}")
            raise
        
@st.cache_data(max_entries=10)
def detect_palm_trees(uploaded_file, _model: PalmTreeDetector, confidence_threshold: float = None) -> Tuple[Image.Image, List[List[int]], List[float], List[int]]:
    """
    Detect palm trees in the uploaded image with caching for better performance
    
    Args:
        uploaded_file: File uploaded by the user
        model: Palm tree detection model
        confidence_threshold: Optional override for model's confidence threshold
        
    Returns:
        Tuple containing (image, boxes, scores, labels)
    """
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Store original threshold and temporarily set new one if provided
    original_threshold = None
    if confidence_threshold is not None and confidence_threshold != _model.confidence_threshold:
        original_threshold = _model.confidence_threshold
        _model.confidence_threshold = confidence_threshold
    
    try:
        # Run detection
        boxes, scores, labels = _model.detect(image)
    finally:
        # Restore original threshold if changed
        if original_threshold is not None:
            _model.confidence_threshold = original_threshold
    
    return image, boxes, scores, labels

def draw_detection_boxes(image: Image.Image, boxes: List[List[int]], scores: List[float], 
                       labels: List[int] = None, colors: List[str] = None) -> Image.Image:
    """
    Draw bounding boxes and scores on the image with enhanced styling
    
    Args:
        image: Original image
        boxes: Detected bounding boxes
        scores: Confidence scores
        labels: Optional class labels
        colors: Optional color list for different classes
        
    Returns:
        Image with bounding boxes
    """
    # Create a copy to avoid modifying the original
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    # Default colors
    if colors is None:
        colors = ["red"]
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each box and score
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        
        # Select color based on label if available, otherwise cycle through colors
        color_idx = labels[i] - 1 if labels is not None else 0  # Assuming label 1 = first class
        color = "red"  # Hardcoded to red
        
        # Draw the bounding box with rounded corners
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=6)
        
        # Draw the score with improved styling
        text = f"Palm Tree: {score:.2f}"
        
        # Calculate text dimensions
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        text_width = right - left
        text_height = bottom - top
        
        # Draw background for text
        draw.rectangle([(x1, y1 - text_height - 6), (x1 + text_width + 6, y1)], 
                      fill=color, outline=color)
        
        # Draw text
        draw.text((x1 + 3, y1 - text_height - 3), text, fill="white", font=font)
    
    return result

@st.cache_data
def generate_gradcam_visualization(_model: PalmTreeDetector, image: Image.Image, 
                                   box: List[int], score: float) -> Image.Image:
    """
    Generate Grad-CAM visualization for a specific detection
    
    Args:
        model: The model
        image: Original PIL image
        box: Bounding box coordinates [x1, y1, x2, y2]
        score: Detection confidence score
        
    Returns:
        PIL Image with Grad-CAM visualization
    """
    # Crop the region
    x1, y1, x2, y2 = box
    crop = image.crop((x1, y1, x2, y2))
    
    # Generate Grad-CAM - try different implementations
    cam = None
    
    # Try the implementation from gradcam_palm_detection first
    try:
        setup_gradcam(_model)
        cam = compute_gradcam(_model, crop)
    except (NameError, Exception) as e:
        logger.warning(f"Primary Grad-CAM implementation failed: {e}")
        # Fall back to gradcam_integration implementation
        try:
            cam = compute_gradcam_integration(crop)
        except (NameError, Exception) as e:
            logger.warning(f"Fallback Grad-CAM implementation failed: {e}")
            # Last resort: generate blank heatmap
            cam = np.zeros(crop.size[::-1], dtype=np.float32)
    
    # Convert CAM to heatmap
    try:
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (crop.width, crop.height))
    
        # Create blended image
        crop_array = np.array(crop)
        blend = (0.7 * crop_array + 0.3 * heatmap).astype(np.uint8)
        return Image.fromarray(blend)
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return crop  # Return original crop if heatmap generation fails

def create_gradcam_comparison(image: Image.Image, boxes: List[List[int]], 
                             scores: List[float], model: PalmTreeDetector, 
                             threshold: float = 0.7) -> Image.Image:
    """
    Create a side-by-side comparison of original vs. GradCAM for multiple detections
    
    Args:
        image: Original image
        boxes: Bounding boxes
        scores: Confidence scores
        model: Model for GradCAM computation
        threshold: Confidence threshold
        
    Returns:
        Comparison image
    """
    # Filter detections by threshold
    valid_indices = [i for i, score in enumerate(scores) if score >= threshold]
    
    if not valid_indices:
        return image
    
    # Original image with boxes
    image_with_boxes = draw_detection_boxes(image, 
        [boxes[i] for i in valid_indices], 
        [scores[i] for i in valid_indices])
    
    # Create overlay image for GradCAM
    image_np = np.array(image)
    overlay = image_np.copy()
    
    # Apply GradCAM to each valid detection
    for i in valid_indices:
        box = boxes[i]
        x1, y1, x2, y2 = box
        
        # Generate GradCAM for this crop
        gradcam_crop = generate_gradcam_visualization(_model=model, image=image, box=box, score=scores[i])
        gradcam_np = np.array(gradcam_crop)
        
        # Insert into overlay
        overlay[y1:y2, x1:x2] = gradcam_np
    
    # Create matplotlib figure for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original with boxes
    ax1.imshow(image_with_boxes)
    ax1.set_title("Original Image with Detections", fontsize=14)
    ax1.axis('off')
    
    # GradCAM overlay with boxes
    ax2.imshow(overlay)
    # Add boxes to the GradCAM visualization
    for i in valid_indices:
        box = boxes[i]
        score = scores[i]
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f"{score:.2f}", color='white', fontsize=10,
                 bbox=dict(facecolor='red', alpha=0.7))
    
    ax2.set_title("Grad-CAM Visualization", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

def main():
    """Main Streamlit application with enhanced UI and visualization options"""
    # Add custom CSS for styling with a more attractive design
    st.markdown("""
    <style>
    /* Modern color palette */
    :root {
        --primary: #1E88E5;       /* Primary blue */
        --primary-dark: #1565C0;  /* Darker blue */
        --secondary: #26A69A;     /* Teal accent */
        --accent: #FF7043;        /* Orange accent */
        --background: #f5f7fa;    /* Light background */
        --card-bg: white;         /* Card background */
        --text-primary: #212121;  /* Dark text */
        --text-secondary: #546E7A; /* Secondary text */
        --success: #4CAF50;       /* Green */
        --warning: #FF9800;       /* Orange */
        --error: #F44336;         /* Red */
    }
    
    /* Override Streamlit's default styling */
    .stApp {
        background-color: var(--background);
    }
    
    /* Beautiful hero section with gradient */
    .hero-section {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 2rem 3rem 3rem 3rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Card Component Styling */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-size: 1.3rem;
        color: var(--primary);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 3px solid var(--primary);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }
    
    /* Radio button styling */
    .stRadio [data-testid="stMarkdownContainer"] > div {
        background-color: var(--card-bg);
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.3rem;
        color: var(--primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 1px solid rgba(30, 136, 229, 0.2);
        padding-bottom: 0.5rem;
    }
    
    .sidebar-subheader {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem 0;
        background-color: var(--card-bg);
        border-radius: 8px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        color: var(--text-secondary);
    }
    
    /* Explanation text styling */
    .explanation-text {
        background-color: rgba(30, 136, 229, 0.05);
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    /* Detection result cards */
    .result-card {
        background-color: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: scale(1.02);
    }
    
    .result-header {
        background-color: var(--primary);
        color: white;
        padding: 0.8rem;
        font-weight: 500;
    }
    
    .result-content {
        padding: 1rem;
    }
    
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    
    /* Fix for blank spaces in columns */
    [data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove extra vertical spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }
    
    /* Make the layout more compact */
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    
    /* Specifically target and hide the problematic column */
    .st-emotion-cache-wt9exi.e6rk8up2,
    div[class*="st-emotion-cache-wt9exi"],
    .stColumn.st-emotion-cache-wt9exi {
        display: none !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a beautiful hero section with gradient background
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🌴 Responsible AI Palm Tree Detection 🌴</h1>
        <p class="hero-subtitle">Explore palm tree detection with explainable AI using Grad-CAM visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with better styling
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/palm-tree.png", width=80)
        st.markdown('<div class="sidebar-header">Settings & Controls</div>', unsafe_allow_html=True)
        
        # Model settings section with improved styling
        st.markdown('<div class="sidebar-subheader">🤖 Model Configuration</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 10px; background-color: rgba(30, 136, 229, 0.1); border-radius: 5px; margin-bottom: 15px;">', unsafe_allow_html=True)
            confidence_threshold = st.slider(
                "Detection Confidence Threshold", 
                min_value=0.1, 
                max_value=1.0, 
                value=CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Minimum confidence score required for a detection to be considered valid"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization settings with improved styling
        st.markdown('<div class="sidebar-subheader">🎨 Visualization Options</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 10px; background-color: rgba(30, 136, 229, 0.1); border-radius: 5px; margin-bottom: 15px;">', unsafe_allow_html=True)
            # Visualization type has been removed
            
            heatmap_intensity = st.slider(
                "Heatmap Intensity",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls the intensity of the GradCAM heatmap overlay"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # About section with card styling
        st.markdown('<div class="sidebar-header">ℹ️ About This App</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        This application demonstrates responsible AI principles through explainable 
        palm tree detection using Grad-CAM visualization techniques.
        
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #f0f0f0;">
        <strong>Developed by:</strong> Responsible AI Team<br>
        <strong>Version:</strong> 1.0.0
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # No documentation button here
    
    # Main content area with card-based design
    st.markdown("""
    <div class="card">
        <div class="card-title">Upload an Image or Use a Sample</div>
        <p>Select an image containing palm trees for detection and GradCAM visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload and sample images in columns with better styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background-color: rgba(30, 136, 229, 0.05); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="font-weight: 500; margin-bottom: 10px; color: #1E88E5;">📤 Upload Your Image</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select an image containing palm trees", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing palm trees to analyze"
        )
    
    with col2:
        st.markdown("""
        <div style="background-color: rgba(38, 166, 154, 0.05); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="font-weight: 500; margin-bottom: 10px; color: #26A69A;">🖼️ Use a Sample</div>
        </div>
        """, unsafe_allow_html=True)
        use_sample = st.button("Load Sample Image", help="Load a pre-included sample image of palm trees")
        # Use a predefined sample image or fallback to dataset_images
        base_dir = os.path.dirname(os.path.abspath(__file__))
        custom_sample = os.path.join(base_dir, "ck2gc6eaggd4m0748e86tcwvm.jpg")
        if use_sample:
            if os.path.exists(custom_sample):
                uploaded_file = open(custom_sample, "rb")
            else:
                sample_directory = os.path.join(base_dir, "dataset_images")
                if os.path.exists(sample_directory):
                    image_files = [f for f in os.listdir(sample_directory) if f.endswith(".jpg")]
                    if image_files:
                        sample_path = os.path.join(sample_directory, image_files[0])
                        uploaded_file = open(sample_path, "rb")
                    else:
                        st.error("No sample images found in the dataset_images directory")
                else:
                    st.error("No sample images directory found")
    
    # Create two columns for results display
    col1, col2 = st.columns(2)
    
    # Process the uploaded image
    # If a file is uploaded or sample is used
    if uploaded_file is not None:
        try:
            # Create a progress indicator for the detection process
            with st.spinner("Processing image and detecting palm trees..."):
                # Load model
                model = load_model()
                
                # Run detection with custom threshold
                image, boxes, scores, labels = detect_palm_trees(uploaded_file, _model=model, confidence_threshold=confidence_threshold)
                
                # Filter based on threshold
                filtered_indices = [i for i, score in enumerate(scores) if score >= confidence_threshold]
                filtered_boxes = [boxes[i] for i in filtered_indices]
                filtered_scores = [scores[i] for i in filtered_indices]
                filtered_labels = [labels[i] for i in filtered_indices]
            
            # Show detection metrics with enhanced attractive card styling
            st.markdown("""
            <div class="card" style="margin-top: 20px; margin-bottom: 25px;">
                <div class="card-title">Detection Results Overview</div>
                <p>Summary metrics from palm tree detection analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                tree_count = len(filtered_boxes)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{tree_count}</div>
                    <div class="metric-label">PALM TREES DETECTED</div>
                </div>
                """, unsafe_allow_html=True)
                
            with metrics_col2:
                avg_confidence = round(sum(filtered_scores)/len(filtered_scores), 2) if filtered_scores else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_confidence:.2f}</div>
                    <div class="metric-label">AVERAGE CONFIDENCE</div>
                </div>
                """, unsafe_allow_html=True)
                
            with metrics_col3:
                est_yield = len(filtered_boxes) * 85  # 85kg per tree
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{est_yield}</div>
                    <div class="metric-label">ESTIMATED YIELD (KG)</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Main visualization section with enhanced tabs design
            st.markdown("""
            <div class="card" style="margin-top: 30px; margin-bottom: 25px;">
                <div class="card-title">Interactive Visualization Dashboard</div>
                <p>Explore different visualization types to understand model decisions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different visualizations with icons
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Standard Detection", 
                "🌡️ GradCAM Visualization", 
                "🔍 Individual Analysis", 
                "💡 Advanced Insights"
            ])
            
            # Tab 1: Standard Detection View
            with tab1:
                # Draw bounding boxes on the image
                image_with_boxes = draw_detection_boxes(image, filtered_boxes, filtered_scores, filtered_labels)
                st.image(image_with_boxes, caption="Standard Palm Tree Detection", use_container_width=True)
                
                # Detection details in an expander
                with st.expander("View Detection Details"):
                    for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                        st.write(f"Detection #{i+1}: Confidence = {score:.2f}, Bounding Box = {box}")
            
            # Tab 2: GradCAM Visualization
            with tab2:
                if len(filtered_boxes) > 0:
                    # Generate GradCAM comparison
                    st.write("GradCAM highlights the features that influenced the model's decision.")
                    
                    # Create a comparison image with original and GradCAM
                    with st.spinner("Generating GradCAM visualization... This may take a moment"):
                        gradcam_image = create_gradcam_comparison(
                            image, filtered_boxes, filtered_scores, model, confidence_threshold)
                    
                    # Display the comparison
                    st.image(gradcam_image, caption="GradCAM Visualization", use_container_width=True)
                    
                    # Add an explainer about heatmaps
                    st.markdown("""
                    <div class="explanation-text">
                        <h4>Understanding the GradCAM Visualization</h4>
                        <p>This visualization shows where the model is focusing when detecting palm trees:</p>
                        <ul>
                            <li><strong>Red/Yellow areas:</strong> Features with strong influence on detection</li>
                            <li><strong>Blue/Green areas:</strong> Features with less influence</li>
                        </ul>
                        <p>For individual analysis of each detection, please see the "🔍 Individual Analysis" tab.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a progress bar for heatmap generation
                    progress_bar = st.progress(0)
                    st.info("Generating individual heatmaps... Please wait.")
                    
                    # Determine grid layout based on number of detections
                    num_detections = len(filtered_boxes)
                    cols_per_row = min(3, num_detections)  # Max 3 columns per row
                    
                    # Pre-compute all heatmaps with progress updates
                    heatmaps = []
                    with st.spinner("Processing heatmaps..."):
                        for idx, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                            # Update progress bar
                            progress = (idx + 1) / num_detections
                            progress_bar.progress(progress)
                            
                            # Generate heatmap for this detection
                            x1, y1, x2, y2 = box
                            crop = image.crop((x1, y1, x2, y2))
                            gradcam_crop = generate_gradcam_visualization(_model=model, image=image, box=box, score=score)
                            
                            # Calculate tree dimensions in pixels and relative size
                            width_px = x2 - x1
                            height_px = y2 - y1
                            area_px = width_px * height_px
                            relative_size = (area_px / (image.width * image.height)) * 100
                            
                            # Store for display
                            heatmaps.append({
                                'idx': idx,
                                'box': box,
                                'score': score,
                                'crop': crop,
                                'gradcam': gradcam_crop,
                                'width': width_px,
                                'height': height_px,
                                'area': area_px,
                                'relative_size': relative_size
                            })
                    
                    # Clear the progress indicators
                    progress_bar.empty()
                    st.success("All heatmaps generated successfully!")
                    
                    # Grid view is now the only option
                    display_option = "Grid View"
                    
                    # Create a grid of columns for displaying individual heatmaps
                    for i in range(0, num_detections, cols_per_row):
                        # Create columns for this row
                        cols = st.columns(cols_per_row)
                        
                        # For each column in this row
                        for j in range(cols_per_row):
                            idx = i + j
                            if idx < num_detections:
                                with cols[j]:
                                    heatmap_data = heatmaps[idx]
                                    score = heatmap_data['score']
                                    
                                    # Add a card-like container for each detection
                                    st.markdown(f"""
                                    <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                        <div style="font-weight: 600; color: #1E88E5; margin-bottom: 8px; font-size: 1.1rem;">Detection #{idx+1}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # 1. First show the original crop with box
                                    st.image(heatmap_data['crop'], caption=f"Original Detection", use_container_width=True)
                                    
                                    # 2. Then show the GradCAM heatmap
                                    st.image(heatmap_data['gradcam'], caption=f"GradCAM Heatmap", use_container_width=True)
                                    
                                    # 3. Show key metrics
                                    st.markdown(f"""
                                    <div style="background-color: rgba(30, 136, 229, 0.05); padding: 10px; border-radius: 5px; margin-top: 10px;">
                                        <div style="color: #666; font-size: 0.9rem;"><strong>Confidence:</strong> {score:.2f}</div>
                                        <div style="color: #666; font-size: 0.9rem;"><strong>Size:</strong> {heatmap_data['width']}×{heatmap_data['height']} px</div>
                                        <div style="color: #666; font-size: 0.9rem;"><strong>Area:</strong> {heatmap_data['area']} px²</div>
                                        <div style="color: #666; font-size: 0.9rem;"><strong>Relative Size:</strong> {heatmap_data['relative_size']:.1f}% of image</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Explanation in an expander
                    with st.expander("Understanding GradCAM Visualization"):
                        st.markdown("""
                        **What is GradCAM?**
                        
                        Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique that produces 
                        visual explanations for decisions made by CNN models. It uses the gradients flowing 
                        into the final convolutional layer to produce a coarse localization map highlighting 
                        important regions in the image for predicting the concept.
                        
                        **How to interpret:**
                        - **Red/Yellow areas**: Features strongly influencing the model's prediction
                        - **Blue/Green areas**: Features with less influence on the prediction
                        
                        This helps us understand what visual patterns the model is focusing on when identifying palm trees.
                        """)
                else:
                    st.write("No detections above the threshold to visualize.")
            
            # Tab 3: Individual detection analysis with enhanced visualization
            with tab3:
                # Add clear header for Individual Heatmaps
                st.subheader("Individual Heatmaps for Each Detection")
                
                if len(filtered_boxes) > 0:
                    # Add a card with explanation
                    st.markdown("""
                    <div class="card">
                        <div class="card-title">Individual Heatmap Analysis</div>
                        <p>Interactive analysis of each detected palm tree with detailed Grad-CAM visualization</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a progress bar for heatmap generation
                    progress_bar = st.progress(0)
                    st.info("Generating individual heatmaps... Please wait.")
                    
                    # Pre-compute all heatmaps with progress updates
                    heatmaps = []
                    with st.spinner("Processing heatmaps..."):
                        for idx, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                            # Update progress bar
                            progress = (idx + 1) / len(filtered_boxes)
                            progress_bar.progress(progress)
                            
                            # Generate heatmap for this detection
                            x1, y1, x2, y2 = box
                            crop = image.crop((x1, y1, x2, y2))
                            gradcam_crop = generate_gradcam_visualization(_model=model, image=image, box=box, score=score)
                            
                            # Calculate tree dimensions in pixels and relative size
                            width_px = x2 - x1
                            height_px = y2 - y1
                            area_px = width_px * height_px
                            relative_size = (area_px / (image.width * image.height)) * 100
                            aspect_ratio = width_px / height_px if height_px > 0 else 0
                            
                            # Store for display
                            heatmaps.append({
                                'idx': idx,
                                'box': box,
                                'score': score,
                                'crop': crop,
                                'gradcam': gradcam_crop,
                                'width': width_px,
                                'height': height_px,
                                'area': area_px,
                                'relative_size': relative_size,
                                'aspect_ratio': aspect_ratio
                            })
                    
                    # Clear the progress indicators
                    progress_bar.empty()
                    st.success("All heatmaps generated successfully!")
                    
                    # Add a selection method with a unique key
                    display_option = st.radio(
                        "Choose display method:",
                        ["Grid View", "Detailed Individual View"],
                        horizontal=True,
                        key="individual_analysis_display_option_2"
                    )
                    
                    if display_option == "Grid View":
                        # Create a grid of columns for displaying individual heatmaps
                        cols_per_row = min(3, len(heatmaps))  # Max 3 columns per row
                        for i in range(0, len(heatmaps), cols_per_row):
                            # Create columns for this row
                            cols = st.columns(cols_per_row)
                            
                            # For each column in this row
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(heatmaps):
                                    with cols[j]:
                                        heatmap_data = heatmaps[idx]
                                        score = heatmap_data['score']
                                        
                                        # Add a card-like container for each detection
                                        st.markdown(f"""
                                        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                            <div style="font-weight: 600; color: #1E88E5; margin-bottom: 8px; font-size: 1.1rem;">Detection #{idx+1}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # 1. First show the original crop with box
                                        st.image(heatmap_data['crop'], caption=f"Original Detection", use_container_width=True)
                                        
                                        # 2. Then show the GradCAM heatmap
                                        st.image(heatmap_data['gradcam'], caption=f"GradCAM Heatmap", use_container_width=True)
                                        
                                        # 3. Show key metrics
                                        st.markdown(f"""
                                        <div style="background-color: rgba(30, 136, 229, 0.05); padding: 10px; border-radius: 5px; margin-top: 10px;">
                                            <div style="color: #666; font-size: 0.9rem;"><strong>Confidence:</strong> {score:.2f}</div>
                                            <div style="color: #666; font-size: 0.9rem;"><strong>Size:</strong> {heatmap_data['width']}×{heatmap_data['height']} px</div>
                                            <div style="color: #666; font-size: 0.9rem;"><strong>Area:</strong> {heatmap_data['area']} px²</div>
                                            <div style="color: #666; font-size: 0.9rem;"><strong>Relative Size:</strong> {heatmap_data['relative_size']:.1f}% of image</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                    else:
                        # Select which detection to view in detail
                        detection_options = [f"Detection #{i+1} (Confidence: {h['score']:.2f})" for i, h in enumerate(heatmaps)]
                        selected_detection = st.selectbox(
                            "Select a detection to analyze in detail:", 
                            detection_options,
                            key="detailed_view_selectbox_2"  # Add a unique key
                        )
                        selected_idx = detection_options.index(selected_detection)
                        
                        # Get the selected heatmap data
                        heatmap_data = heatmaps[selected_idx]
                        
                        # Show detailed view with side-by-side comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Original Detection</div>
                                <p>Palm tree as identified by the detection model</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(heatmap_data['crop'], use_container_width=True)
                            
                            # Show statistics for this detection
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-header">Detection Statistics</div>
                                <div class="result-content">
                                    <table style="width:100%; border-collapse: collapse;">
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Confidence Score:</strong></td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['score']:.4f}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Dimensions:</strong></td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['width']}×{heatmap_data['height']} pixels</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Detection Area:</strong></td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['area']} px²</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Aspect Ratio:</strong></td>
                                            <td style="padding: 8px; border-bottom: 1px solid #eee;">{heatmap_data['aspect_ratio']:.2f}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 8px;"><strong>Relative Size:</strong></td>
                                            <td style="padding: 8px;">{heatmap_data['relative_size']:.2f}% of image</td>
                                        </tr>
                                    </table>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">Grad-CAM Heatmap</div>
                                <p>Visual explanation of model's detection decision</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.image(heatmap_data['gradcam'], use_container_width=True)
                            
                            # Explanation of the heatmap
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-header">Heatmap Interpretation</div>
                                <div class="result-content">
                                    <p>The Grad-CAM heatmap visualizes which parts of the image most influenced the model's decision:</p>
                                    <ul>
                                        <li><strong>Red/Yellow areas:</strong> Features strongly contributing to palm tree detection</li>
                                        <li><strong>Blue/Green areas:</strong> Features with less contribution to the detection</li>
                                    </ul>
                                    <p>This helps understand what visual patterns the model recognizes as characteristics of palm trees.</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No detections to analyze. Please upload an image with palm trees.")
            
            # Tab 4: Advanced Analysis
            with tab4:
                if len(filtered_boxes) > 0:
                    st.markdown("#### Multi-layer GradCAM Analysis")
                    st.write("This analysis shows how different layers of the model contribute to the detection.")
                    
                    # In a real implementation, this would call the explain_model_decision function
                    # from gradcam_enhancements to get multi-layer visualizations
                    
                    # For now, let's create a placeholder visualization
                    st.info("Advanced multi-layer analysis is available when the full model is loaded with feature extraction capabilities.")
                    
                    # Show yield estimation metrics
                    st.markdown("#### Yield Estimation Analysis")
                    
                    # Calculate estimated production based on detected palm trees
                    estimated_trees = len(filtered_boxes)
                    YIELD_PER_TREE_KG = 85
                    estimated_production_kg = estimated_trees * YIELD_PER_TREE_KG
                    
                    # Create a chart
                    yield_data = {
                        "Metric": ["Number of Trees", "Estimated Yield (kg)"],
                        "Value": [estimated_trees, estimated_production_kg]
                    }
                    
                    # Display as a bar chart
                    st.bar_chart(data=yield_data, x="Metric", y="Value")
                    
                    # Detailed estimation
                    st.markdown(f"""
                    **Detailed Yield Estimation:**
                    - Number of Palm Trees: {estimated_trees}
                    - Average yield per tree: {YIELD_PER_TREE_KG} kg
                    - Total estimated production: {estimated_production_kg} kg
                    - Estimated market value: ${estimated_production_kg * 0.5:.2f} (at $0.50 per kg)
                    """)
                else:
                    st.write("No detections above the threshold for advanced analysis.")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            logger.error(f"Error during detection: {e}", exc_info=True)
        
        # Add a visually enhanced section on Responsible AI considerations
        st.markdown("""
        <div class="card" style="margin-top: 40px; margin-bottom: 25px;">
            <div class="card-title">🔎 Responsible AI Considerations</div>
            <p>Ethical principles and considerations in our palm tree detection system</p>
        </div>
        """, unsafe_allow_html=True)
        
        fair_col, acc_col, trans_col = st.columns(3)
        
        with fair_col:
            st.markdown("""
            <div class="result-card">
                <div class="result-header">
                    🕵️ Fairness
                </div>
                <div class="result-content">
                    <p><strong>Current Limitations:</strong></p>
                    <ul>
                        <li>Performance variations across geographic regions</li>
                        <li>Sensitivity to image quality and lighting conditions</li>
                    </ul>
                    <p><strong>Future Work:</strong></p>
                    <ul>
                        <li>Evaluate across diverse palm tree species and environments</li>
                        <li>Implement fairness metrics to detect potential biases</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with acc_col:
            st.markdown("""
            <div class="result-card">
                <div class="result-header">
                    📊 Accountability
                </div>
                <div class="result-content">
                    <p><strong>Current Implementation:</strong></p>
                    <ul>
                        <li>Confidence scores for all detections</li>
                        <li>Grad-CAM explanations for model decisions</li>
                    </ul>
                    <p><strong>Future Work:</strong></p>
                    <ul>
                        <li>Performance metrics across diverse datasets</li>
                        <li>Uncertainty quantification for detections</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with trans_col:
            st.markdown("""
            <div class="result-card">
                <div class="result-header">
                    🔍 Transparency
                </div>
                <div class="result-content">
                    <p><strong>Current Implementation:</strong></p>
                    <ul>
                        <li>Visual explanations through Grad-CAM</li>
                        <li>Detailed model architecture information</li>
                    </ul>
                    <p><strong>Future Work:</strong></p>
                    <ul>
                        <li>Feature importance analysis</li>
                        <li>Comparison with human expert annotations</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Right column - Model explainability
    with col2:
        if uploaded_file is not None and 'model' in locals() and len(boxes) > 0:
            st.markdown('<h2 class="sub-header">Model Explainability</h2>', unsafe_allow_html=True)
            
            # Add tabs for different visualization modes
            explainability_tabs = st.tabs(["Single Detection", "All Detections", "Detailed Analysis"])
            
            # Tab 1: Single detection analysis (original implementation)
            with explainability_tabs[0]:
                if len(boxes) > 0:
                    # Create a dropdown to select which detection to explain
                    detection_options = [f"Detection {i+1}: Confidence {score:.2f}" for i, score in enumerate(scores)]
                    selected_detection = st.selectbox(
                        "Select a detection to explain with Grad-CAM",
                        options=detection_options
                    )
                    
                    # Get the selected detection index
                    selected_idx = detection_options.index(selected_detection)
                
                    try:
                        # Generate Grad-CAM explanation for the selected detection
                        gradcam_image = explain_detection(
                            image=image,
                            boxes=boxes,
                            scores=scores,
                            model=model.model,
                            detection_index=selected_idx
                        )
                        
                        # Show the Grad-CAM visualization
                        st.image(gradcam_image, caption="Grad-CAM Explanation", use_container_width=True)
                        
                        # Explanation text
                        st.markdown("""
                        <div class="explanation-text">
                        <h4>What is this visualization showing?</h4>
                        <p>The Grad-CAM heatmap highlights regions that influenced the model's detection decision:
                        <ul>
                            <li><b>Red areas</b>: Strongest influence on detection</li>
                            <li><b>Yellow/Green areas</b>: Moderate influence</li>
                            <li><b>Blue areas</b>: Minimal influence</li>
                        </ul>
                        This helps us understand what visual features the model is using to identify palm trees.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Failed to generate Grad-CAM: {e}")
                        logger.error(f"Grad-CAM error: {e}", exc_info=True)
            
            # Tab 2: All detections visualization
            with explainability_tabs[1]:
                if len(boxes) > 0:
                    st.write("This view shows all detected palm trees with Grad-CAM heatmaps for each detection.")
                    
                    try:
                        # Create interactive visualization with heatmaps for all detections
                        interactive_fig = create_interactive_gradcam_visualization(
                            image_pil, boxes, scores, labels, model
                        )
                        
                        # Display the interactive visualization
                        st.pyplot(interactive_fig)
                        
                        # Display grid of individual detections
                        st.write("### Individual Detection Analysis")
                        comparison_grid = generate_comparison_grid(
                            image_pil, boxes, scores, model, threshold=0.5
                        )
                        st.image(comparison_grid, caption="Grad-CAM for each detected palm tree", use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        <div class="explanation-text">
                        <p>The grid above shows each detected palm tree with its corresponding Grad-CAM heatmap overlay. 
                        Red/yellow regions indicate areas that strongly influenced the model's detection decision.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Failed to generate all-detection visualizations: {e}")
                        logger.error(f"All-detection visualization error: {e}", exc_info=True)
                        
            # Tab 3: Detailed analysis of model decisions
            with explainability_tabs[2]:
                if len(boxes) > 0:
                    st.write("This analysis examines how different layers of the model contribute to detections.")
                    
                    # Create a dropdown to select which detection to explain
                    detection_options = [f"Detection {i+1}: Confidence {score:.2f}" for i, score in enumerate(scores)]
                    selected_detection = st.selectbox(
                        "Select a detection for detailed analysis",
                        options=detection_options,
                        key="detailed_analysis_select"
                    )
                    
                    # Get the selected detection index
                    selected_idx = detection_options.index(selected_detection)
                    
                    try:
                        # Generate detailed explanation
                        explanation = explain_model_decision(
                            model, image_pil, selected_idx,
                            feature_layers=["backbone.body.layer4", "backbone.body.layer3"]
                        )
                        
                        if explanation["success"]:
                            # Show detection info
                            st.write(f"**Confidence Score:** {explanation['confidence']:.4f}")
                            
                            # Display the original crop
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(explanation["crop"], caption="Original Detection")
                            
                            # Display Grad-CAM visualizations for different layers
                            viz_cols = st.columns(len(explanation["layer_visualizations"]))
                            
                            for i, (layer_name, viz) in enumerate(explanation["layer_visualizations"].items()):
                                with viz_cols[i]:
                                    if isinstance(viz, Image.Image):
                                        st.image(viz, caption=f"Layer: {layer_name}")
                                    else:
                                        st.error(viz)  # Display error message
                            
                            # Add explanation about different layers
                            st.markdown("""
                            <div class="explanation-text">
                            <h4>Interpreting Different Layers</h4>
                            <p>
                            <ul>
                                <li><b>layer4</b>: Final convolutional layer that captures high-level features like overall tree shape and structure</li>
                                <li><b>layer3</b>: Intermediate layer that focuses on medium-level features like leaf patterns and branch structures</li>
                            </ul>
                            Comparing these visualizations helps us understand what specific features the model is using at different levels to identify palm trees.
                            </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(explanation["message"])
                    except Exception as e:
                        st.error(f"Failed to generate detailed analysis: {e}")
                        logger.error(f"Detailed analysis error: {e}", exc_info=True)
        
        # Show placeholder when no image is uploaded
        elif uploaded_file is None:
            st.markdown('<h2 class="sub-header">Model Explainability</h2>', unsafe_allow_html=True)
            st.info("Upload an image to see Grad-CAM explainability visualizations")
            
            # Show explanation tabs for empty state
            empty_tabs = st.tabs(["About Grad-CAM", "Visualization Types", "Benefits"])
            
            with empty_tabs[0]:
                # Show sample Grad-CAM explanation
                st.markdown("""
                <div class="explanation-text">
                <h4>How Grad-CAM helps with model explainability</h4>
                <p>Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that produces visual explanations for decisions made by CNN-based models like our palm tree detector.</p>
                <p>Grad-CAM uses the gradients flowing into the final convolutional layer to highlight important regions in the image for prediction.</p>
                <p>This transparency is important for:</p>
                <ul>
                    <li>Verifying that the model is looking at relevant features (the palm tree, not the background)</li>
                    <li>Identifying potential biases in the model's decision process</li>
                    <li>Building trust in the model's predictions by making them more interpretable</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
            with empty_tabs[1]:
                st.markdown("""
                <div class="explanation-text">
                <h4>Different Visualization Types</h4>
                <p>Our enhanced Grad-CAM implementation offers three types of visualizations:</p>
                <ol>
                    <li><b>Single Detection Analysis</b>: Focuses on one detected palm tree, highlighting exactly what the model is looking at when making that specific detection.</li>
                    <li><b>All Detections View</b>: Shows all detected palm trees with their corresponding heatmaps, giving you a global view of what the model finds important across the entire image.</li>
                    <li><b>Detailed Analysis</b>: Examines different layers of the model to understand how early, middle, and late layers contribute to the detection decisions.</li>
                </ol>
                <p>These different views help provide a comprehensive understanding of the model's decision-making process.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with empty_tabs[2]:
                st.markdown("""
                <div class="explanation-text">
                <h4>Benefits of Grad-CAM for Palm Tree Detection</h4>
                <p>Using Grad-CAM in our palm tree detection system provides several important benefits:</p>
                <ul>
                    <li><b>Transparency:</b> See exactly what the model focuses on when detecting palm trees</li>
                    <li><b>Debugging:</b> Quickly identify if the model is focusing on irrelevant features</li>
                    <li><b>User Trust:</b> Build confidence in the system by making the AI decision process visible</li>
                    <li><b>Responsible AI:</b> Ensure the model behaves as expected and addresses ethical considerations</li>
                    <li><b>Data Collection Guidance:</b> Identify what kinds of images might need more representation in the training data</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Add a section on Responsible AI considerations
    st.markdown('<h2 class="sub-header">Responsible AI Considerations</h2>', unsafe_allow_html=True)
    
    fair_col, acc_col, trans_col = st.columns(3)
    
    with fair_col:
        st.markdown("#### Fairness")
        st.markdown("""
        **Current Limitations:**
        - The model may perform differently on palm trees from different geographic regions
        - Detection performance may vary based on image quality and lighting conditions
        
        **Future Work:**
        - Evaluate performance across diverse palm tree species and environments
        - Add fairness metrics to detect potential biases
        """)
    
    with acc_col:
        st.markdown("#### Accountability")
        st.markdown("""
        **Current Implementation:**
        - Confidence scores provided for all detections
        - Grad-CAM explanations to understand model decisions
        
        **Future Work:**
        - Add model performance metrics on different datasets
        - Implement uncertainty quantification for detections
        """)
    
    with trans_col:
        st.markdown("#### Transparency")
        st.markdown("""
        **Current Implementation:**
        - Visual explanations through Grad-CAM
        - Detailed information about the model architecture
        
        **Future Work:**
        - Add feature importance analysis
        - Provide comparison with human expert annotations
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
    <p>Responsible AI Palm Tree Detection Web App | Created as part of the Responsible AI Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
