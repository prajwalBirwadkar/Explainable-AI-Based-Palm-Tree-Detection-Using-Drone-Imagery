"""
Streamlit web application for palm tree detection with Grad-CAM explainability.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import os
import sys
import logging
from typing import Tuple, List, Dict, Optional
import time

# Import custom modules
from model import PalmTreeDetector
from gradcam import explain_detection, compare_original_and_gradcam
# Import enhanced Grad-CAM visualizations
from gradcam_enhancements import (
    create_interactive_gradcam_visualization,
    generate_comparison_grid,
    explain_model_decision
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_CHECKPOINT_PATH = "MODEL_CHECKPOINTS/model_epoch_14.pth"
MODEL_WEIGHTS_PATH = "MODEL_WEIGHTS/palm_tree_model.h5"
CONFIDENCE_THRESHOLD = 0.7
NUM_CLASSES = 2  # Background + Palm Tree

def load_model() -> PalmTreeDetector:
    """
    Load the palm tree detection model
    
    Returns:
        PalmTreeDetector instance
    """
    model = PalmTreeDetector(num_classes=NUM_CLASSES, confidence_threshold=CONFIDENCE_THRESHOLD)
    
    try:
        # First try to load from checkpoint
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_CHECKPOINT_PATH)
        if os.path.exists(model_path):
            model.load_model_from_checkpoint(model_path)
            return model
        
        # If checkpoint not found, try H5 weights
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_WEIGHTS_PATH)
        if os.path.exists(weights_path):
            model.load_model_from_h5(weights_path)
            return model
            
        raise FileNotFoundError("Neither model checkpoint nor weights file found")
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        logger.error(f"Error loading model: {e}")
        raise
        
def detect_palm_trees(uploaded_file, model: PalmTreeDetector) -> Tuple[Image.Image, List[List[int]], List[float], List[int]]:
    """
    Detect palm trees in the uploaded image
    
    Args:
        uploaded_file: File uploaded by the user
        model: Palm tree detection model
        
    Returns:
        Tuple containing (image, boxes, scores, labels)
    """
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run detection
    boxes, scores, labels = model.detect(image)
    
    return image, boxes, scores, labels

def draw_detection_boxes(image: Image.Image, boxes: List[List[int]], scores: List[float]) -> Image.Image:
    """
    Draw bounding boxes and scores on the image
    
    Args:
        image: Original image
        boxes: Detected bounding boxes
        scores: Confidence scores
        
    Returns:
        Image with bounding boxes
    """
    # Create a copy to avoid modifying the original
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each box and score
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        
        # Draw the bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
        
        # Draw the score
        text = f"Palm Tree: {score:.2f}"
        # Use textbbox instead of deprecated textsize
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        text_width = right - left
        text_height = bottom - top
        draw.rectangle([(x1, y1 - text_height - 4), (x1 + text_width, y1)], fill="green")
        draw.text((x1, y1 - text_height - 2), text, fill="white", font=font)
    
    return result

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Responsible AI: Palm Tree Detection",
        page_icon="🌴",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
    }
    .explanation-text {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.markdown('<h1 class="main-header">🌴 Responsible AI: Palm Tree Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="explanation-text">
    <p>This application uses a <b>Faster R-CNN</b> model to detect palm trees in images. 
    It incorporates <b>explainability</b> through Grad-CAM visualizations that highlight 
    the regions of the image that influenced the model's decision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Settings & Information</h2>', unsafe_allow_html=True)
        
        # Model settings
        st.subheader("Model Configuration")
        confidence_threshold = st.slider("Detection Confidence Threshold", 
                                         min_value=0.1, max_value=1.0, 
                                         value=CONFIDENCE_THRESHOLD, step=0.05)
        
        # Explainability settings
        st.subheader("Explainability")
        st.markdown("""
        Grad-CAM (Gradient-weighted Class Activation Mapping) produces visual explanations for 
        the model's decisions by highlighting the important regions in the image for prediction.
        
        **How it works:**
        1. It uses the gradients flowing into the final convolutional layer to understand which regions are important
        2. Areas highlighted in red have the strongest influence on the model's detection
        """)
        
        # Performance note
        st.subheader("Performance Note")
        st.info("The first detection may take a few seconds while the model loads.")
        
        # Fairness & Bias information
        st.subheader("Fairness & Bias")
        st.markdown("""
        **Potential Biases:**
        - Geographic bias: The model may perform better on palm trees from regions well-represented in the training data
        - Lighting bias: Detection performance may vary under different lighting conditions
        - Species bias: Some palm tree species may be detected more accurately than others
        
        This application is designed to make AI decision-making more transparent through explanations.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load the model
            try:
                with st.spinner("Loading model..."):
                    model = load_model()
                
                # Perform detection
                with st.spinner("Running detection..."):
                    image, boxes, scores, labels = detect_palm_trees(uploaded_file, model)
                    
                    if len(boxes) == 0:
                        st.warning("No palm trees detected in this image.")
                        st.image(image, caption="Original Image", use_container_width=True)
                    else:
                        # Draw boxes on the image
                        result_image = draw_detection_boxes(image, boxes, scores)
                        st.image(result_image, caption="Detection Results", use_container_width=True)
                        
                        # Display detection stats
                        st.success(f"Detected {len(boxes)} palm trees!")
                        
                        # Create a table of detections
                        detection_data = []
                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            detection_data.append({
                                "Detection #": i+1,
                                "Confidence": f"{score:.2f}",
                                "Box Coordinates": f"{box}"
                            })
                        
                        if detection_data:
                            st.markdown('<h3 class="sub-header">Detection Details</h3>', unsafe_allow_html=True)
                            st.table(detection_data)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error in main flow: {e}", exc_info=True)
            finally:
                pass
    
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
                    finally:
                        pass
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
