"""
Enhanced Grad-CAM implementations for palm tree detection
This file contains additional Grad-CAM functionality to extend your existing implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
from typing import List, Tuple, Dict, Optional, Union

def generate_class_activation_map(model, img, target_layer_name="backbone.body.layer4"):
    """
    Generate CAM for full image (not just detected regions)
    
    Args:
        model: The model object
        img: PIL Image
        target_layer_name: Layer to use for CAM generation
        
    Returns:
        np.ndarray: Class activation map
    """
    # Convert image to tensor
    img_tensor = model.preprocess_image(img).unsqueeze(0)
    
    # Variables to store activations and gradients
    activations = []
    gradients = []
    
    # Forward hook to get activations
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Backward hook to get gradients
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Get the target layer
    if target_layer_name == "backbone.body.layer4":
        target_layer = model.model.backbone.body.layer4
    else:
        raise ValueError(f"Unsupported layer: {target_layer_name}")
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        with torch.no_grad():
            features = model.model.backbone(img_tensor)
            
        # Get proposals from RPN
        proposals, _ = model.model.rpn(features, [torch.Size(img.size)])
        
        # Get box features
        box_features = model.model.roi_heads.box_roi_pool(features, proposals, [img_tensor.shape[2:]])
        box_features = model.model.roi_heads.box_head(box_features)
        
        # Get class scores
        class_scores = model.model.roi_heads.box_predictor.cls_score(box_features)
        
        if len(proposals[0]) > 0:
            # Get score for the palm tree class (usually class index 1)
            target_class = 1  # palm tree class
            score = class_scores[:, target_class].sum()
            
            # Backward pass
            model.model.zero_grad()
            score.backward()
            
            if len(gradients) > 0 and len(activations) > 0:
                # Get gradients and activations
                grads = gradients[0]
                acts = activations[0]
                
                # Global average pooling of gradients
                weights = grads.mean(dim=(2, 3), keepdim=True)
                
                # Weight activations by gradients
                cam = (weights * acts).sum(dim=1, keepdim=True)
                
                # Apply ReLU
                cam = F.relu(cam)
                
                # Resize to input image size
                cam = F.interpolate(
                    cam,
                    size=img.size[::-1],
                    mode='bilinear',
                    align_corners=False
                )
                
                # Convert to numpy and normalize
                cam = cam.squeeze().detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                
                return cam
    
    finally:
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
    
    # Return empty heatmap if failed
    return np.zeros(img.size[::-1], dtype=np.float32)

def create_interactive_gradcam_visualization(image, boxes, scores, labels, model):
    """
    Create an interactive Grad-CAM visualization for all detections
    
    Args:
        image: PIL Image
        boxes: List of bounding boxes
        scores: List of confidence scores
        labels: List of class labels
        model: Model object
        
    Returns:
        fig: Matplotlib figure for Streamlit
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Show original image
    ax.imshow(img_array)
    
    # Generate CAM for full image
    cam_full = generate_class_activation_map(model, image)
    
    # Create colored overlay for the full image
    heatmap_full = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
    heatmap_full = cv2.cvtColor(heatmap_full, cv2.COLOR_BGR2RGB)
    
    # Apply global CAM overlay with low alpha
    alpha_global = 0.3
    img_with_global_cam = img_array * (1 - alpha_global) + heatmap_full * alpha_global
    img_with_global_cam = img_with_global_cam.astype(np.uint8)
    
    # Display the image with global CAM
    ax.imshow(img_with_global_cam)
    
    # Draw boxes with higher opacity
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        
        # Draw box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                            linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        # Add text with better visibility
        ax.text(x1, y1 - 5, f"Palm Tree: {score:.2f}", 
                fontsize=10, color='white', 
                bbox=dict(facecolor='green', alpha=0.7))
    
    # Add title
    palm_count = len([s for s in scores if s > 0.5])  # Count palms with score > 0.5
    ax.set_title(f"Palm Tree Detection with Grad-CAM ({palm_count} trees identified)", fontsize=14)
    
    # Remove axis
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def generate_comparison_grid(image, boxes, scores, model, threshold=0.7):
    """
    Generate a grid of detection visualizations with Grad-CAM for each detected palm tree
    
    Args:
        image: PIL Image
        boxes: List of bounding boxes
        scores: List of confidence scores
        model: Model object
        threshold: Confidence threshold
        
    Returns:
        PIL Image with the comparison grid
    """
    # Filter detections based on threshold
    valid_detections = [(box, score) for box, score in zip(boxes, scores) if score >= threshold]
    
    if not valid_detections:
        return image
    
    # Determine grid size
    n_detections = len(valid_detections)
    cols = min(3, n_detections)
    rows = (n_detections + cols - 1) // cols
    
    # Create figure
    fig = plt.figure(figsize=(4*cols, 4*rows))
    
    # Process each detection
    for i, (box, score) in enumerate(valid_detections):
        x1, y1, x2, y2 = box
        
        # Crop region
        crop = image.crop((x1, y1, x2, y2))
        
        # Generate Grad-CAM for crop
        gradcam = generate_class_activation_map(model, crop)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (crop.width, crop.height))
        
        # Blend original and heatmap
        crop_array = np.array(crop)
        blend = (0.6 * crop_array + 0.4 * heatmap).astype(np.uint8)
        
        # Add to subplot
        plt.subplot(rows, cols, i+1)
        plt.imshow(blend)
        plt.title(f"Detection {i+1}: {score:.2f}", fontsize=10)
        plt.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    comparison_img = Image.open(buf)
    
    return comparison_img

def explain_model_decision(model, image, detection_idx, 
                          feature_layers=["backbone.body.layer4", "backbone.body.layer3"]):
    """
    Generate a detailed explanation of model decision with multiple feature layer visualizations
    
    Args:
        model: Model object
        image: PIL Image
        detection_idx: Index of the detection to explain
        feature_layers: List of feature layers to visualize
        
    Returns:
        dict: Dictionary of explanations and visualizations
    """
    # Run detection
    boxes, scores, labels = model.detect(image)
    
    if not boxes or detection_idx >= len(boxes):
        return {
            "success": False,
            "message": "No valid detection at specified index"
        }
    
    target_box = boxes[detection_idx]
    target_score = scores[detection_idx]
    
    # Crop the detection region
    x1, y1, x2, y2 = target_box
    crop = image.crop((x1, y1, x2, y2))
    
    # Generate explanations for different layers
    layer_visualizations = {}
    
    for layer_name in feature_layers:
        try:
            # Generate Grad-CAM for this layer
            cam = generate_class_activation_map(model, crop, target_layer_name=layer_name)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (crop.width, crop.height))
            
            # Blend original and heatmap
            crop_array = np.array(crop)
            blend = (0.6 * crop_array + 0.4 * heatmap).astype(np.uint8)
            
            # Save to layer visualizations
            layer_visualizations[layer_name] = Image.fromarray(blend)
            
        except Exception as e:
            layer_visualizations[layer_name] = f"Error: {str(e)}"
    
    # Return results
    return {
        "success": True,
        "detection_index": detection_idx,
        "confidence": target_score,
        "box": target_box,
        "crop": crop,
        "layer_visualizations": layer_visualizations
    }
