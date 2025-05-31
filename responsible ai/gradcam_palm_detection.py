# Grad-CAM Implementation Adapted for Palm Tree Detection Model

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import PalmTreeDetector  # Import the palm tree detector model

# Global variables to store activations and gradients
activations = []
gradients = []

# Forward hook to capture activations
def forward_hook(module, input, output):
    activations.append(output)

# Backward hook to capture gradients
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def setup_gradcam(model):
    """
    Set up Grad-CAM by registering hooks on the target layer
    
    Args:
        model: The PalmTreeDetector model
    """
    # Clear any existing hooks
    activations.clear()
    gradients.clear()
    
    # Get the target layer - using the model's built-in method
    feature_map_name = model.get_feature_map_names()[0]  # Get first feature map name
    target_layer = model.get_feature_map(feature_map_name)
    
    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    return target_layer

def compute_gradcam(model, image_crop, target_class=1):
    """
    Compute Grad-CAM for a cropped palm tree image
    
    Args:
        model: The PalmTreeDetector model
        image_crop: PIL Image of the cropped region
        target_class: Class to compute Grad-CAM for (1 = palm tree)
        
    Returns:
        Grad-CAM heatmap
    """
    # Clear previous activations and gradients
    activations.clear()
    gradients.clear()
    
    # Ensure model is in eval mode
    model.model.eval()
    
    # Preprocess the image
    input_tensor = model.preprocess_image(image_crop).unsqueeze(0)
    
    # Forward pass
    # Need to run the model in a modified way to get class scores
    # For Faster R-CNN, we need to extract class scores from the box predictor
    features = model.model.backbone(input_tensor)
    proposals, _ = model.model.rpn(features, [torch.Size(image_crop.size)])
    box_features = model.model.roi_heads.box_roi_pool(features, proposals, [input_tensor.shape[2:]])
    box_features = model.model.roi_heads.box_head(box_features)
    class_scores = model.model.roi_heads.box_predictor.cls_score(box_features)
    
    # If there are proposals, compute gradients
    if len(proposals[0]) > 0:
        # Get the score for the target class
        score = class_scores[:, target_class].sum()
        
        # Backward pass to get gradients
        model.model.zero_grad()
        score.backward()
        
        # Get the gradient and activation
        if len(gradients) > 0 and len(activations) > 0:
            grad = gradients[0]
            act = activations[0]
            
            # Global average pooling of gradients
            weights = grad.mean(dim=(2, 3), keepdim=True)
            
            # Weight the activations by the gradients
            cam = (weights * act).sum(dim=1, keepdim=True)
            
            # Apply ReLU to focus on features that have a positive influence
            cam = F.relu(cam)
            
            # Resize to the input image size
            cam = F.interpolate(
                cam, 
                size=image_crop.size[::-1],  # Reverse width, height for PyTorch interpolate
                mode='bilinear', 
                align_corners=False
            )
            
            # Normalize the CAM
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            
            return cam
    
    # If no proposals or gradients, return a blank heatmap
    return np.zeros(image_crop.size[::-1], dtype=np.float32)

def visualize_predictions_with_gradcam(model, image_path, threshold=0.7):
    """
    Visualize model predictions with Grad-CAM heatmaps
    
    Args:
        model: The PalmTreeDetector model
        image_path: Path to the image file
        threshold: Confidence threshold for detection
    """
    from matplotlib import gridspec  # For flexible layout
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_overlay = image_np.copy()
    
    # Set up Grad-CAM hooks
    setup_gradcam(model)
    
    # Get predictions
    boxes, scores, labels = model.detect(image)
    
    # Count trees and initialize yield estimation
    tree_count = len(scores)
    YIELD_PER_TREE_KG = 85
    
    # Process each detection
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
            
        xmin, ymin, xmax, ymax = box
        
        # Crop the image for Grad-CAM
        crop_pil = image.crop((xmin, ymin, xmax, ymax))
        
        # Compute Grad-CAM for the crop
        cam = compute_gradcam(model, crop_pil)
        
        # Resize CAM to match the box dimensions
        cam_resized = cv2.resize(cam, (xmax - xmin, ymax - ymin))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on the image
        blend = (0.5 * image_overlay[ymin:ymax, xmin:xmax] + 0.5 * heatmap).astype(np.uint8)
        image_overlay[ymin:ymax, xmin:xmax] = blend

    # Create visualization figure
    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(2, 2, height_ratios=[20, 1])
    
    # Original image with boxes
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.imshow(image_np)
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                linewidth=1, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(xmin, ymin - 5, f"{score:.2f}", color='black', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5))
    ax1.set_title("Original Image with Detections", fontsize=14)
    ax1.axis('off')
    
    # Grad-CAM visualization
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.imshow(image_overlay)
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                linewidth=1, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title("Grad-CAM Visualization", fontsize=14)
    ax2.axis('off')
    
    # Bottom text - yield estimation
    estimated_production_kg = tree_count * YIELD_PER_TREE_KG
    subtitle = f"Detected Palm Trees: {tree_count} | Estimated Production: {estimated_production_kg} kg"
    
    ax_text = fig.add_subplot(spec[1, :])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, subtitle, fontsize=16, ha='center',
                bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.show()

# Example usage:
"""
# Load your trained model
palm_detector = PalmTreeDetector(num_classes=2, confidence_threshold=0.7)
palm_detector.load_model_from_checkpoint('path/to/model_checkpoint.pth')

# Test image path
test_image_path = 'path/to/test_image.jpg'

# Visualize predictions with Grad-CAM
visualize_predictions_with_gradcam(palm_detector, test_image_path, threshold=0.7)
"""
