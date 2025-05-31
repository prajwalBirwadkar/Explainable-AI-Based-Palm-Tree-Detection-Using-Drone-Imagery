"""
Grad-CAM implementation for Faster R-CNN model to visualize heatmaps of detected palm trees.
This module provides explainability by showing what features the model is focusing on.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradCAM:
    """
    Grad-CAM implementation for Faster R-CNN to provide explainability
    """
    
    def __init__(self, model, feature_layer_name):
        """
        Initialize Grad-CAM with model and target layer
        
        Args:
            model: Faster R-CNN model
            feature_layer_name: Name of the layer to use for Grad-CAM
        """
        self.model = model
        self.model.eval()
        self.feature_layer = None
        self.target_layer_name = feature_layer_name
        self.gradients = None
        self.activations = None
        
        # Get the specified feature map
        if feature_layer_name == "backbone.body.layer4":
            self.feature_layer = model.backbone.body.layer4
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Initialized Grad-CAM with target layer: {feature_layer_name}")
        
    def _register_hooks(self):
        """
        Register forward and backward hooks to get activations and gradients
        """
        if self.feature_layer is None:
            logger.error("Feature layer is None. Cannot register hooks.")
            return
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register the hooks
        self.forward_handle = self.feature_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.feature_layer.register_full_backward_hook(backward_hook)
        
    def __del__(self):
        """Clean up hooks when the object is destroyed"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def _compute_gradcam(self, target_box_index):
        """
        Compute Grad-CAM heatmap for specified box index
        
        Args:
            target_box_index: Index of the box to compute Grad-CAM for
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        # Compute gradients with respect to the target box
        if self.gradients is None or self.activations is None:
            logger.error("Gradients or activations are None")
            return None
            
        # Compute weights as global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted activation map
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)
        
        # Normalize the CAM
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # Convert to numpy and resize to image dimensions
        cam = cam.cpu().detach().numpy().squeeze()
        return cam

    def generate_cam(self, image: Image.Image, boxes: List[List[int]], 
                    scores: List[float], detection_index: int = 0) -> np.ndarray:
        """
        Generate Grad-CAM visualization for a specific detection
        
        Args:
            image: Original image 
            boxes: Detected bounding boxes
            scores: Confidence scores for each box
            detection_index: Index of the detection to visualize
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        if detection_index >= len(boxes):
            logger.error(f"Detection index {detection_index} is out of range (only {len(boxes)} detections)")
            return None
        
        logger.info(f"Generating Grad-CAM for detection index {detection_index}")
        
        # Convert PIL image to tensor
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0)
        
        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Get the box coordinates for the target detection
        target_box = boxes[detection_index]
        
        # Forward pass
        self.model.zero_grad()
        with torch.enable_grad():
            outputs = self.model([img_tensor])
            
            # Get predicted box and score
            predicted_boxes = outputs[0]['boxes']
            predicted_scores = outputs[0]['scores']
            
            # Find the corresponding prediction that matches our filtered box
            matched_idx = None
            for i, box in enumerate(predicted_boxes):
                iou = self._compute_iou(box.detach().cpu().numpy(), target_box)
                if iou > 0.8:  # High IoU threshold to ensure we get the right box
                    matched_idx = i
                    break
            
            if matched_idx is None:
                logger.error("Failed to match the filtered detection with raw predictions")
                return None
                
            # Compute gradients with respect to the target box score
            target_score = predicted_scores[matched_idx]
            target_score.backward()
            
            # Compute the Grad-CAM
            cam = self._compute_gradcam(matched_idx)
            
            # Resize CAM to match the image dimensions
            img_width, img_height = image.size
            cam_resized = cv2.resize(cam, (img_width, img_height))
            
            return cam_resized
    
    def _compute_iou(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Convert to [x1, y1, x2, y2] format if not already
        box1 = np.array(box1).astype(float)
        box2 = np.array(box2).astype(float)
        
        # Intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Intersection area
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
        
    def visualize(self, image: Image.Image, boxes: List[List[int]], 
                 scores: List[float], gradcam_output: np.ndarray, 
                 detection_index: int = 0, alpha: float = 0.5) -> Image.Image:
        """
        Generate a visualization of Grad-CAM overlaid on the original image
        
        Args:
            image: Original image
            boxes: Detected bounding boxes
            scores: Confidence scores for each box
            gradcam_output: Grad-CAM heatmap
            detection_index: Index of the detection to visualize
            alpha: Transparency for heatmap overlay
            
        Returns:
            PIL Image with Grad-CAM and bounding box
        """
        logger.info("Creating Grad-CAM visualization")
        
        # Convert PIL image to numpy array
        img_np = np.array(image)
        
        # Generate heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_output), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Overlay heatmap on original image
        superimposed_img = heatmap * alpha + img_np * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        # Draw bounding box
        if detection_index < len(boxes):
            box = boxes[detection_index]
            cv2.rectangle(superimposed_img, 
                         (box[0], box[1]), 
                         (box[2], box[3]), 
                         (0, 255, 0), 2)
            
            # Add score text
            score = scores[detection_index]
            cv2.putText(superimposed_img, 
                       f"Palm Tree: {score:.2f}", 
                       (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            
        # Convert back to PIL Image
        return Image.fromarray(superimposed_img)

def compare_original_and_gradcam(original_img: Image.Image, gradcam_img: Image.Image, 
                                save_path: Optional[str] = None) -> None:
    """
    Creates a side-by-side comparison of original image and Grad-CAM visualization
    
    Args:
        original_img: Original image
        gradcam_img: Grad-CAM overlay image
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gradcam_img)
    plt.title("Grad-CAM Visualization")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison visualization to {save_path}")
    
    plt.close()

def explain_detection(image: Image.Image, boxes: List[List[int]], scores: List[float], 
                     model, detection_index: int = 0, 
                     feature_layer_name: str = "backbone.body.layer4") -> Image.Image:
    """
    High-level function to generate Grad-CAM explanation for a detection
    
    Args:
        image: Original image
        boxes: Detected bounding boxes
        scores: Confidence scores for each box
        model: The Faster R-CNN model
        detection_index: Index of the detection to explain
        feature_layer_name: Name of the feature layer to use for Grad-CAM
        
    Returns:
        Visualization with Grad-CAM overlay
    """
    if not boxes or detection_index >= len(boxes):
        logger.warning("No valid detections to explain")
        return image
    
    try:
        # Initialize Grad-CAM
        gradcam = GradCAM(model=model, feature_layer_name=feature_layer_name)
        
        # Generate Grad-CAM heatmap
        heatmap = gradcam.generate_cam(image, boxes, scores, detection_index)
        
        if heatmap is None:
            logger.error("Failed to generate Grad-CAM heatmap")
            return image
        
        # Create visualization with heatmap and bounding box
        visualization = gradcam.visualize(image, boxes, scores, heatmap, detection_index)
        
        return visualization
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return image
