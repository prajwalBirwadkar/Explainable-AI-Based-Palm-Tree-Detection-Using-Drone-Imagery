"""
Model module for palm tree detection with Faster R-CNN.
Handles loading the model and performing inference.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PalmTreeDetector:
    """
    Palm tree detection model using Faster R-CNN with ResNet50 backbone.
    Provides methods for loading the model and performing detection.
    """
    
    def __init__(self, num_classes: int = 2, confidence_threshold: float = 0.7):
        """
        Initialize the palm tree detector.
        
        Args:
            num_classes: Number of classes (1 for palm trees + 1 for background)
            confidence_threshold: Threshold for detection confidence
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.model = None
        logger.info(f"Initializing model with {num_classes} classes, using device: {self.device}")
        
    def load_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the model from a PyTorch checkpoint (.pth file).
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            # Initialize model architecture
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            
            # Replace the classifier with a new one having the correct number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            
            # Load the model weights
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_model_from_h5(self, h5_path: str) -> None:
        """
        Load the model from an H5 file.
        
        Args:
            h5_path: Path to the H5 file containing model weights
        """
        logger.info(f"Loading model from H5: {h5_path}")
        try:
            # Initialize model architecture
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            
            # Replace the classifier with a new one having the correct number of classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            
            # Load weights from H5 file
            import h5py
            with h5py.File(h5_path, 'r') as f:
                for name, param in self.model.named_parameters():
                    if name in f:
                        param.data = torch.from_numpy(f[name][()])
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully from H5")
        except Exception as e:
            logger.error(f"Error loading model from H5: {e}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the image for the model.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Preprocessed image tensor
        """
        # Convert PIL image to tensor and normalize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(image)
        return img_tensor.to(self.device)

    def detect(self, image: Image.Image) -> Tuple[List[List[int]], List[float], List[int]]:
        """
        Detect palm trees in the given image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple containing (boxes, scores, labels)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_checkpoint() first.")
        
        logger.info("Running detection")
        img_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            prediction = self.model([img_tensor])
        
        # Extract results
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        indices = np.where(scores >= self.confidence_threshold)[0]
        filtered_boxes = boxes[indices].astype(int).tolist()
        filtered_scores = scores[indices].tolist()
        filtered_labels = labels[indices].tolist()
        
        logger.info(f"Detection completed. Found {len(filtered_boxes)} objects above threshold {self.confidence_threshold}")
        return filtered_boxes, filtered_scores, filtered_labels

    def get_feature_map_names(self) -> List[str]:
        """
        Get the names of feature maps for Grad-CAM visualization.
        
        Returns:
            List of feature map names
        """
        # For Faster R-CNN with ResNet50 + FPN, return relevant feature layer names
        return ["backbone.body.layer4"]

    def get_feature_map(self, name: str) -> nn.Module:
        """
        Get the specified feature map from the model.
        
        Args:
            name: Name of the feature map
            
        Returns:
            The feature map module
        """
        if name == "backbone.body.layer4":
            return self.model.backbone.body.layer4
        else:
            raise ValueError(f"Unknown feature map: {name}")
