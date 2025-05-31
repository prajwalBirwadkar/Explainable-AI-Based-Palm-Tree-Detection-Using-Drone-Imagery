# Grad-CAM Implementation for Palm Tree Detection Visualization

import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load a pretrained classifier model (you can replace this with your trained palm tree classifier)
classifier = models.resnet18(pretrained=True)
classifier.eval()
target_layer = classifier.layer4[-1]

# Hooks to capture gradients and activations
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Grad-CAM function
def compute_gradcam(image_crop_pil):
    activations.clear()
    gradients.clear()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image_crop_pil).unsqueeze(0)
    output = classifier(input_tensor)
    class_idx = output.argmax(dim=1).item()

    classifier.zero_grad()
    output[0, class_idx].backward()

    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)

    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().detach().numpy()

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    return cam

def visualize_predictions_with_gradcam(image_path, predictions, threshold=0.8):
    from matplotlib import gridspec  # for flexible layout

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_overlay = image_np.copy()

    tree_count = 0
    YIELD_PER_TREE_KG = 85

    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score < threshold:
            continue

        tree_count += 1
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        crop_pil = image.crop((xmin, ymin, xmax, ymax))
        cam = compute_gradcam(crop_pil)
        cam_resized = cv2.resize(cam, (xmax - xmin, ymax - ymin))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        blend = (0.5 * image_overlay[ymin:ymax, xmin:xmax] + 0.5 * heatmap).astype(np.uint8)
        image_overlay[ymin:ymax, xmin:xmax] = blend

    # Draw bounding boxes and scores on the overlay image
    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(2, 2, height_ratios=[20, 1])

    # Original image
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.imshow(image_np)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')

    # Grad-CAM image with boxes
    ax2 = fig.add_subplot(spec[0, 1])
    ax2.imshow(image_overlay)

    for box, score in zip(predictions['boxes'], predictions['scores']):
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(xmin, ymin - 5, f"{score:.2f}", color='black', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5))
    ax2.set_title("Grad-CAM", fontsize=14)
    ax2.axis('off')

    # Bottom text
    estimated_production_kg = tree_count * YIELD_PER_TREE_KG
    subtitle = f"Estimated Trees = {tree_count} | Estimated Production = {estimated_production_kg} kg"

    ax_text = fig.add_subplot(spec[1, :])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, subtitle, fontsize=16, ha='center',
                 bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

# Example usage:
# test_image_path = "/path/to/test_image.jpg"
# predictions = model(image)  # Your model predictions
# visualize_predictions_with_gradcam(test_image_path, predictions)
