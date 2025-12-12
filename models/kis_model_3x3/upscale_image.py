import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import os

# Model configuration (must match training)
INPUT_SIZE = 9
OUTPUT_SIZE = 9
HIDDEN_LAYERS = [81, 81]
DEFAULT_MODEL_PATH = "models/kis_model/best_model.pth"


class UpscaleNet(nn.Module):
    """
    Fully connected neural network for pixel upscaling.
    Must match the architecture used during training.
    """
    def __init__(self, input_size, hidden_layers, output_size):
        super(UpscaleNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_model(model_path):
    """
    Load the trained model from a .pth file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UpscaleNet(INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    print(f"Running on: {device}")
    
    return model, device


def upscale_image(model, device, image_path):
    """
    Upscale an entire image using the trained neural network.
    Saves the result with '_ANN' appended to the filename.
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    height, width = img.shape
    print(f"Upscaling image: {image_path}")
    print(f"Original size: {width}x{height}")
    
    # Create output image (3x larger)
    output_img = np.zeros((height * 3, width * 3), dtype=np.uint8)
    
    # Process each pixel (skip borders)
    with torch.no_grad():
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Extract 3x3 window
                input_window = img[y-1:y+2, x-1:x+2].flatten()
                
                # Normalize to [0, 1]
                input_tensor = torch.FloatTensor(input_window / 255.0).unsqueeze(0).to(device)
                
                # Get model prediction
                output_tensor = model(input_tensor)
                
                # Convert back to pixel space [0, 255]
                output_window = (output_tensor.squeeze().cpu().numpy() * 255.0).reshape(3, 3)
                
                # Place in output image
                out_y = y * 3
                out_x = x * 3
                output_img[out_y:out_y+3, out_x:out_x+3] = np.clip(output_window, 0, 255).astype(np.uint8)
    
    # Generate output filename
    base_name = os.path.splitext(image_path)[0]
    ext = os.path.splitext(image_path)[1]
    output_path = f"{base_name}_ANN{ext}"
    
    # Save upscaled image
    cv2.imwrite(output_path, output_img)
    print(f"Upscaled image saved to: {output_path}")
    print(f"Output size: {width*3}x{height*3}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python upscale_image.py <image_path> [model_path]")
        print(f"  If model_path is not provided, uses: {DEFAULT_MODEL_PATH}")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    # Load model and upscale image
    model, device = load_model(model_path)
    upscale_image(model, device, image_path)


if __name__ == "__main__":
    main()