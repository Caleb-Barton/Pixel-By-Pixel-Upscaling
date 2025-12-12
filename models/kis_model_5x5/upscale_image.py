import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
import os

# Model configuration (must match training)
INPUT_SIZE = 25  # 5x5 window
OUTPUT_SIZE = 9  # 3x3 output
HIDDEN_LAYERS = [100, 135, 81]
DEFAULT_MODEL_PATH = "models/kis_model_5x5/best_model.pth"


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
            layers.append(nn.LeakyReLU(0.01))
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


def upscale_channel(model, device, channel):
    """
    Upscale a single channel using the trained neural network with 5x5 input windows.
    Returns the upscaled channel.
    """
    height, width = channel.shape
    
    # Create output channel (3x larger)
    output_channel = np.zeros((height * 3, width * 3), dtype=np.uint8)
    
    # Process each pixel (skip 2-pixel border for 5x5 windows)
    with torch.no_grad():
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                # Extract 5x5 window
                input_window = channel[y-2:y+3, x-2:x+3]
                
                # Check if middle 3x3 region is homogeneous
                middle_region = input_window[1:4, 1:4]
                if np.all(middle_region == middle_region[0, 0]):
                    # All middle pixels are the same - output same value
                    homogeneous_value = middle_region[0, 0]
                    output_window = np.full((3, 3), homogeneous_value, dtype=np.uint8)
                else:
                    # Use model to predict
                    input_flat = input_window.flatten()
                    
                    # Normalize to [0, 1]
                    input_tensor = torch.FloatTensor(input_flat / 255.0).unsqueeze(0).to(device)
                    
                    # Get model prediction
                    output_tensor = model(input_tensor)
                    
                    # Convert back to pixel space [0, 255]
                    output_window = (output_tensor.squeeze().cpu().numpy() * 255.0).reshape(3, 3)
                    output_window = np.clip(output_window, 0, 255).astype(np.uint8)
                
                # Place in output channel
                out_y = y * 3
                out_x = x * 3
                output_channel[out_y:out_y+3, out_x:out_x+3] = output_window
    
    return output_channel


def upscale_image(model, device, image_path):
    """
    Upscale an entire image (color or grayscale) using the trained neural network.
    Saves the result with '_ANN_5x5' appended to the filename.
    """
    # Load image with all channels (including alpha if present)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Determine image type
    if len(img.shape) == 2:
        # Grayscale image
        print(f"Upscaling grayscale image: {image_path}")
        height, width = img.shape
        channels = [img]
        has_alpha = False
    elif img.shape[2] == 3:
        # Color image (BGR)
        print(f"Upscaling color image: {image_path}")
        height, width, _ = img.shape
        channels = [img[:, :, i] for i in range(3)]
        has_alpha = False
    elif img.shape[2] == 4:
        # Color image with alpha (BGRA)
        print(f"Upscaling color image with transparency: {image_path}")
        height, width, _ = img.shape
        channels = [img[:, :, i] for i in range(4)]
        has_alpha = True
    else:
        print(f"Error: Unsupported image format")
        return
    
    print(f"Original size: {width}x{height}")
    
    # Upscale each channel
    upscaled_channels = []
    for i, channel in enumerate(channels):
        channel_name = ["Blue", "Green", "Red", "Alpha"][i] if len(channels) > 1 else "Grayscale"
        print(f"Processing {channel_name} channel...")
        upscaled_channel = upscale_channel(model, device, channel)
        upscaled_channels.append(upscaled_channel)
    
    # Combine channels
    if len(upscaled_channels) == 1:
        output_img = upscaled_channels[0]
    else:
        output_img = np.stack(upscaled_channels, axis=2)
    
    # Generate output filename
    base_name = os.path.splitext(image_path)[0]
    ext = os.path.splitext(image_path)[1]
    output_path = f"{base_name}_ANN_5x5{ext}"
    
    # Save upscaled image
    cv2.imwrite(output_path, output_img)
    print(f"Upscaled image saved to: {output_path}")
    print(f"Output size: {width*3}x{height*3}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python upscale_image_5x5.py <image_path> [model_path]")
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