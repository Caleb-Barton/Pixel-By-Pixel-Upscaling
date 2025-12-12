import numpy as np
import cv2
import sys
import os

def baseline_upscale(input_window):
    """
    Baseline upscaling algorithm.
    
    Input: 9-element array representing a 3x3 window [TL, T, TR, ML, M, MR, BL, B, BR]
    Output: 9-element array representing the upscaled 3x3 output
    
    Formula: output_pixel = (2/3 * center_pixel) + (1/3 * corresponding_input_pixel)
    Exception: center output pixel = center input pixel
    """
    input_window = input_window.reshape(3, 3)
    center_value = input_window[1, 1]
    
    output_window = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                # Center pixel stays the same
                output_window[i, j] = center_value
            else:
                # Other pixels: 2/3 center + 1/3 corresponding
                output_window[i, j] = (2/3) * center_value + (1/3) * input_window[i, j]
    
    return output_window.flatten()


def test_model():
    """
    Test the baseline model on the test dataset and report MSE.
    """
    # Load test data
    test_inputs = np.load("data/kis_model/test_inputs.npy")
    test_outputs = np.load("data/kis_model/test_outputs.npy")
    
    print("Testing baseline model...")
    print(f"Test samples: {len(test_inputs)}")
    
    # Generate predictions
    predictions = np.array([baseline_upscale(input_window) for input_window in test_inputs])
    
    # Calculate MSE
    mse = np.mean((predictions - test_outputs) ** 2)
    
    print(f"\nBaseline Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
    
    return mse


def upscale_image(image_path):
    """
    Upscale an entire image using the baseline model.
    Saves the result with '_FM' appended to the filename.
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
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Extract 3x3 window
            input_window = img[y-1:y+2, x-1:x+2].flatten()
            
            # Get upscaled output
            output_window = baseline_upscale(input_window).reshape(3, 3)
            
            # Place in output image
            out_y = y * 3
            out_x = x * 3
            output_img[out_y:out_y+3, out_x:out_x+3] = np.clip(output_window, 0, 255).astype(np.uint8)
    
    # Generate output filename
    base_name = os.path.splitext(image_path)[0]
    ext = os.path.splitext(image_path)[1]
    output_path = f"{base_name}_FM{ext}"
    
    # Save upscaled image
    cv2.imwrite(output_path, output_img)
    print(f"Upscaled image saved to: {output_path}")
    print(f"Output size: {width*3}x{height*3}")


def main():
    if len(sys.argv) > 1:
        # Image path provided - upscale the image
        image_path = sys.argv[1]
        upscale_image(image_path)
    else:
        # No arguments - run test
        test_model()


if __name__ == "__main__":
    main()