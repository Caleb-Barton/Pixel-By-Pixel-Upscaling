import cv2
import numpy as np
import os

SMALL_IMAGES_PATH = "freepik_images/small_pngs_3/"
LARGE_IMAGES_PATH = "freepik_images/large_pngs/"
OUTPUT_PATH = "data/kis_model/"

# It would be better if I mixed and matched data from each image, but this worked for the first try...
TRAIN_IMAGES = ["business.png", "camp.png", "city.png", "dinner.png", 
                "fish.png", "pets.png"]
TEST_IMAGES = ["owners.png", "night.png"]


# Returns a 2D numpy array of pixel values.
def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")
    return img


# Verify that the large image dimensions are exactly 3x the small image dimensions.
def verify_image_dimensions(small_img, large_img):
    small_h, small_w = small_img.shape
    large_h, large_w = large_img.shape
    
    if large_h != small_h * 3 or large_w != small_w * 3:
        raise ValueError(
            f"Dimension mismatch! Small: {small_h}x{small_w}, "
            f"Large: {large_h}x{large_w}, Expected: {small_h*3}x{small_w*3}"
        )


def extract_windows(small_img, large_img):
    """
    Extract all 3x3 windows from the small image and their corresponding
    3x3 output regions from the large image.
    
    Returns two lists:
    - input_windows: list of 3x3 input windows
    - output_windows: list of 3x3 output regions
    
    Skips homogeneous windows (where all 9 input pixels have the same value).
    """
    height, width = small_img.shape
    input_windows = []
    output_windows = []
    
    # Iterate through each pixel in the small image (skip borders)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Extract 3x3 window from small image centered at (y, x)
            input_window = small_img[y-1:y+2, x-1:x+2]
            
            # Check if window is homogeneous (all pixels same value)
            if np.all(input_window == input_window[0, 0]):
                # 99% chance of skipping
                if np.random.rand() < 0.95:
                    continue
            
            # Get corresponding 3x3 region from large image
            # The center pixel (y, x) in small image corresponds to a 3x3 block in large image
            large_y = y * 3
            large_x = x * 3
            output_window = large_img[large_y:large_y+3, large_x:large_x+3]
            
            # Flatten and add to lists
            input_windows.append(input_window.flatten())
            output_windows.append(output_window.flatten())
    return input_windows, output_windows


def process_image_pair(small_path, large_path):
    """
    Process a single pair of small and large images.
    Returns input and output window arrays for this image pair.
    """
    small_img = load_image(small_path)
    large_img = load_image(large_path)
    
    verify_image_dimensions(small_img, large_img)
    
    input_windows, output_windows = extract_windows(small_img, large_img)
    
    return input_windows, output_windows


def generate_dataset(image_list, small_dir, large_dir):
    """
    Generate dataset from a list of images.
    Returns two numpy arrays: inputs (N, 9) and outputs (N, 9).
    """
    all_inputs = []
    all_outputs = []
    
    for img_name in image_list:
        small_path = os.path.join(small_dir, img_name)
        large_path = os.path.join(large_dir, img_name)
        
        print(f"Processing {img_name}...")
        input_windows, output_windows = process_image_pair(small_path, large_path)
        
        all_inputs.extend(input_windows)
        all_outputs.extend(output_windows)
        print(f"  Added {len(input_windows)} windows")
    
    # Convert lists to numpy arrays
    inputs = np.array(all_inputs, dtype=np.float32)
    outputs = np.array(all_outputs, dtype=np.float32)
    
    return inputs, outputs


def main():
    """
    Main function to generate training and test datasets.
    Creates train_inputs.npy, train_outputs.npy, test_inputs.npy, test_outputs.npy
    in the OUTPUT_PATH directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    train_inputs, train_outputs = generate_dataset(
        TRAIN_IMAGES, SMALL_IMAGES_PATH, LARGE_IMAGES_PATH
    )
    
    # Generate test data
    print("\nGenerating test data...")
    test_inputs, test_outputs = generate_dataset(
        TEST_IMAGES, SMALL_IMAGES_PATH, LARGE_IMAGES_PATH
    )
    
    # Save datasets as .npy files
    print("\nSaving datasets...")
    np.save(os.path.join(OUTPUT_PATH, "train_inputs.npy"), train_inputs)
    np.save(os.path.join(OUTPUT_PATH, "train_outputs.npy"), train_outputs)
    np.save(os.path.join(OUTPUT_PATH, "test_inputs.npy"), test_inputs)
    np.save(os.path.join(OUTPUT_PATH, "test_outputs.npy"), test_outputs)
    
    # Print statistics about the generated datasets
    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    print(f"Training samples: {len(train_inputs)}")
    print(f"Test samples: {len(test_inputs)}")
    print(f"Input shape: {train_inputs.shape}")
    print(f"Output shape: {train_outputs.shape}")
    print(f"\nFiles saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()