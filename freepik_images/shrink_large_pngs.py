import cv2
import os

def shrink_image_3(filename, factor=3):
    input_filename = f'large_pngs\\{filename}.png' 
    output_filename_3 = f'small_pngs_{factor}\\{filename}.png' 
    scale_factor = 1/factor

    if not os.path.exists(input_filename):
        print(f"Error: Input file not found at '{input_filename}'")
    else:
        image = cv2.imread(input_filename)

        if image is None:
            print(f"Error: Could not read image from '{input_filename}'")
        else:
            resized_image = cv2.resize(image, 
                                    None, 
                                    fx=scale_factor, 
                                    fy=scale_factor, 
                                    interpolation=cv2.INTER_AREA)

            try:
                cv2.imwrite(output_filename_3, resized_image)
                print(f"Successfully shrunk image and saved to '{output_filename_3}'")
                print(f"Original dimensions (H, W): {image.shape[:2]}")
                print(f"New dimensions (H, W): {resized_image.shape[:2]}")
                
            except Exception as e:
                print(f"Error: Could not save image to '{output_filename_3}'. Reason: {e}")

images = ["business", "camp", "city", "dinner", "fish", "icons", "night", "owners", "park", "pets",]

for filename in images:
    shrink_image_3(filename, factor=2)
    shrink_image_3(filename, factor=3)