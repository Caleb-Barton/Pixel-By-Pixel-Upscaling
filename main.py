import cv2
import os

filename = "park"
# --- Configuration ---
input_filename = f'freepik_images\\large_pngs\\{filename}.png' 
output_filename_2 = f'freepik_images\\small_pngs_2\\{filename}.png' 
scale_factor = 0.5 # 0.5 = 50% scale (shrinking by a factor of 2)
# ---------------------

# 1. Check if the input file exists
if not os.path.exists(input_filename):
    print(f"Error: Input file not found at '{input_filename}'")
else:
    # 2. Read the image
    image = cv2.imread(input_filename)

    if image is None:
        print(f"Error: Could not read image from '{input_filename}'")
    else:
        # 3. Shrink the image using the scale factor
        # We set the explicit destination size (dsize) to None
        # and provide 'fx' (x-factor) and 'fy' (y-factor) instead.
        #
        # cv2.INTER_AREA is the recommended interpolation method for shrinking.
        resized_image = cv2.resize(image, 
                                   None, 
                                   fx=scale_factor, 
                                   fy=scale_factor, 
                                   interpolation=cv2.INTER_AREA)

        # 4. Save the resized image
        try:
            cv2.imwrite(output_filename_2, resized_image)
            print(f"Successfully shrunk image and saved to '{output_filename_2}'")
            # Optional: Print new dimensions
            print(f"Original dimensions (H, W): {image.shape[:2]}")
            print(f"New dimensions (H, W): {resized_image.shape[:2]}")
            
        except Exception as e:
            print(f"Error: Could not save image to '{output_filename_2}'. Reason: {e}")