import skimage as sk
import skimage.io
import skimage.filters
import matplotlib.pyplot as plt
import numpy as np

def create_blurred_image(input_path, output_path, sigma=10):
    try:
        img = skimage.io.imread(input_path)
        # Blur the image to keep only low frequencies
        blurred = sk.filters.gaussian(img, sigma=sigma, channel_axis=-1)
        
        # Normalize to 0-1 if needed (gaussian usually returns float 0-1)
        if blurred.max() > 1.0:
            blurred /= 255.0
            
        plt.imsave(output_path, blurred)
        print(f"Saved blurred image to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Blur image2.jpg to create a 'plain' object
    create_blurred_image("photos/image2.jpg", "photos/plain_object.jpg", sigma=15)
