import numpy as np
import matplotlib.pyplot as plt

def create_grid_pattern(output_path, size=(256, 256), step=16):
    img = np.zeros(size, dtype=np.float64)
    
    # Create vertical lines
    img[:, ::step] = 1.0
    
    # Create horizontal lines
    img[::step, :] = 1.0
    
    # Make lines a bit thicker (optional, but helps visibility)
    img[:, 1::step] = 1.0
    img[1::step, :] = 1.0
    
    # Save as RGB (3 channels) to match other images
    img_rgb = np.dstack((img, img, img))
    
    plt.imsave(output_path, img_rgb)
    print(f"Saved grid pattern to {output_path}")

if __name__ == "__main__":
    create_grid_pattern("photos/grid.jpg", step=16)
