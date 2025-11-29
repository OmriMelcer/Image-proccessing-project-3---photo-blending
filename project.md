# Laplacian Pyramid Blending Project

## Overview
This project implements a robust **Laplacian Pyramid Blending** algorithm to seamlessly merge two images. It allows users to blend an object from one image into a background image using a binary mask, creating a natural transition between the two.

## Features

### 1. Advanced Image Alignment
-   **Resolution Synchronization**: Automatically downsamples the object or mask to match the lower resolution of the two, ensuring pixel-perfect alignment.
-   **Global Scaling**: Upsamples the object to the nearest power of 2 (required for pyramid algorithms) and scales the background image by the exact same ratio to maintain relative size.
-   **Interactive Placement**: Users can click on the background image to define exactly where the object should be placed.
-   **Smart Cropping**: The background is cropped to match the object's dimensions, with automatic padding if the selected region extends beyond image boundaries.

### 2. RGB & Grayscale Support
-   **Full Color Blending**: The algorithm handles 3-channel RGB images by splitting them into Red, Green, and Blue channels, processing each independently, and recombining them.
-   **Grayscale Compatibility**: Automatically detects and handles 2D grayscale images.

### 3. Laplacian Pyramid Algorithm
-   **Custom Kernels**: Uses a 5x5 Gaussian kernel `[1, 4, 6, 4, 1]` for high-quality blurring and interpolation.
-   **Deep Blending**: The pyramid construction continues down to a **4x4 pixel** base layer. This depth allows for the blending of very low-frequency components (like overall lighting and color tone), resulting in significantly smoother transitions than standard implementations.
-   **Reconstruction**: The final image is reconstructed by iteratively upsampling and adding Laplacian levels.

### 4. Analysis Tools
-   **Fourier Magnitude Map**: Automatically generates and saves the Fourier Magnitude Spectrum of the final result to analyze its frequency content.

## How to Run

1.  Ensure you have the required dependencies installed:
    ```bash
    pip install numpy matplotlib scikit-image scipy
    ```
2.  Place your source images in the `photos/` directory:
    -   `image1.jpg` (Background)
    -   `image2.jpg` (Object to blend)
    -   `mask.jpg` (Binary mask for the object)
3.  Run the script:
    ```bash
    python main.py
    ```
    *(Or `uv run main.py` if using uv)*
4.  **Interact**: A window will open showing the background. Click the point where you want the center of the object to be.

## Output
The script generates the following files in the `photos/` directory:
-   `blended_snippet.jpg`: The blended object cutout.
-   `final_result.jpg`: The full resolution background with the object seamlessly blended in.
-   `final_magnitude.jpg`: The Log-Magnitude Fourier spectrum of the result.
-   `verify_composite.jpg`: A simple cut-and-paste composite for comparison.
