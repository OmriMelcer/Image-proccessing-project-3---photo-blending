# Project: Laplacian Pyramid Blending

## Overview
This project implements image blending using Laplacian Pyramids. The goal is to seamlessly blend two images together using a binary mask, ensuring smooth transitions for low frequencies (color/lighting) while maintaining sharp details for high frequencies.

## Requirements & Constraints

### Libraries
*   **Allowed**: `numpy`, `scipy`, `matplotlib`.
*   **Forbidden**: High-level computer vision libraries (e.g., OpenCV functions that directly implement pyramids or blending).
*   **Plotting**: **NO runtime viewing**. All plots must be closed and saved to disk.

### Input Specifications
*   **Images**: Two source images (A and B).
*   **Mask**: A binary mask indicating the blending region.
*   **Dimensions**:
    *   Images must be converted to **Grayscale**.
    *   Images must be cropped to have the **same dimensions**.
    *   Dimensions must be a **power of two** (e.g., 512x512, 1024x1024) to facilitate pyramid construction.
    *   The mask must be extended (padded with zeros) or cropped to match the images.

## Algorithm: Laplacian Pyramid Blending

### 1. Preprocessing
1.  Load Image A and Image B.
2.  Convert both to grayscale.
3.  Resize/Crop images to the nearest common power-of-two dimensions.
4.  Prepare the Mask to match these dimensions.

### 2. Pyramid Construction
*   **Kernel**: Use a **5x5 Gaussian kernel** (separable approximation: `[1, 4, 6, 4, 1] / 16`).
*   **Depth**: Build levels until the smallest dimension is approximately **16x16**.
*   **Structures**:
    *   **Gaussian Pyramid** for the Mask ($G_M$).
    *   **Laplacian Pyramid** for Image A ($L_A$).
    *   **Laplacian Pyramid** for Image B ($L_B$).
*   **Downsampling**: Each level is 1/4 the size of the previous (decimate by 2 in both X and Y).

### 3. Blending
For each level $k$:
$$L_{out}^{(k)} = G_M^{(k)} \cdot L_A^{(k)} + (1 - G_M^{(k)}) \cdot L_B^{(k)}$$
*   Perform element-wise weighted averaging using the mask's Gaussian level.

### 4. Reconstruction
1.  Start with the coarsest blended level.
2.  Upsample (expand) and add the next finer blended level.
3.  Repeat until the full resolution is reached.
4.  Clip values to valid range [0, 1] or [0, 255].

### 5. Analysis
*   Compute the **Fourier Magnitude** of the final blended result.
*   Save the visualization of the Fourier Magnitude.

## Deliverables
1.  **Implementation Code**: Python script/module implementing the algorithm.
2.  **Bad Example**: A demonstration of a case where the algorithm fails (e.g., structural misalignment, transparency issues).
