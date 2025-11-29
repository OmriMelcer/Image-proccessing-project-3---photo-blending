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

### 1. Preprocessing & Interaction
1.  Load Image A, Image B, and the Mask.
2.  Convert all to grayscale.
3.  **Interactive Selection**: Display Image A and Image B to the user. The user clicks a point on each image to define the **center** where the mask should be applied.
4.  **Snippet Extraction**:
    *   Determine the bounding box of the Mask.
    *   Extract a snippet from Image A centered at the user's click, matching the Mask's dimensions.
    *   Extract a snippet from Image B centered at the user's click, matching the Mask's dimensions.
    *   *Note*: Ensure the snippet dimensions are a power of two (pad/crop the mask and snippets if necessary) for the pyramid.

### 2. Pyramid Construction
*   **Kernel**: Use a **5x5 Gaussian kernel** (separable approximation: `[1, 4, 6, 4, 1] / 16`).
*   **Depth**: Build levels until the smallest dimension is approximately **16x16**.
*   **Structures**:
    *   **Gaussian Pyramid** for the Mask Snippet ($G_M$).
    *   **Laplacian Pyramid** for Image A Snippet ($L_A$).
    *   **Laplacian Pyramid** for Image B Snippet ($L_B$).
*   **Downsampling**: Each level is 1/4 the size of the previous (decimate by 2 in both X and Y).

### 3. Blending
For each level $k$:
$$L_{out}^{(k)} = G_M^{(k)} \cdot L_A^{(k)} + (1 - G_M^{(k)}) \cdot L_B^{(k)}$$
*   Perform element-wise weighted averaging using the mask's Gaussian level.

### 4. Reconstruction & Planting
1.  Collapse the blended Laplacian pyramid to form the **Blended Snippet**.
2.  **Planting**: Paste the **Blended Snippet** back into **Image A** at the original selected coordinates.
3.  Clip values to valid range [0, 1] or [0, 255].

### 5. Analysis
*   Compute the **Fourier Magnitude** of the final blended result.
*   Save the visualization of the Fourier Magnitude.

## Deliverables
1.  **Implementation Code**: Python script/module implementing the algorithm.
2.  **Bad Example**: A demonstration of a case where the algorithm fails (e.g., structural misalignment, transparency issues).
