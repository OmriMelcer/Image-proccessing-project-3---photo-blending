import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.transform
import scipy as sp 


def get_user_point(img, title="Click center point"):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    pts = plt.ginput(1, timeout=-1)
    plt.close()
    if not pts:
        raise ValueError("No point selected!")
    return pts[0] # (x, y) -> (col, row)

def resize_to_power_of_two(img):
    """
    Resizes image to the nearest power of 2 dimensions.
    """
    rows, cols = img.shape
    new_rows = 2 ** int(np.round(np.log2(rows)))
    new_cols = 2 ** int(np.round(np.log2(cols)))
    return sk.transform.resize(img, (new_rows, new_cols), anti_aliasing=True)

def crop_around_center(img, center, size):
    """
    Crops a region of 'size' (rows, cols) from 'img' centered at 'center' (x, y).
    """
    rows, cols ,colors = size
    c_x, c_y = center # x is col, y is row
    
    start_row = int(c_y - rows // 2)
    start_col = int(c_x - cols // 2)
    
    # Simple slicing with clamping
    end_row = start_row + rows
    end_col = start_col + cols
    
    cropped = img[max(0, start_row):min(img.shape[0], end_row), 
                  max(0, start_col):min(img.shape[1], end_col)]
                  
    # Pad if necessary to match exact size (if point was near edge)
    # Check if dimensions match (ignoring channels for now)
    if cropped.shape[:2] != (rows, cols):
        print(f"Warning: Cropped size {cropped.shape} does not match target {size}. Padding...")
        
        # Create padded array with correct dimensions (2D or 3D)
        if img.ndim == 3:
            padded = np.zeros((rows, cols, img.shape[2]), dtype=img.dtype)
        else:
            padded = np.zeros((rows, cols), dtype=img.dtype)
            
        # Calculate where to paste the cropped region into the padded array
        # We need to align the crop relative to the target window
        
        # Original requested start (might be negative)
        req_start_row = int(c_y - rows // 2)
        req_start_col = int(c_x - cols // 2)
        
        # Offset in the destination array is how much we chopped off from the start
        dst_r_start = max(0, -req_start_row)
        dst_c_start = max(0, -req_start_col)
        
        # End indices in destination
        dst_r_end = dst_r_start + cropped.shape[0]
        dst_c_end = dst_c_start + cropped.shape[1]
        
        padded[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = cropped
        return padded
        
    return cropped

def create_laplacian_geocian (img1, img2, mask):
    img_1_geucian_pyr =[img1]
    img_2_geucian_pyr =[img2]
    mask_geucian_pyr = [mask]
    img1_laplacian_pyr =[]
    img2_laplacian_pyr =[]
    mixture_laplacian_pyr =[]
    while True:
        img1_downsampled = geucian_layer(img_1_geucian_pyr[-1])
        img_1_geucian_pyr.append(img1_downsampled)
        img2_downsampled = geucian_layer(img_2_geucian_pyr[-1])
        img_2_geucian_pyr.append(img2_downsampled)
        mask_downsampled = geucian_layer(mask_geucian_pyr[-1])
        mask_geucian_pyr.append(mask_downsampled)
        img1_laplacian_pyr.append(img_1_geucian_pyr[-2]-upsample(img_1_geucian_pyr[-1]))
        img2_laplacian_pyr.append(img_2_geucian_pyr[-2]-upsample(img_2_geucian_pyr[-1]))
        if(img_1_geucian_pyr[-1].shape[0] <= 4 or img_1_geucian_pyr[-1].shape[1] <= 4):
            img1_laplacian_pyr.append(img_1_geucian_pyr[-1])
            img2_laplacian_pyr.append(img_2_geucian_pyr[-1])
            break        
    for i in range(1,len(img1_laplacian_pyr)+1):
        mixture_laplacian_pyr.append(img1_laplacian_pyr[-i]*mask_geucian_pyr[-i]+img2_laplacian_pyr[-i]*(1-mask_geucian_pyr[-i]))
    final_cut = mixture_laplacian_pyr[0]
    for i in range(1,len(mixture_laplacian_pyr)):
        final_cut = mixture_laplacian_pyr[i]+upsample(final_cut)
    return final_cut


def upsample (img):
    # recieves an image and returns an upsampled by 4 image
    # uses convolution with the geucian kernel in both axis
    # uses the upsampled image to calculate the next image in the geucian pyr
    # returns the upsampled image and the next image in the geucian pyr
    kernel_geucian_row = np.array([[1,4,6,4,1]])/16
    kernel_geucian_col = np.array([[1],[4],[6],[4],[1]])/16
    up = np.zeros((img.shape[0]*2, img.shape[1]*2))
    up[::2, ::2] = img
    up = sp.signal.convolve2d(up, kernel_geucian_row, mode='same', boundary='fill', fillvalue=0)
    up = sp.signal.convolve2d(up, kernel_geucian_col, mode='same', boundary='fill', fillvalue=0)
    return up*4
    
    
def geucian_layer(img):
    # recieves an image and returns a downsampled by 4 image
    # uses convolution with the geucian kernel in both axis
    # uses the downsampled image to calculate the next image in the geucian pyr
    # returns the downsampled image and the next image in the geucian pyr
    
    kernel_geucian_row = np.array([[1,4,6,4,1]])/16
    kernel_geucian_col = np.array([[1],[4],[6],[4],[1]])/16
    img_blured = sp.signal.convolve2d(img, kernel_geucian_row, mode='same', boundary='fill', fillvalue=0)
    img_blured = sp.signal.convolve2d(img_blured, kernel_geucian_col, mode='same', boundary='fill', fillvalue=0)
    img_downsampled = np.zeros((img.shape[0]//2, img.shape[1]//2))
    img_downsampled = img_blured[::2, ::2]
    return img_downsampled

def load_image(path):
    img = plt.imread(path)
    # Normalize to 0-1 if integer
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
        
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        if img.shape[2] == 3:
            return img # RGB
        elif img.shape[2] == 4:
            return sk.color.rgba2rgb(img) # Drop alpha
        else:
            raise ValueError(f"Unsupported image format: {img.shape}")
    else:
        raise ValueError(f"Unsupported image dimension: {img.ndim}")

def main():
    # 1. Load Images
    try:
        img1 = load_image("photos/image1.jpg") # Background
        img2 = load_image("photos/image2.jpg") # Object
        mask = load_image("photos/mask.jpg")   # Mask
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return
    #fix bad white couple of line that are somehow in my top of the mask.
    mask[0:10] = 0
    
    # 2. Sync img2 and mask (Lowest Resolution Rule)
    # The user states they cover the same area but have different resolutions.
    # We must downsample the larger one to match the smaller one's dimensions exactly.
    
    # Check if img2 is larger than mask (in terms of pixel count)
    if img2.shape[0] * img2.shape[1] > mask.shape[0] * mask.shape[1]:
        print(f"Downsampling img2 {img2.shape} to match mask {mask.shape}")
        img2 = sk.transform.resize(img2, mask.shape, anti_aliasing=True)
    # Check if mask is larger than img2
    elif mask.shape[0] * mask.shape[1] > img2.shape[0] * img2.shape[1]:
        print(f"Downsampling mask {mask.shape} to match img2 {img2.shape}")
        mask = sk.transform.resize(mask, img2.shape, anti_aliasing=True)
        
    # 3. Global Upsampling to Power of 2
    # Calculate target dimensions (Next Power of 2)
    # User requested "upsample", so we use ceil()
    rows, cols ,colors = img2.shape
    new_rows = 2 ** int(np.ceil(np.log2(rows)))
    new_cols = 2 ** int(np.ceil(np.log2(cols)))
    
    # Calculate Scaling Ratios
    ratio_r = new_rows / rows
    ratio_c = new_cols / cols
    
    print(f"Upsampling Object {img2.shape} -> ({new_rows}, {new_cols}) (Ratio: {ratio_r:.2f}, {ratio_c:.2f})")
    
    # Resize Object and Mask
    img2 = sk.transform.resize(img2, (new_rows, new_cols), anti_aliasing=True)
    mask = sk.transform.resize(mask, (new_rows, new_cols), anti_aliasing=True)
    
    # [SAFETY] Force mask to match img2 exactly
    if mask.shape != img2.shape:
        mask = sk.transform.resize(mask, img2.shape, anti_aliasing=True)
        
    # Resize Background (img1) by the SAME ratio
    img1_new_rows = int(img1.shape[0] * ratio_r)
    img1_new_cols = int(img1.shape[1] * ratio_c)
    print(f"Upsampling Background {img1.shape} -> ({img1_new_rows}, {img1_new_cols}) to match scale.")
    img1 = sk.transform.resize(img1, (img1_new_rows, img1_new_cols), anti_aliasing=True)
    
    print(f"Normalized Object/Mask Size: {img2.shape}")

    # 4. Interactive Selection & Background Crop
    print("Please select the center point on Image 1 (Background)...")
    center1 = get_user_point(img1, "Click where to place the object")
    
    img1_cropped = crop_around_center(img1, center1, img2.shape)
    
    # 5. Save & Verify
    plt.imsave("photos/aligned_img1.jpg", img1_cropped, cmap='gray')
    plt.imsave("photos/aligned_img2.jpg", img2, cmap='gray')
    plt.imsave("photos/aligned_mask.jpg", mask, cmap='gray')
    print("Aligned images saved to photos/")
    # 4. Verify Alignment
    masked_bg = img1_cropped * (1 - mask)
    masked_obj = img2 * mask
    composite = masked_bg + masked_obj
    plt.imsave("photos/verify_composite.jpg", composite, cmap='gray')
    print("Composite verification saved to photos/verify_composite.jpg")
    
    # 5. Run Laplacian Pyramid Blending
    print("Running Laplacian Pyramid Blending...")
    blended_snippet_red = create_laplacian_geocian(img2[:, :, 0], img1_cropped[:, :, 0], mask[:, :, 0])
    blended_snippet_green = create_laplacian_geocian(img2[:, :, 1], img1_cropped[:, :, 1], mask[:, :, 1])
    blended_snippet_blue = create_laplacian_geocian(img2[:, :, 2], img1_cropped[:, :, 2], mask[:, :, 2])
    blended_snippet = np.dstack((blended_snippet_red, blended_snippet_green, blended_snippet_blue))
    blended_snippet = np.clip(blended_snippet, 0, 1)
    plt.imsave("photos/blended_snippet.jpg", blended_snippet)
    
    # 6. Plant Snippet back into Original Image
    final_result = plant_snippet(img1, blended_snippet, center1)
    plt.imsave("photos/final_result.jpg", final_result)
    print("Final full image saved to photos/final_result.jpg")
    
    # 7. Magnitude Map (Fourier Transform)
    # Convert to grayscale for magnitude map
    if final_result.ndim == 3:
        gray_result = sk.color.rgb2gray(final_result)
    else:
        gray_result = final_result
    save_magnitude_spectrum(gray_result, "photos/final_magnitude.jpg")
    print("Magnitude map saved to photos/final_magnitude.jpg")

def plant_snippet(full_img, snippet, center):
    """
    Pastes the 'snippet' into 'full_img' centered at 'center', handling edge cases.
    Inverse of crop_around_center.
    """
    img = full_img.copy()
    rows, cols = snippet.shape[:2] # Handle RGB
    c_x, c_y = center
    
    # Calculate target bounds on the full image
    req_start_row = int(c_y - rows // 2)
    req_start_col = int(c_x - cols // 2)
    
    # Calculate valid intersection with the image
    start_row = max(0, req_start_row)
    start_col = max(0, req_start_col)
    end_row = min(img.shape[0], req_start_row + rows)
    end_col = min(img.shape[1], req_start_col + cols)
    
    # Calculate source bounds from the snippet (handling padding)
    src_r_start = max(0, -req_start_row)
    src_c_start = max(0, -req_start_col)
    
    copy_rows = end_row - start_row
    copy_cols = end_col - start_col
    
    src_r_end = src_r_start + copy_rows
    src_c_end = src_c_start + copy_cols
    
    # Paste
    img[start_row:end_row, start_col:end_col] = snippet[src_r_start:src_r_end, src_c_start:src_c_end]
    return img

def save_magnitude_spectrum(img, path):
    """
    Computes and saves the log-magnitude of the Fourier Transform.
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize to 0-1 for saving
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    plt.imsave(path, magnitude_spectrum, cmap='gray')

if __name__ == "__main__":
    main()
