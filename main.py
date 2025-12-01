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
    pyramyd_dict = {"img_1_geucian_pyr":img_1_geucian_pyr,"img_2_geucian_pyr":img_2_geucian_pyr,"mask_geucian_pyr":mask_geucian_pyr,"img1_laplacian_pyr":img1_laplacian_pyr,"img2_laplacian_pyr":img2_laplacian_pyr,"mixture_laplacian_pyr":mixture_laplacian_pyr}
    return final_cut, pyramyd_dict


def upsample (img):
    # recieves an image and returns an upsampled by 4 image
    # uses convolution with the geucian kernel in both axis
    # uses the upsampled image to calculate the next image in the geucian pyr
    # returns the upsampled image and the next image in the geucian pyr
    kernel_geucian_row = np.array([[1,4,6,4,1]])/16
    kernel_geucian_col = np.array([[1],[4],[6],[4],[1]])/16
    
    # Pad the small image with edge values to handle boundaries correctly
    # This avoids black borders ('fill') and artifacts ('symm' on sparse array)
    padded_img = np.pad(img, ((1,1), (1,1)), mode='edge')
    
    # Upsample (insert zeros)
    up = np.zeros((padded_img.shape[0]*2, padded_img.shape[1]*2))
    up[::2, ::2] = padded_img
    
    # Convolve
    up = sp.signal.convolve2d(up, kernel_geucian_row, mode='same', boundary='fill', fillvalue=0)
    up = sp.signal.convolve2d(up, kernel_geucian_col, mode='same', boundary='fill', fillvalue=0)
    
    # Crop the center to remove the padding effects
    # We added 1 pixel on each side of small img -> 2 pixels on each side of upsampled img
    # So we crop [2:-2, 2:-2]
    return up[2:-2, 2:-2] * 4

def create_hybrid_image(img1, img2):
    ratio = 1
    row_geucian_kernel = np.array([[1,4,6,4,1]])/16 
    col_geucian_kernel = np.array([[1],[4],[6],[4],[1]])/16 
    # Convolve with self to expand kernel (increase blur)
    # mode='full' allows the kernel to grow (e.g. 5x5 -> 9x9 -> 17x17 -> 33x33)
    # We loop 1 time to get a moderate kernel size (9x9)
    for _ in range(1):
        row_geucian_kernel = sp.signal.convolve2d(row_geucian_kernel, row_geucian_kernel, mode='full')
        col_geucian_kernel = sp.signal.convolve2d(col_geucian_kernel, col_geucian_kernel, mode='full')
    
    # Normalize to ensure brightness stays constant
    row_geucian_kernel /= row_geucian_kernel.sum()
    col_geucian_kernel /= col_geucian_kernel.sum()
    far_image_blurry = geucian_layer(img1, row_geucian_kernel, col_geucian_kernel, ratio)
    close_image_detail = img2-geucian_layer(img2, row_geucian_kernel, col_geucian_kernel, ratio)
    return far_image_blurry+close_image_detail
    

    
def geucian_layer(img, kernel_geucian_row = np.array([[1,4,6,4,1]])/16, kernel_geucian_col = np.array([[1],[4],[6],[4],[1]])/16, down_sample_ratio = 2):
    # recieves an image and returns a downsampled by 4 image
    # uses convolution with the geucian kernel in both axis
    # uses the downsampled image to calculate the next image in the geucian pyr
    # returns the downsampled image and the next image in the geucian pyr
    img_blured = sp.signal.convolve2d(img, kernel_geucian_row, mode='same', boundary='fill', fillvalue=0)
    img_blured = sp.signal.convolve2d(img_blured, kernel_geucian_col, mode='same', boundary='fill', fillvalue=0)
    img_downsampled = np.zeros((img.shape[0]//down_sample_ratio, img.shape[1]//down_sample_ratio))
    img_downsampled = img_blured[::down_sample_ratio, ::down_sample_ratio]
    return img_downsampled

def turn__2_rgb(img):
    if img.shape[2] == 3:
        return sk.color.rgb2gray(img)
    if img.shape[2] == 4:
        return sk.color.rgb2gray(sk.color.rgba2rgb(img))
    return img
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

def save_specific_pyrs(pyr_blue, pyr_green, pyr_red,name):
    for i in range(len(pyr_blue)):
        # for j in range(i):
        #     pyr_red[i] = upsample(pyr_red[i])
        #     pyr_green[i] = upsample(pyr_green[i])
        #     pyr_blue[i] = upsample(pyr_blue[i])
        pyr_red[i] = sk.transform.resize(pyr_red[i], (256, 256), order=0, anti_aliasing=False)
        pyr_green[i] = sk.transform.resize(pyr_green[i], (256, 256), order=0, anti_aliasing=False)
        pyr_blue[i] = sk.transform.resize(pyr_blue[i], (256, 256), order=0, anti_aliasing=False)
        pyr_red[i] = np.clip(pyr_red[i], 0, 1)
        pyr_green[i] = np.clip(pyr_green[i], 0, 1)
        pyr_blue[i] = np.clip(pyr_blue[i], 0, 1)
        to_save = np.dstack((pyr_red[i], pyr_green[i], pyr_blue[i]))
        plt.imsave(f"photos/pyramids/{name}_{i}.jpg", to_save)
    
def save_all_pyrs(pyrs_blue: dict , pyrs_green: dict, pyrs_red: dict):
    for name in pyrs_blue.keys():
        save_specific_pyrs(pyrs_blue[name], pyrs_green[name], pyrs_red[name], name)

def main():
    # 1. Load Images
    try:
        img1 = load_image("photos/image1.jpg") # Background
        img2 = load_image("photos/grid.jpg") # Object
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
    # center1 = (img1.shape[1]//2, img1.shape[0]//2)
    
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
    
    # Save Magnitude of the Bad Blend (Cut and Paste)
    if composite.ndim == 3:
        gray_composite = sk.color.rgb2gray(composite)
    else:
        gray_composite = composite
    save_magnitude_spectrum(gray_composite, "photos/composite_magnitude.jpg")
    print("Bad Blend (Composite) Magnitude saved to photos/composite_magnitude.jpg")
    
    # 5. Run Laplacian Pyramid Blending
    print("Running Laplacian Pyramid Blending...")
    blended_snippet_red , pyrs_red= create_laplacian_geocian(img2[:, :, 0], img1_cropped[:, :, 0], mask[:, :, 0])
    blended_snippet_green , pyrs_green = create_laplacian_geocian(img2[:, :, 1], img1_cropped[:, :, 1], mask[:, :, 1])
    blended_snippet_blue , pyrs_blue = create_laplacian_geocian(img2[:, :, 2], img1_cropped[:, :, 2], mask[:, :, 2])
    blended_snippet = np.dstack((blended_snippet_red, blended_snippet_green, blended_snippet_blue))
    blended_snippet = np.clip(blended_snippet, 0, 1)
    plt.imsave("photos/blended_snippet.jpg", blended_snippet)
    save_all_pyrs(pyrs_blue, pyrs_green, pyrs_red)
    
    # 6. Plant Snippet back into Original Image
    final_result = plant_snippet(img1, blended_snippet, center1)
    plt.imsave("photos/final_result.jpg", final_result)
    print("Final full image saved to photos/final_result.jpg")
    
    #6.2 create a hybrid image of the snippets and save it to photos/hybrid_snippet.jpg
    grey_img1 = turn__2_rgb(img1_cropped)
    grey_img2 = turn__2_rgb(img2)
    hybrid_grey = create_hybrid_image(grey_img2, grey_img1)
    hybrid_snippet_red = create_hybrid_image(img2[:, :, 0], img1_cropped[:, :, 0])
    hybrid_snippet_green = create_hybrid_image(img2[:, :, 1], img1_cropped[:, :, 1])
    hybrid_snippet_blue = create_hybrid_image(img2[:, :, 2], img1_cropped[:, :, 2])
    hybrid_snippet_red = np.clip(hybrid_snippet_red, 0, 1)
    hybrid_snippet_green = np.clip(hybrid_snippet_green, 0, 1)
    hybrid_snippet_blue = np.clip(hybrid_snippet_blue, 0, 1)
    hybrid_grey = np.clip(hybrid_grey, 0, 1)
    hybrid_snippet = np.dstack((hybrid_snippet_red, hybrid_snippet_green, hybrid_snippet_blue))
    plt.imsave("photos/hybrid_grey.jpg", hybrid_grey, cmap='gray')
    plt.imsave("photos/hybrid_snippet.jpg", hybrid_snippet)
    print("Hybrid snippet saved to photos/hybrid_snippet.jpg")

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
 at 'center', handling edge cases.
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
