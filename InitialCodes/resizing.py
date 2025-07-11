#upscaling - Improved Version

import numpy as np
import cv2
import os

def load_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    print(f"Loading image from: {image_path}")
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}. Check if it's a valid image file.")

    print(f"Image dimensions: {img.shape}")
    return img 

def apply_antialiasing(img, scale_x, scale_y):
    """Apply Gaussian blur for antialiasing when downscaling."""
    # Only apply antialiasing when downscaling
    if scale_x >= 1.0 and scale_y >= 1.0:
        return img
        
    sigma_x = max(0.5 / scale_x, 0.01)
    sigma_y = max(0.5 / scale_y, 0.01)

    # Kernel size: at least 3 and odd
    ksize_x = max(3, int(2 * np.ceil(2 * sigma_x) + 1))
    ksize_y = max(3, int(2 * np.ceil(2 * sigma_y) + 1))
    
    # Ensure odd kernel sizes
    ksize_x = ksize_x if ksize_x % 2 == 1 else ksize_x + 1
    ksize_y = ksize_y if ksize_y % 2 == 1 else ksize_y + 1

    return cv2.GaussianBlur(img, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

def resize_nearest_neighbour(img, new_h, new_w):
    """Resize using nearest neighbor interpolation."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    
    # Create output array with correct shape
    if channels == 1:
        output = np.zeros((new_h, new_w), dtype=img.dtype)
    else:
        output = np.zeros((new_h, new_w, channels), dtype=img.dtype)

    row_scale, col_scale = h / new_h, w / new_w

    for i in range(new_h):
        for j in range(new_w):
            src_i = min(int(i * row_scale), h - 1)
            src_j = min(int(j * col_scale), w - 1)
            output[i, j] = img[src_i, src_j]

    return output

def resize_bilinear(img, new_h, new_w):
    """Resize using bilinear interpolation."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    
    row_scale, col_scale = h / new_h, w / new_w
    
    # Apply antialiasing for downscaling
    if new_h < h or new_w < w:
        img = apply_antialiasing(img, 1/col_scale, 1/row_scale)

    # Create output array with correct shape
    if channels == 1:
        output = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        output = np.zeros((new_h, new_w, channels), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            src_i = i * row_scale
            src_j = j * col_scale

            i0, j0 = int(src_i), int(src_j)
            i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
            
            a = src_i - i0
            b = src_j - j0

            # Bilinear interpolation
            if channels == 1:
                output[i, j] = ((1 - a) * (1 - b) * img[i0, j0] + 
                               a * (1 - b) * img[i1, j0] + 
                               (1 - a) * b * img[i0, j1] + 
                               a * b * img[i1, j1])
            else:
                output[i, j] = ((1 - a) * (1 - b) * img[i0, j0] + 
                               a * (1 - b) * img[i1, j0] + 
                               (1 - a) * b * img[i0, j1] + 
                               a * b * img[i1, j1])
    
    return np.clip(output, 0, 255).astype(np.uint8)

def bicubic_kernel(t, a=-0.75):
    """Bicubic interpolation kernel."""
    abs_t = np.abs(t)
    abs_t2 = abs_t ** 2
    abs_t3 = abs_t ** 3

    result = np.zeros_like(t)
    mask1 = (abs_t <= 1)
    mask2 = (abs_t > 1) & (abs_t < 2)

    result[mask1] = ((a + 2) * abs_t3 - (a + 3) * abs_t2 + 1)[mask1]
    result[mask2] = (a * abs_t3 - 5*a * abs_t2 + 8*a * abs_t - 4*a)[mask2]

    return result

def resize_bicubic(img, new_h, new_w):
    """Resize using bicubic interpolation."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    row_scale, col_scale = h / new_h, w / new_w

    # Pad the image with edge replication
    if img.ndim == 2:
        img = np.pad(img, ((1, 2), (1, 2)), mode='edge')
    else:
        img = np.pad(img, ((1, 2), (1, 2), (0, 0)), mode='edge')

    # Apply antialiasing for downscaling
    if new_h < h or new_w < w:
        img = apply_antialiasing(img, 1/col_scale, 1/row_scale)

    # Create output array with correct shape
    if channels == 1:
        output = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        output = np.zeros((new_h, new_w, channels), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            x = j * col_scale
            y = i * row_scale
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))

            dxs = x - (x0 + np.array([-1, 0, 1, 2]))
            dys = y - (y0 + np.array([-1, 0, 1, 2]))
            kx = bicubic_kernel(dxs)
            ky = bicubic_kernel(dys)
            weights = np.outer(ky, kx)  # Shape (4, 4)

            if channels == 1:
                patch = img[y0:y0+4, x0:x0+4]
                output[i, j] = np.sum(patch * weights)
            else:
                for c in range(channels):
                    patch = img[y0:y0+4, x0:x0+4, c]
                    output[i, j, c] = np.sum(patch * weights)

    return np.clip(output, 0, 255).astype(np.uint8)

def sinc(x):
    """Normalized sinc function."""
    x = np.where(x == 0, 1e-10, x)  # avoid division by zero
    return np.sin(np.pi * x) / (np.pi * x)

def lanczos_kernel(x, a=3):
    """Lanczos windowed sinc kernel."""
    x = np.array(x)
    condition = np.abs(x) < a
    return np.where(condition, sinc(x) * sinc(x / a), 0)

def resize_lanczos(img, new_h, new_w, a=3):
    """Resize using Lanczos interpolation."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    row_scale = h / new_h
    col_scale = w / new_w

    # Pad the image to avoid boundary issues
    pad = a
    if img.ndim == 2:
        img = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    else:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Apply antialiasing when downscaling
    if new_h < h or new_w < w:
        img = apply_antialiasing(img, 1/col_scale, 1/row_scale)

    # Create correct output shape
    if channels == 1:
        output = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        output = np.zeros((new_h, new_w, channels), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            x = j * col_scale + pad
            y = i * row_scale + pad
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))

            dxs = np.arange(x0 - a + 1, x0 + a + 1) - x
            dys = np.arange(y0 - a + 1, y0 + a + 1) - y
            kx = lanczos_kernel(dxs, a)
            ky = lanczos_kernel(dys, a)
            weights = np.outer(ky, kx)

            if channels == 1:
                patch = img[y0 - a + 1:y0 + a + 1, x0 - a + 1:x0 + a + 1]
                output[i, j] = np.sum(patch * weights)
            else:
                for c in range(channels):
                    patch = img[y0 - a + 1:y0 + a + 1, x0 - a + 1:x0 + a + 1, c]
                    output[i, j, c] = np.sum(patch * weights)

    return np.clip(output, 0, 255).astype(np.uint8)

def resize_area(img, new_h, new_w):
    """Resize using area averaging (good for downscaling)."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    
    if channels == 1:
        output = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        output = np.zeros((new_h, new_w, channels), dtype=np.float32)

    scale_y = h / new_h
    scale_x = w / new_w

    for i in range(new_h):
        for j in range(new_w):
            y0 = int(i * scale_y)
            y1 = min(int((i + 1) * scale_y), h)
            x0 = int(j * scale_x)
            x1 = min(int((j + 1) * scale_x), w)
            
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:  # Handle edge case
               src_i = min(int(i * scale_y), h - 1)
               src_j = min(int(j * scale_x), w - 1)
               if channels == 1:
                   output[i, j] = img[src_i, src_j]
               else:
                   output[i, j] = img[src_i, src_j]
               continue
                
            if channels == 1:
                output[i, j] = np.mean(patch)
            else:
                for c in range(channels):
                    output[i, j, c] = np.mean(patch[..., c])

    return np.clip(output, 0, 255).astype(np.uint8)

def add_label(image, label):
    """Add a label to an image."""
    labeled = image.copy()
    # Ensure image is 3-channel for consistent labeling
    if labeled.ndim == 2:
        labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)
    
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(labeled, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return labeled

def pad_to_height(image, height):
    """Pad image vertically to match target height."""
    pad = height - image.shape[0]
    if pad > 0:
        return cv2.copyMakeBorder(image, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image

def get_valid_input():
    """Get and validate user input."""
    while True:
        try:
            image_path = input("Enter the path to the image: ").strip().strip('"\'')
            if not image_path:
                print("Please enter a valid path.")
                continue
            return image_path
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()

def get_resize_method():
    """Get and validate resize method."""
    valid_methods = ["nearest", "bilinear", "bicubic", "lanczos", "area"]
    while True:
        method = input(f"Enter interpolation method ({'/'.join(valid_methods)}): ").strip().lower()
        if method in valid_methods:
            return method
        print(f"Invalid method. Choose from: {', '.join(valid_methods)}")

def get_dimensions():
    """Get and validate output dimensions."""
    while True:
        try:
            dims = input("Enter output dimensions (height width): ").strip().split()
            if len(dims) != 2:
                print("Please enter exactly two numbers.")
                continue
            new_h, new_w = map(int, dims)
            if new_h <= 0 or new_w <= 0:
                print("Dimensions must be positive.")
                continue
            return new_h, new_w
        except ValueError:
            print("Please enter valid integers.")

if __name__ == "__main__":
    try:
        # Get user input with validation
        image_path = get_valid_input()
        img = load_image(image_path)
        
        method = get_resize_method()
        new_h, new_w = get_dimensions()
        
        print(f"Resizing image to {new_h}x{new_w} using {method} interpolation...")

        # Perform resizing based on method
        if method == "nearest":
            resized_img = resize_nearest_neighbour(img, new_h, new_w)
        elif method == "bilinear":
            resized_img = resize_bilinear(img, new_h, new_w)
        elif method == "bicubic":
            resized_img = resize_bicubic(img, new_h, new_w)
        elif method == "lanczos":
            while True:
                try:
                    a_input = input("Enter Lanczos parameter (default 3): ").strip()
                    a = int(a_input) if a_input else 3
                    if a <= 0:
                        print("Parameter must be positive.")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid integer.")
            print(f"Using Lanczos with parameter {a}")
            resized_img = resize_lanczos(img, new_h, new_w, a)
        elif method == "area":
            resized_img = resize_area(img, new_h, new_w)

        # Create labeled images for display
        h1, w1 = img.shape[:2]
        h2, w2 = resized_img.shape[:2]

        img_labeled = add_label(img, f"Original: {w1}x{h1}")
        resized_labeled = add_label(resized_img, f"Resized ({method}): {w2}x{h2}")

        # Create comparison display
        gap = 40
        max_height = max(img_labeled.shape[0], resized_labeled.shape[0])
        separator = np.zeros((max_height, gap, 3), dtype=np.uint8)

        img_labeled = pad_to_height(img_labeled, max_height)
        resized_labeled = pad_to_height(resized_labeled, max_height)

        combined = np.hstack((img_labeled, separator, resized_labeled))

        # Display results
        cv2.imshow("Image Resizing Comparison", combined)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Ask if user wants to save the result
        save = input("Save resized image? (y/n): ").strip().lower()
        if save in ['y', 'yes']:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_resized_{method}_{new_w}x{new_h}.jpg"
            cv2.imwrite(output_path, resized_img)
            print(f"Saved resized image as: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your input and try again.")