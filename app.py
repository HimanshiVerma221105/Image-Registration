from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import os
import base64
from werkzeug.utils import secure_filename
import requests
from PIL import Image
import io
import uuid
from scipy.ndimage import gaussian_filter   
from scipy.spatial.distance import cdist

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Your existing image processing functions
def apply_antialiasing(img, scale_x, scale_y):
    """Apply Gaussian blur for antialiasing when downscaling."""
    if scale_x >= 1.0 and scale_y >= 1.0:
        return img
        
    sigma_x = max(0.5 / scale_x, 0.01)

    sigma_y = max(0.5 / scale_y, 0.01)

    ksize_x = max(3, int(2 * np.ceil(2 * sigma_x) + 1))
    ksize_y = max(3, int(2 * np.ceil(2 * sigma_y) + 1))
    
    ksize_x = ksize_x if ksize_x % 2 == 1 else ksize_x + 1
    ksize_y = ksize_y if ksize_y % 2 == 1 else ksize_y + 1

    return cv2.GaussianBlur(img, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

def resize_nearest_neighbour(img, new_h, new_w):
    """Resize using nearest neighbor interpolation."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    
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
    
    if new_h < h or new_w < w:
        img = apply_antialiasing(img, 1/col_scale, 1/row_scale)

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

    if img.ndim == 2:
        img = np.pad(img, ((1, 2), (1, 2)), mode='edge')
    else:
        img = np.pad(img, ((1, 2), (1, 2), (0, 0)), mode='edge')

    if new_h < h or new_w < w:
        img = apply_antialiasing(img, 1/col_scale, 1/row_scale)

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
            weights = np.outer(ky, kx)

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
    x = np.where(x == 0, 1e-10, x)
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

    pad = a
    if img.ndim == 2:
        img = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    else:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    if new_h < h or new_w < w:
        img = apply_antialiasing(img, 1/col_scale, 1/row_scale)

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
    """Resize using area averaging."""
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
            if patch.size == 0:
                continue
                
            if channels == 1:
                output[i, j] = np.mean(patch)
            else:
                for c in range(channels):
                    output[i, j, c] = np.mean(patch[..., c])

    return np.clip(output, 0, 255).astype(np.uint8)

def image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def load_image_from_url(url):
    """Load image from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {str(e)}")

def load_image_from_path(path):
    """Load image from local path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img

def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iter=1000):
    np.random.seed(14) 
    
    best_H = None
    best_inliers = np.zeros(len(src_pts), dtype=bool)
    n = len(src_pts)

    for _ in range(max_iter):
        if n < 4:
            break
        idx = np.random.choice(n, 4, replace=False)
        src_s = src_pts[idx]
        dst_s = dst_pts[idx]

        H, _ = cv2.findHomography(src_s, dst_s, 0)
        if H is None:
            continue

        src32 = src_pts.astype(np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(src32, H).reshape(-1, 2)
        errs = np.linalg.norm(dst_pts - proj, axis=1)
        inliers = errs < threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers


# --- Registration functions ---
def register_orb(img_ref_color, img_mov_color):
    img_ref_gray = cv2.cvtColor(img_ref_color, cv2.COLOR_BGR2GRAY)
    img_mov_gray = cv2.cvtColor(img_mov_color, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    kp_ref_cv = fast.detect(img_ref_gray, None)
    kp_mov_cv = fast.detect(img_mov_gray, None)

    if len(kp_ref_cv) == 0 or len(kp_mov_cv) == 0:
        raise ValueError("No keypoints detected.")

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)
    kp_ref_cv, desc_ref = brief.compute(img_ref_gray, kp_ref_cv)
    kp_mov_cv, desc_mov = brief.compute(img_mov_gray, kp_mov_cv)

    if desc_ref is None or desc_mov is None:
        raise ValueError("Could not compute BRIEF descriptors.")

    pts_ref = np.array([kp.pt for kp in kp_ref_cv], dtype=np.float32)
    pts_mov = np.array([kp.pt for kp in kp_mov_cv], dtype=np.float32)

    def match_descriptors(desc1, desc2):
        D = cdist(desc1, desc2, metric='hamming')
        return np.argmin(D, axis=1)

    matches = match_descriptors(desc_ref, desc_mov)
    if matches.size == 0:
        raise ValueError("No matches found")

    src = pts_mov[matches]
    dst = pts_ref

    H, inliers = ransac_homography(src, dst)
    if H is None:
        raise ValueError("RANSAC failed to find a homography")

    aligned = cv2.warpPerspective(img_mov_color, H, (img_ref_color.shape[1], img_ref_color.shape[0]))
    return aligned

def harris_corners(img, window_size=3, k=0.04, threshold=1e-5, max_pts=500):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  #(source, depth, dx, dy, kernelsize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = gaussian_filter(Ix * Ix, sigma=1)  #ntensity of gradient in x-direction (squared)
    Iyy = gaussian_filter(Iy * Iy, sigma=1)
    Ixy = gaussian_filter(Ix * Iy, sigma=1)  #product of gradients (for cross-terms)

    height, width, _ = img.shape
    R = np.zeros((height, width))

    offset = window_size // 2
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Syy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()
            Sxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1].sum()

            det = Sxx * Syy - Sxy * Sxy
            trace = Sxx + Syy
            R[y, x] = det - k * trace * trace

    R[R < threshold * R.max()] = 0   #harris Corner response formula
    keypoints = np.argwhere(R)
    if len(keypoints) > max_pts:
        idx = np.argsort(R[tuple(keypoints.T)])[::-1][:max_pts]
        keypoints = keypoints[idx]
    return keypoints

def extract_descriptors(img, keypoints, patch_size=8):
    descriptors = []
    valid_kps = []
    half = patch_size // 2
    for y, x in keypoints:
        if y - half < 0 or y + half >= img.shape[0] or x - half < 0 or x + half >= img.shape[1]:
            continue
        patch = img[y - half:y + half, x - half:x + half].astype(np.float32)
        patch = patch - np.mean(patch)
        norm = np.linalg.norm(patch)
        if norm != 0:
            patch = patch / norm
        descriptors.append(patch.flatten())
        valid_kps.append((x, y))
    return np.array(descriptors), np.array(valid_kps)

def match_descriptors(desc1, desc2):
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    distances = cdist(desc1, desc2, metric='euclidean')
    matches = np.argmin(distances, axis=1)
    return matches


def register_sift(img_ref, img_mov):
    # Convert to grayscale for detection
    # gray_ref = cv2.cvtColor(img_ref_color, cv2.COLOR_BGR2GRAY)
    # gray_mov = cv2.cvtColor(img_mov_color, cv2.COLOR_BGR2GRAY)

    kp1 = harris_corners(img_ref)
    kp2 = harris_corners(img_mov)

    desc1, kp1 = extract_descriptors(img_ref, kp1)
    desc2, kp2 = extract_descriptors(img_mov, kp2)

    matches = match_descriptors(desc1, desc2)
    if len(matches) == 0:
        raise ValueError("No matches found.")

    src_pts = kp2[matches]
    dst_pts = kp1

    H, inliers = ransac_homography(np.float32(src_pts), np.float32(dst_pts))
    if H is None:
        raise ValueError("Homography estimation failed.")

    # Warp the color moving image (NOT grayscale)
    aligned = cv2.warpPerspective(img_mov, H, (img_ref.shape[1], img_ref.shape[0]))
    return aligned

@app.route('/stitch', methods=['POST'])
def stitch_image():
    try:
        data = request.json
        img1 = cv2.imdecode(np.frombuffer(base64.b64decode(data['img1'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(base64.b64decode(data['img2'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        direction = data.get('direction', 'horizontal')

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect features and compute descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Match features using Lowe's ratio test
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 4:
            return jsonify({'success': False, 'error': 'Not enough good matches for stitching.'})

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if direction == "vertical":
            corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        else:
            corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners = np.concatenate((warped_corners, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

        stitched = cv2.warpPerspective(img2, T @ H, (xmax - xmin, ymax - ymin))
        stitched[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

        # Crop black borders
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        cropped = stitched[y:y+h, x:x+w]

        result_b64 = image_to_base64(cropped)
        return jsonify({'success': True, 'result': result_b64})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Add these functions after your existing registration functions

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images"""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

import cv2
import numpy as np

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM) between two images"""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray, img2_gray = img1, img2
    
    # Convert to float32
    img1_gray = img1_gray.astype(np.float32)
    img2_gray = img2_gray.astype(np.float32)
    
    # Gaussian blur
    mu1 = cv2.GaussianBlur(img1_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_gray, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_gray * img2_gray, (11, 11), 1.5) - mu1_mu2

    # Constants for stability
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # SSIM map
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return np.mean(ssim_map)


# ===== ROUTES =====

@app.route('/')
def home():
    """Main page - serves home.html"""
    return render_template('home.html')

@app.route('/resize')
def resize_page():
    """Resize page - serves index.html for image resizing functionality"""
    return render_template('index.html')

@app.route('/register')
def register_page():
    """Registration page - serves register.html for image registration functionality"""
    return render_template('register.html')

# ===== API ENDPOINTS =====

from PIL import UnidentifiedImageError

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        image = None
        filename = None

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.ppm', '.pgm', '.pbm']:
                try:
                    pil_img = Image.open(filepath).convert("RGB")  # Convert to RGB
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='PNG')  # Save as PNG in memory
                    buffer.seek(0)
                    image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                   return jsonify({'error': f'Failed to load PPM/PGM/PBM image: {str(e)}'}), 400

            else:
                image = cv2.imread(filepath)


        elif 'url' in request.form and request.form['url']:
            url = request.form['url']
            image = load_image_from_url(url)
            filename = f"url_image_{uuid.uuid4().hex[:8]}.jpg"

        elif 'path' in request.form and request.form['path']:
            path = request.form['path']
            image = load_image_from_path(path)
            filename = os.path.basename(path)

        else:
            return jsonify({'error': 'No image source provided'}), 400

        if image is None:
            return jsonify({'error': 'Failed to load image'}), 400

        h, w = image.shape[:2]
        img_base64 = image_to_base64(image)

        return jsonify({
            'success': True,
            'image': img_base64,
            'width': w,
            'height': h,
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/resize_image', methods=['POST'])
def resize_image():
    """API endpoint for image resizing - renamed from /resize to avoid conflict"""
    try:
        data = request.json
        
        # Decode the base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        new_w = int(data['width'])
        new_h = int(data['height'])
        method = data['method']
        
        # Perform resizing
        if method == "nearest":
            resized_img = resize_nearest_neighbour(img, new_h, new_w)
        elif method == "bilinear":
            resized_img = resize_bilinear(img, new_h, new_w)
        elif method == "bicubic":
            resized_img = resize_bicubic(img, new_h, new_w)
        elif method == "lanczos":
            a = int(data.get('lanczos_a', 3))
            resized_img = resize_lanczos(img, new_h, new_w, a)
        elif method == "area":
            resized_img = resize_area(img, new_h, new_w)
        else:
            return jsonify({'error': 'Invalid resize method'}), 400
        
        # Convert to base64
        resized_base64 = image_to_base64(resized_img)
        
        return jsonify({
            'success': True,
            'resized_image': resized_base64,
            'original_size': f"{img.shape[1]}x{img.shape[0]}",
            'new_size': f"{new_w}x{new_h}",
            'method': method
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/register_image', methods=['POST'])
def register_image():
    """API endpoint for image registration"""
    try:
        data = request.json
        ref_data = data['ref']
        mov_data = data['mov']
        method = data['method']

        # Decode base64 images
        ref_img = cv2.imdecode(np.frombuffer(base64.b64decode(ref_data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        mov_img = cv2.imdecode(np.frombuffer(base64.b64decode(mov_data.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        if ref_img is None or mov_img is None:
            return jsonify({'success': False, 'error': 'Could not decode images.'})

        if method == 'orb':
            result = register_orb(ref_img, mov_img)
        elif method == 'sift':
            result = register_sift(ref_img, mov_img)
        else:
            return jsonify({'success': False, 'error': 'Invalid method.'})

        metrics = {
             'mse': float(calculate_mse(ref_img, result)),
             'psnr': float(calculate_psnr(ref_img, result)),
             'ssim': float(calculate_ssim(ref_img, result))
          }
        # Encode result to base64
        _, buf = cv2.imencode('.jpg', result)
        result_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
        return jsonify({'success': True, 'result': result_b64, 'metrics': metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stitch')
def stitch_page():
    return render_template('stitch.html')

@app.route('/save', methods=['POST'])
def save_image():
    try:
        data = request.json
        
        # Decode the base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Generate filename
        filename = f"resized_{data['method']}_{data['width']}x{data['height']}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Save image
        cv2.imwrite(filepath, img)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
