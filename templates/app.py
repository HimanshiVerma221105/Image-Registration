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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        image = None
        filename = None
        
        if 'file' in request.files and request.files['file'].filename:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = cv2.imread(filepath)
            
        elif 'url' in request.form and request.form['url']:
            # Handle URL
            url = request.form['url']
            image = load_image_from_url(url)
            filename = f"url_image_{uuid.uuid4().hex[:8]}.jpg"
            
        elif 'path' in request.form and request.form['path']:
            # Handle local path
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

@app.route('/resize', methods=['POST'])
def resize_image():
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
    app.run(debug=True)