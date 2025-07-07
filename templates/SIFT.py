# Feature detection, matching, and RANSAC homography from scratch (no SIFT_create, no FLANN)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
# from skimage.metrics import structural_similarity as ssim


# ------------------ Step 1: Harris Corner Detection ------------------
def harris_corners(img, window_size=3, k=0.04, threshold=1e-5, max_pts=500):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  #(source, depth, dx, dy, kernelsize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = gaussian_filter(Ix * Ix, sigma=1)  #ntensity of gradient in x-direction (squared)
    Iyy = gaussian_filter(Iy * Iy, sigma=1)
    Ixy = gaussian_filter(Ix * Iy, sigma=1)  #product of gradients (for cross-terms)

    height, width = img.shape
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

# ------------------ Step 2: Patch Descriptors ------------------
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

# ------------------ Step 3: Descriptor Matching and RANSAC ------------------
def match_descriptors(desc1, desc2):
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    distances = cdist(desc1, desc2, metric='euclidean')
    matches = np.argmin(distances, axis=1)
    return matches

def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iter=1000):
    max_inliers = []
    best_H = None

    for _ in range(max_iter):
        if len(src_pts) < 4:
            break
        idx = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[idx]
        dst_sample = dst_pts[idx]

        H, _ = cv2.findHomography(src_sample, dst_sample, 0)
        if H is None:
            continue

        projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        error = np.linalg.norm(dst_pts - projected, axis=1)
        inliers = error < threshold

        if np.sum(inliers) > np.sum(max_inliers):
            max_inliers = inliers
            best_H = H

    return best_H, max_inliers

# ------------------ Step 4: Main Pipeline ------------------
img1 = cv2.imread('data/graf/3.ppm', cv2.IMREAD_GRAYSCALE)   # moving image
img2 = cv2.imread('data/graf/1.ppm', cv2.IMREAD_GRAYSCALE)   # reference image

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

# Detect keypoints
kp1 = harris_corners(img1)
kp2 = harris_corners(img2)

# Extract descriptors
desc1, kp1 = extract_descriptors(img1, kp1)
desc2, kp2 = extract_descriptors(img2, kp2)

# Match descriptors directly (1-to-1, no ratio test)
matches = match_descriptors(desc1, desc2)

if len(matches) == 0:
    raise RuntimeError("No matches found between descriptors.")

src_pts = kp1
dst_pts = kp2[matches]

# Estimate homography with RANSAC
src_pts_h = np.float32(src_pts).reshape(-1, 1, 2)
dst_pts_h = np.float32(dst_pts).reshape(-1, 1, 2)
H, inliers = cv2.findHomography(src_pts_h, dst_pts_h, cv2.RANSAC, 5.0)

if H is not None:
    registered_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.title("Image 1 (Moving)"); plt.imshow(img1, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Image 2 (Reference)"); plt.imshow(img2, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Warped Image 1"); plt.imshow(registered_img1, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()

    # Compute MSE
    mse = np.mean((img2.astype(np.float32) - registered_img1.astype(np.float32)) ** 2)
    print(f"MSE: {mse:.2f}")
else:
    print("Homography estimation failed.")

# Make sure both images are float in [0, 1]
img2_norm = img2.astype(np.float32) / 255.0
reg_img1_norm = registered_img1.astype(np.float32) / 255.0

# Compute SSIM (grayscale images)
# ssim_score, ssim_map = ssim(img2_norm, reg_img1_norm, full=True)

# print(f"SSIM: {ssim_score:.4f}"
