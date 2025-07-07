import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(12)
# === Step 1: Load grayscale images ===
img_ref = cv2.imread("data/graf/1.ppm", cv2.IMREAD_GRAYSCALE)   # Reference
img_mov = cv2.imread("data/graf/3.ppm", cv2.IMREAD_GRAYSCALE)   # Moving

if img_ref is None or img_mov is None:
    raise FileNotFoundError("Could not load one or both images")

# === Step 2: FAST Keypoint Detection ===
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
kp_ref_cv = fast.detect(img_ref, None)
kp_mov_cv = fast.detect(img_mov, None)

# Convert to NumPy arrays of (x,y) for later
pts_ref = np.array([kp.pt for kp in kp_ref_cv], dtype=np.float32)
pts_mov = np.array([kp.pt for kp in kp_mov_cv], dtype=np.float32)

# === Step 3: BRIEF Descriptor Extraction ===
# (requires opencv-contrib-python)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)
kp_ref_cv, desc_ref = brief.compute(img_ref, kp_ref_cv)
kp_mov_cv, desc_mov = brief.compute(img_mov, kp_mov_cv)

# Refresh the point arrays in case some keypoints dropped
pts_ref = np.array([kp.pt for kp in kp_ref_cv], dtype=np.float32)
pts_mov = np.array([kp.pt for kp in kp_mov_cv], dtype=np.float32)

# === Step 4: Descriptor Matching (Hamming, no ratio test) ===
def match_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None:
        return np.array([], dtype=int)
    # Hamming distance for binary BRIEF
    D = cdist(desc1, desc2, metric='hamming')
    # for each descriptor in desc1, pick best match in desc2
    return np.argmin(D, axis=1)

matches = match_descriptors(desc_ref, desc_mov)
if matches.size == 0:
    raise RuntimeError("No matches found between descriptors")

# === Step 5: RANSAC-based Homography ===
def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iter=1000):
    best_H = None
    best_inliers = np.zeros(len(src_pts), dtype=bool)

    n = len(src_pts)
    for _ in range(max_iter):
        if n < 4:
            break
        # random 4-point sample
        idx = np.random.choice(n, 4, replace=False)
        src_s = src_pts[idx]
        dst_s = dst_pts[idx]

        H, _ = cv2.findHomography(src_s, dst_s, 0)
        if H is None:
            continue

        # cast to float32 for perspectiveTransform
        src32 = src_pts.astype(np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(src32, H).reshape(-1, 2)

        errs = np.linalg.norm(dst_pts - proj, axis=1)
        inliers = errs < threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers

# **Swap** so that we map moving → reference
src = pts_mov[matches]   # moving-image points
dst = pts_ref            # reference-image points

H, inliers = ransac_homography(src, dst)
if H is None:
    raise RuntimeError("RANSAC failed to find a valid homography")

# === Step 6: Warp moving into reference frame ===
aligned = cv2.warpPerspective(img_mov, H, (img_ref.shape[1], img_ref.shape[0]))

# === Display results ===
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Reference")
plt.imshow(img_ref, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Moving (Original)")
plt.imshow(img_mov, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Aligned Moving→Reference")
plt.imshow(aligned, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()