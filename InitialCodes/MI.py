import cv2
import numpy as np

# === Load Images ===
# Make sure both images are grayscale
# Reference (RGB)
rgb_img = cv2.imread('data/Pole_mounted/Visible/21-07-13_10-53-21_000005.png')
rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# Moving (Thermal)
thermal_img = cv2.imread('data/Pole_mounted/Ir/21-07-13_10-53-21_000005.png')
thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)

# Resize thermal image to match RGB (if needed)
thermal_gray = cv2.resize(thermal_gray, (rgb_gray.shape[1], rgb_gray.shape[0]))

# === Define motion model: TRANSLATION, EUCLIDEAN, AFFINE ===
warp_mode = cv2.MOTION_AFFINE  # or cv2.MOTION_TRANSLATION

# Initialize the warp matrix
warp_matrix = np.eye(2, 3, dtype=np.float32)

# Define stopping criteria: (max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

# === Apply ECC Alignment ===
(cc, warp_matrix) = cv2.findTransformECC(rgb_gray, thermal_gray, warp_matrix, warp_mode, criteria)

# Warp thermal image
aligned_thermal = cv2.warpAffine(thermal_img, warp_matrix, (rgb_img.shape[1], rgb_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# === Blend and Display ===
blended = cv2.addWeighted(rgb_img, 0.5, aligned_thermal, 0.5, 0)

cv2.imshow("Blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
