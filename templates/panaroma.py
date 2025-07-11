import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Load your images
img1 = cv2.imread("left.jpg")   # Reference (left)
img2 = cv2.imread("right.jpg")  # To be warped

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Use SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Use BFMatcher with ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract points
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)



matched_img = cv2.drawMatches(img1, kp1, img2, kp2,
                              good_matches,
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


cv2.imshow("Matches", matched_img)
cv2.imwrite("matches.jpg", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Homography via RANSAC
np.random.seed(42)  # For reproducibility
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)



# Step 5: Compute canvas size and translation
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
warped_corners = cv2.perspectiveTransform(corners_img2, H)

corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
all_corners = np.concatenate((warped_corners, corners_img1), axis=0)

[xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

T = np.array([[1, 0, -xmin],
              [0, 1, -ymin],
              [0, 0, 1]])

# Step 6: Warp and stitch
result = cv2.warpPerspective(img2, T @ H, (xmax - xmin, ymax - ymin))
result[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

# Optional: crop black borders
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
x, y, w, h = cv2.boundingRect(mask)
cropped = result[y:y+h, x:x+w]

# Save or show
cv2.imwrite("sift_stitched_fixed.jpg", cropped)
cv2.imshow("Fixed Stitch", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
