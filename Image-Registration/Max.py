import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use a GUI backend
import matplotlib.pyplot as plt
import os


def load_pair(rgb_path, thermal_path):
    rgb = cv2.imread(rgb_path)  # shape: (H, W, 3)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
    
    thermal_resized = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))
    return rgb, thermal_resized

def per_channel_max_fusion(rgb, thermal):
    fused_channels = []
    for c in range(3):
        ch = rgb[:, :, c]
        fused = np.maximum(ch, thermal)
        fused_channels.append(fused.astype(np.uint8))
    return cv2.merge(fused_channels)


def show_images(rgb, thermal, fused_rgb):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(thermal, cmap='gray')
    plt.title("Thermal Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(fused_rgb, cv2.COLOR_BGR2RGB))
    plt.title(f"Fused RGB, alpha")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace with your image paths
    rgb_path = "/home/csio-isens/Desktop/abc/rgb/kettle.jpg"
    thermal_path = "/home/csio-isens/Desktop/abc/IR/kettle.jpg"

    rgb, thermal = load_pair(rgb_path, thermal_path)
    fused = per_channel_max_fusion(rgb, thermal)

    show_images(rgb, thermal, fused)

    # Save output
    os.makedirs("Max/output", exist_ok=True)
    cv2.imwrite("Max/output/kettle.jpg", fused)
    print("Fused image saved to output/kettle.jpg")

print(rgb.shape)
print(thermal.shape)




#REmarks:
#whitw appearing more
