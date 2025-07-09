import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# from skimage.metrics import structural_similarity as ssim

def weighted_average_fusion(rgb_original, thermal_original, alpha=0.7):
    """
    Fuse aligned RGB and thermal images using weighted average
    alpha: weight for RGB (0-1), thermal weight = (1-alpha)
    Thermal stays in grayscale, only converted to RGB for fusion
    """
    # Convert thermal to 3-channel only for fusion calculation
    thermal_3ch = cv2.cvtColor(thermal_original, cv2.COLOR_GRAY2BGR)
    
    # Weighted average fusion
    fused = cv2.addWeighted(rgb_original, alpha, thermal_3ch, 1-alpha, 0)
    
    return fused

def load_image_pair(rgb_path, thermal_path):
    """Load and resize RGB and thermal image pair"""
    rgb = cv2.imread(rgb_path)
    thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
    
    if rgb is None or thermal is None:
        raise ValueError("Could not load images. Check file paths.")
    
    # Resize thermal to match RGB dimensions
    thermal_resized = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))
    
    return rgb, thermal_resized

def display_results(rgb_original, thermal_original, fused, alpha):
    """Display RGB, Thermal, and Fused image in a single horizontal row"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # RGB Image
    axes[0].imshow(cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original RGB", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    # Thermal Image
    axes[1].imshow(thermal_original, cmap='gray')
    axes[1].set_title("Original Thermal (Grayscale)", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    # Fused Image
    axes[2].imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Fused Image\nWeighted Average (α={alpha})", fontsize=14, fontweight='bold')
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def save_results(rgb_original, thermal_original, fused, output_dir="cnn_alignment_output"):
    """Save alignment and fusion results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images (thermal stays in grayscale)

    cv2.imwrite(os.path.join(output_dir, "thermal_grayscale.jpg"), thermal_original)  # Grayscale thermal
    cv2.imwrite(os.path.join(output_dir, "fused_image.jpg"), fused)
    
    print(f"\nResults saved to '{output_dir}/':")
    print("- thermal_grayscale.jpg (kept in grayscale)") 
    print("- fused_image.jpg")
 
if __name__  == "__main__":
    """
    Main function for CNN-based image alignment and fusion
    """
    # Configuration
    rgb_path = "/home/csio-isens/Desktop/image-resizer/data/M3fd/visible/03778.png"
    thermal_path = "/home/csio-isens/Desktop/image-resizer/data/M3fd/IR/03778.png"
    alpha = 0.5 
    print(f"Fusion weight (RGB): {alpha}")
    try:
        # Step 1: Load images
        print("Loading images...")
        rgb_original, thermal_original = load_image_pair(rgb_path, thermal_path)
        print(f"RGB shape: {rgb_original.shape}")
        print(f"Thermal shape: {thermal_original.shape}")
    

        # Step 3: Weighted average fusion
        print("\nPerforming weighted average fusion...")
        fused_image = weighted_average_fusion(rgb_original, thermal_original, alpha)
        
        # Step 4: Display results
        print("\nDisplaying results...")
        display_results(rgb_original, thermal_original, fused_image, alpha)
        
        # Step 5: Save results
        save_results(rgb_original, thermal_original, fused_image)
        

        print(f"Fusion weight (RGB): {alpha}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your image paths and ensure the images exist.")

    
    # score, _ = ssim(rgb_original, thermal_original, full=True)
    # print("SSIM after alignment:", score)

