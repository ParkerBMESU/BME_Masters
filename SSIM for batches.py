import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim

def process_images(control_folder, treated_folder):
    control_files = os.listdir(control_folder)
    treated_files = os.listdir(treated_folder)

    control_img = cv2.imread(os.path.join(control_folder, control_files[0]))
    Gray1 = cv2.cvtColor(control_img, cv2.COLOR_BGR2GRAY)

    differences_start = [100]  # Control image SSIM score

    for treated_file in treated_files:
        img_path = os.path.join(treated_folder, treated_file)
        img_copy = cv2.imread(img_path)
        Gray2 = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        (score, _) = compare_ssim(Gray2, Gray1, full=True)  # Structural similarity algorithm

        perc_change = (1 - score) * 100  # Convert SSIM to percentage change
        differences_start.append(perc_change)

        print(f"Difference from Control for {treated_file}: {perc_change:.4f}%")

    return differences_start

def plot_ssim_curve(labels, ssim_differences):
    plt.figure(figsize=(10, 6))
    plt.plot(labels, ssim_differences, marker='o')
    plt.xlabel('Treatment dilution applied')
    plt.ylabel(" % Similarity between images ")
    plt.title('Structural similarity (SSIM) between control and treated images Atrazine')
    ##plt.legend(loc="upper right")
    ##plt.xticks(rotation=45)
    plt.grid(False)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('ssim_difference_curveATZ.png')

    plt.show()

control_folder = folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\SSIM\SSIM ATZ\Control'

treated_folder = folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\SSIM\SSIM ATZ\Treated - Copy'


labels = ['Control', '50X', '25X', '10X', '5X', '2X', 'Undiluted']
ssim_differences = process_images(control_folder, treated_folder)

plot_ssim_curve(labels, ssim_differences)








plt.show()
