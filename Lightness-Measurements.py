import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_lab(rgb_image):
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    return lab_image


def plot_lab_lightness(control_lab, treated_lab_images):
    treated_lightness = []

    for treated_lab in treated_lab_images:
        treated_lightness.append(treated_lab[:, :, 0].flatten())  # L channel

    plt.figure(figsize=(10, 6))
    plt.boxplot(treated_lightness, labels=['50X', '25X', '10X', '5X', '2X','Undiluted'])
    plt.axhline(y=control_lab[:, :, 0].flatten().mean(), color='r', linestyle='dashed', label='Control Mean Lightness')
    plt.title('Lightness (L*) Comparison Atrazine')
    plt.xlabel('Dilution Factor')
    plt.ylabel('Lightness (L*)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    control_folder = folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\SSIM\SSIM ATZ\Control\0.jpg'

    treated_folder = folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\SSIM\SSIM ATZ\Treated - Copy'

    control_image = cv2.cvtColor(cv2.imread(control_folder), cv2.COLOR_BGR2RGB)
    control_lab = rgb_to_lab(control_image)

    treated_images = [os.path.join(treated_folder, file) for file in os.listdir(treated_folder)]
    treated_lab_images = [rgb_to_lab(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)) for image_path in treated_images]

    plot_lab_lightness(control_lab, treated_lab_images)





