import cv2
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as compare_ssim
import argparse

## Each Exp = 8 plates - 2 control , 6 treated (1x,2x,5x,10x,25x,50x),
## Each group will have 300 images, therefore the control image group needs to be analysed in a loop
## The treated group will need to be analysed individually and all together in one folder of 300
## The time stamp needs to be added to the labels and the analyses and visualization needs to be shown chronologically
## Binary classification performed to see if the model can determine a detection limit on the various concentration images


d1=0
folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\NAP and ATZ Experiments\ATZ\FToxcontrolATZ1000xiii'
img_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
total_images = len(img_list)
print(total_images)
height = 960
width = 720
#height = 660
#width = 500
#height = 4032
#width = 3024
##kernal = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) ##edge detection Kernal
kernal = np.array([[0,-1,0],[-1,20,-1],[0,-1,0]]) ##sharpen detection Kernal

colors = ('b', 'g', 'r')
# Load the list of images
control_img = cv2.filter2D(cv2.imread(img_list[0]), -1,kernal) ## -2 usually -1 testing if the filter will change the features outputted
#control_img2 = cv2.filter2D(cv2.imread(img_list[1]),-1, kernal)
for i, color in enumerate(colors):
    hist = cv2.calcHist([control_img], [i], None, [256], [0, 256])
    #hist_mask = cv2.calcHist([control_img], [i], mask, [256], [0, 256])

    plt.plot(hist, color=color)
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel frequency")
    plt.title('Color distribution of control image')
    plt.xlim([0,256])

#plt.show()
print(control_img.shape)
#print(control_img2.shape)
# Denoise the control image

control_imga = cv2.fastNlMeansDenoisingColored(control_img, None, 20, 20, 7, 21)


# Apply histogram equalization and image normalization to the control image
control_imgb = cv2.cvtColor(control_imga, cv2.COLOR_BGR2Lab)
L, A, B = cv2.split(control_imgb)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
L_clahe = clahe.apply(L)
lab_clahe = cv2.merge((L_clahe, A, B))

control_imgc = cv2.cvtColor(control_imgb, cv2.COLOR_Lab2BGR)
control_imgd = cv2.normalize(control_imgc, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

img_resized = cv2.resize(control_imgd, (0,0), fx = 0.1, fy = 0.1)
cv2.imshow("control image d", img_resized)
cv2.waitKey(0)

# Extract blue-green features from the control image
hsv = cv2.cvtColor(control_imgd, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (65, 30, 30), (135, 255, 255)) #75,150
blue_green_img = cv2.bitwise_and(hsv, control_imgd, mask=mask)

# Apply edge detection to the blue-green image
edges = cv2.Canny(blue_green_img, 100, 200)

# Apply segmentation to the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Classify the contours as either object or noise based on their blue-green color
object_contours = []
for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    roi = control_imgd[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) 
    mask = cv2.inRange(hsv_roi, (70, 30, 30), (155, 255, 255)) # 70 ; 150
    if cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1]) > 0.5:
        object_contours.append(contour)


# Draw the object contours on the control image
control_imge = cv2.drawContours(control_imgd, object_contours, -1, (0, 0, 255), 2)
colors = ('b', 'g', 'r')

#cv2.imwrite('C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\ProcessedimagesBcontrol05232023\controlimageB.d1%jpg'%d1, control_imge)

for i, color in enumerate(colors):
    hist = cv2.calcHist([control_imge], [i], None, [256], [0, 256])
    #hist_mask = cv2.calcHist([control_img], [i], mask, [256], [0, 256])
    plt.xlabel("Pixel value")
    plt.ylabel("Pixel frequency")
    plt.title('Color distribution of processed control image')
    plt.plot(hist, color=color)
    plt.xlim([0,256])

control_imge1 = control_imge
#plt.show()

img_resize = cv2.resize(control_imge, (0,0), fx = 0.1, fy = 0.1)
#cv2.imshow("control imge + contours", img_resize)

cv2.waitKey(0)
d=0

## Control iterator

folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\NAP and ATZ Experiments\ATZ\FToxcontrolATZ1000xiv'
img_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
differences_start = [1]
differences_start1 = [1]
msedifferences_secondary = []

for folder_path in img_list[0:]:
    img = cv2.imread(folder_path)
    plt.imshow(img)
    img = cv2.filter2D(img, -1, kernal)
    # Denoise the image
    img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)#21
    # Apply histogram equalization and image normalization to the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img)  # intensity L - LAB color space
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    cv2.imwrite( r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\Filtered and normalized images\filteredimage_%d.jpg' % d, img)

    ##img_resized1 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    ##cv2.imshow("LAB iterator", img_resized1)
    ##cv2.waitKey(0)
    # Extract blue-green features from the image
    hsv1= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv1, (30, 30, 30), (150, 255, 255))
    blue_green_imgb = cv2.bitwise_and(hsv1, img, mask=mask)

    ##img_resized2 = cv2.resize(blue_green_imgb, (0, 0), fx=0.25, fy=0.25)
    ##cv2.imshow("BG iterator", img_resized2)
    ##cv2.waitKey(0)

    # Apply edge detection to the blue-green image
    edges1 = cv2.Canny(blue_green_imgb,100, 200)

    # Apply segmentation to the edges
    contours, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Classify the contours as either object or noise based on their blue-green color
    object_contours1 = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        roi1 = img[y:y+h, x:x+w]
        hsv_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_roi1, (30, 30, 30), (150, 255, 255))
        if cv2.countNonZero(mask1) / (roi1.shape[0] * roi1.shape[1]) > 0.5:
            object_contours1.append(contour)

    # Draw the object contours on the original image
    img1 = cv2.drawContours(img, object_contours1, -1, (0, 0, 255), 2)
    cv2.imwrite(
        r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\NAP and ATZ Experiments\ATZ\ProcessedimagescontrolATZ500xiii\ControlImagesATZ500xiii_%d.jpg' % d,
        img1)
    img_copy = img1.copy()
    d+=1
    ##cv2.imshow("img with contours",img_copy)
    ##cv2.waitKey(0)

## Treatment iterator

folder_path = r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\NAP and ATZ Experiments\ATZ\FToxtreatedATZ500xiii'
img_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])
differences_start = [1]
differences_start1 = [1]
msedifferences_secondary = []

for folder_path in img_list[0:]:
    img = cv2.imread(folder_path)
    plt.imshow(img)
    img = cv2.filter2D(img, -1, kernal)
    # Denoise the image
    img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)#21
    # Apply histogram equalization and image normalization to the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img)  # intensity L - LAB color space
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge((L_clahe, A, B))

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    ##img_resized1 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    ##cv2.imshow("LAB iterator", img_resized1)
    ##cv2.waitKey(0)
    # Extract blue-green features from the image
    hsv1= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv1, (30, 30, 30), (150, 255, 255))
    blue_green_imgb = cv2.bitwise_and(hsv1, img, mask=mask)

    ##img_resized2 = cv2.resize(blue_green_imgb, (0, 0), fx=0.25, fy=0.25)
    ##cv2.imshow("BG iterator", img_resized2)
    ##cv2.waitKey(0)

    # Apply edge detection to the blue-green image
    edges1 = cv2.Canny(blue_green_imgb,100, 200)

    # Apply segmentation to the edges
    contours, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Classify the contours as either object or noise based on their blue-green color
    object_contours1 = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        roi1 = img[y:y+h, x:x+w]
        hsv_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_roi1, (30, 30, 30), (150, 255, 255))
        if cv2.countNonZero(mask1) / (roi1.shape[0] * roi1.shape[1]) > 0.5:
            object_contours1.append(contour)

    # Draw the object contours on the original image
    img1 = cv2.drawContours(img, object_contours1, -1, (0, 0, 255), 2)
    cv2.imwrite(r'C:\Users\Irshaad Parker\Desktop\MEngSc Biomedical Engineering 2022\Plate data\NAP and ATZ Experiments\ATZ\ProcessedimagestreatedATZ500xiii\TreatedImagesATZ500xiii_%d.jpg'%d, img1)
    img_copy = img1.copy()
    d+=1
    ##cv2.imshow("img with contours",img_copy)
    ##cv2.waitKey(0)

