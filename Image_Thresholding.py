import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load Image
img = cv2.imread("C:\\Users\\Acer\\Desktop\\Tiger.jpeg")

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Binary Thresholding 
_, threshold1 = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

# Apply adaptive mean thresholding
threshold2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 14)

# Apply adaptive gaussian thresholding 
threshold3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 10)

# Display all images side by side 
plt.figure(figsize=(20,8))

# For Binary Thresholding
plt.subplot(1,3,1)
plt.imshow(threshold1, cmap='gray')
plt.title('Binary Thresholding')
plt.axis('off')

# For Adaptive Mean Thresholding
plt.subplot(1,3,2)
plt.imshow(threshold2, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.axis('off')

# For Adaptive Gaussian Thresholding
plt.subplot(1,3,3)
plt.imshow(threshold3, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.axis('off')

# To display all images
plt.show()


