import numpy as np
import cv2

img1 = cv2.imread('images/img1.jpg')
img2 = cv2.imread('images/img2.jpg')

roi1 = img1[0:800,1500:2500]


print(img1.shape)
print(img1.dtype)

