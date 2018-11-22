import numpy as np
import cv2

img1 = cv2.imread('images/img1.jpg')
img2 = cv2.imread('images/img2.jpg')

roi1 = img1[0:600,1500:2591].astype(np.uint8)
roi2 = img2[0:600,1500:2591].astype(np.uint8)

print(img1.shape)
print(img1.dtype)

# cv2.imshow('image',roi1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# disparity settings
window_size = 5
min_disp = 0
num_disp = 64
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# compute disparity
disparity = stereo.compute(roi1, roi2).astype(np.float32)/16
#cv2.imshow('image',roi1)
cv2.imshow('image',disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

def disparity_map(image1, image2, line_coord):
    assert image1.shape == image2.shape
    assert 0 <= line_coord < image1.shape[0]
    print(image1)