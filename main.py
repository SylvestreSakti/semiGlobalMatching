import numpy as np
import cv2

for i in range(1,40) :
    img1 = cv2.imread("images/blend ("+str(i)+").jpg")
    img2 = cv2.imread("images/blend (" + str(i+1) + ").jpg")

    # disparity settings
    window_size = 5
    min_disp = 0
    num_disp = 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print('computing disparity...')
    disp1 = stereo.compute(img1, img2)

    disp1 = cv2.normalize(src=disp1, dst=disp1, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    disp1 = np.uint8(disp1)
    #cv2.imshow('Disparity Map', cv2.resize(disp1,None,None,0.3,0.3))
    cv2.imwrite("images/dispblend ("+str(i)+").jpg",disp1)
    print(i)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

def disparity_map(image2, image1, line_coord):
    assert image1.shape == image2.shape
    assert 0 <= line_coord < image1.shape[0]
    print(image1)