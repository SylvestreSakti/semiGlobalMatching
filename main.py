import numpy as np
import cv2
import os


def find_image_files(input_directory):
    files_list = os.listdir(input_directory)
    images_list = []
    for filename in files_list:
        if filename[-4:] in ['.jpg','jpeg','.png'] :
            print(filename)
            images_list.append(os.path.join(input_directory, filename))
    return images_list


def disparity_map(image1, image2):
    assert image1.shape == image2.shape
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
    return disp1


input_directory = "input_images"
output_directory = "output_images"
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
image_files = find_image_files(input_directory)
for i in range(len(image_files) - 1):
    print("Loading "+image_files[i]+" and "+image_files[i+1]+" ...")
    img1 = cv2.imread(image_files[i])
    img2 = cv2.imread(image_files[i + 1])
    disp = disparity_map(img1, img2)
    cv2.imwrite(os.path.join(output_directory,"disp(" + str(i) + ").jpg"), disp)
    print("Output saved in "+os.path.join(output_directory,"disp(" + str(i) + ").jpg"))