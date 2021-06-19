#Convert a .png file to image buffer. 

import cv2
import os
import numpy as np
# path of input image
path = r'3.png'

# Reading an image in default mode 
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

outname = "3.h"

npimage = np.round(np.asarray(image))
npimage = npimage.ravel()
npimage = npimage.astype(int)
print(image.shape)
print("inputs.h file created..\n")
np.savetxt(outname, npimage[None], fmt='%d',delimiter=',', header='#define DIGIT_IMG_DATA {', footer='}\n',comments='')
