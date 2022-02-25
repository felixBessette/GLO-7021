import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import corner_peaks, BRIEF

# Load the image
img = cv2.imread('remise/source/bw-rectified-left-022148small.png')
image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)
keypoints_with_nonmax = fast.detect(image, None)
y = 0
image_with_nonmax = np.copy(img)
image_without_nonmax = np.copy(img)
cv2.drawKeypoints(img, keypoints_with_nonmax, image_with_nonmax, color=(0,220,202), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(image_with_nonmax)
plt.show()
