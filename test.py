import cv2
from debug_var import *
import numpy as np
from detection import *

video = cv2.VideoCapture("test/testvideo_human.mp4")
image = cv2.imread("test/testimage_human.jpg")
detection = DetectionManager()

while True:
    ret, frame = video.read()
    if not ret:
        break
    key = cv2.waitKey(1)
    if key == 27:
        break
    detection.pose_eval(frame)

#detection(image)
cv2.destroyAllWindows()
