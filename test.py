import cv2
from detection import *

video = cv2.VideoCapture("test/testvideo_human.mp4")
image = cv2.imread("test/testimage_human.jpg")

while True:
    ret, frame = video.read()
    if not ret:
        break
    key = cv2.waitKey(1)
    if key == 27:
        break
    track(frame)

#detection(image)
cv2.destroyAllWindows()