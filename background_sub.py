import cv2
import numpy as np

# import matplotlib
#
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
#
# img = cv2.imread('sherlock.jpg')
# mask = np.zeros(img.shape[:2], np.uint8)
#
# fgdModel = np.zeros((1, 65), np.float64)
# bgdModel = np.zeros((1, 65), np.float64)
#
# rect = (50, 50, 450, 290)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# img = img * mask2[:, :, np.newaxis]
#
# plt.imshow(img), plt.show()

cap = cv2.VideoCapture('dc.mp4')

panel = np.zeros([100, 70, 3], np.uint8)

cv2.namedWindow("panel")


def nothing(x):
    pass


cv2.createTrackbar("L - h", "panel", 0, 179, nothing)
cv2.createTrackbar("U - h", "panel", 179, 179, nothing)

cv2.createTrackbar("L - s", "panel", 0, 255, nothing)
cv2.createTrackbar("U - s", "panel", 255, 255, nothing)

cv2.createTrackbar("L - v", "panel", 0, 255, nothing)
cv2.createTrackbar("U - v", "panel", 255, 255, nothing)

while True:
    _, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # conversion from bgr(BLUE,GREEN,RED) color space to hsv(HUE,
    # SATURATION,VALUE) color space

    l_h = cv2.getTrackbarPos("L - h", "panel")
    u_h = cv2.getTrackbarPos("U - h", "panel")
    l_s = cv2.getTrackbarPos("L - s", "panel")
    u_s = cv2.getTrackbarPos("U - s", "panel")
    l_v = cv2.getTrackbarPos("L - v", "panel")
    u_v = cv2.getTrackbarPos("U - v", "panel")

    lower_green = np.array([l_h, l_s, l_v])  # setting min hue,saturation and value params
    upper_green = np.array([u_h, u_s, u_v])  # setting max hue,saturation and value params

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # cv2.imshow("mask", mask)
    # cv2.imshow("frame", frame)
    cv2.imshow("background", bg)
    cv2.imshow("foreground", fg)
    cv2.imshow("panel", panel)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
