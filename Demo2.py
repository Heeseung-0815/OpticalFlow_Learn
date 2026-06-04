# -*- coding: utf-8 -*-
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

if not ret:
    print("Failed to capture video")
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:

    ret, frame2 = cap.read()

    if not ret:
        break

    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prvs,
        nxt,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, ang = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv[..., 0] = ang * 180 / np.pi / 2

    hsv[..., 2] = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    cv2.imshow("Dense Flow", rgb)

    if cv2.waitKey(30) == 27:
        break

    prvs = nxt

cap.release()
cv2.destroyAllWindows()