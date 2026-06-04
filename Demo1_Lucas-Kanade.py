# -*- coding: utf-8 -*-
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, old_frame = cap.read()

if not ret:
    print("Failed to capture video")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

mask = np.zeros_like(old_frame)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None
    )

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):

        a, b = new.ravel()
        c, d = old.ravel()

        mask = cv2.line(
            mask,
            (int(a), int(b)),
            (int(c), int(d)),
            (0, 255, 0),
            2
        )

        frame = cv2.circle(
            frame,
            (int(a), int(b)),
            4,
            (0, 0, 255),
            -1
        )

    img = cv2.add(frame, mask)

    cv2.imshow("Lucas Kanade", img)

    key = cv2.waitKey(30)

    if key == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()