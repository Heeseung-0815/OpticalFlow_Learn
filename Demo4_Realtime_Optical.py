import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, old_frame = cap.read()

old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=200,
    qualityLevel=0.3,
    minDistance=7
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        gray,
        p0,
        None
    )

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for new, old in zip(
            good_new,
            good_old):

        a, b = new.ravel()
        c, d = old.ravel()

        cv2.line(
            frame,
            (int(a), int(b)),
            (int(c), int(d)),
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Realtime Optical Flow",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

    old_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()