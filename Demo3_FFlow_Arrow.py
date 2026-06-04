import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

step = 16

while True:

    ret, frame2 = cap.read()

    if not ret:
        break

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    h, w = gray1.shape

    for y in range(0, h, step):
        for x in range(0, w, step):

            fx, fy = flow[y, x]

            cv2.arrowedLine(
                frame2,
                (x, y),
                (int(x + fx), int(y + fy)),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    cv2.imshow("Flow Arrow", frame2)

    if cv2.waitKey(30) == 27:
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()