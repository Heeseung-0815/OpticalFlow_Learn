import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

gray1 = cv2.cvtColor(
    frame1,
    cv2.COLOR_BGR2GRAY
)

while True:

    ret, frame2 = cap.read()

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

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

    mag, _ = cv2.cartToPolar(
        flow[...,0],
        flow[...,1]
    )

    mask = np.uint8(
        mag > 3
    ) * 255

    cv2.imshow(
        "Motion Mask",
        mask
    )

    gray1 = gray2

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()