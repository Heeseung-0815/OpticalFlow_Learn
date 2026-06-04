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

    if not ret:
        break

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

    mag, ang = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    motion_mask = (mag > 2).astype(
        np.uint8
    ) * 255

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(
            frame2,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Motion Detection",
        frame2
    )

    if cv2.waitKey(30) == 27:
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()