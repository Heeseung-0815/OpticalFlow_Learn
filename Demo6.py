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

    mag, _ = cv2.cartToPolar(
        flow[...,0],
        flow[...,1]
    )

    heatmap = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap = heatmap.astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    cv2.imshow(
        "Flow Heatmap",
        heatmap
    )

    if cv2.waitKey(1)==27:
        break

    gray1 = gray2

cap.release()
cv2.destroyAllWindows()