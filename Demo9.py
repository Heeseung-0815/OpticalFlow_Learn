import cv2

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()

gray1 = cv2.cvtColor(
    frame1,
    cv2.COLOR_BGR2GRAY
)

tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

while True:

    ret, frame2 = cap.read()

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    flow = tvl1.calc(
        gray1,
        gray2,
        None
    )

    mag, ang = cv2.cartToPolar(
        flow[...,0],
        flow[...,1]
    )

    result = cv2.normalize(
        mag,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    cv2.imshow(
        "TVL1",
        result.astype("uint8")
    )

    gray1 = gray2

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()