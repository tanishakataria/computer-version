import cv2
import numpy as np

source = "video.mp4"

cap = cv2.VideoCapture(source)
bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

zx1, zy1, zx2, zy2 = 200, 100, 450, 350

while True:
    ok, frame = cap.read()
    if not ok:
        break

    fg = bg.apply(frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.dilate(fg, None, iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    alarm = False
    for c in contours:
        if cv2.contourArea(c) < 1500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if x < zx2 and x + w > zx1 and y < zy2 and y + h > zy1:
            alarm = True

    color = (0, 0, 255) if alarm else (0, 255, 0)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 3)
    if alarm:
        cv2.putText(frame, "ALARM! Unauthorized object", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print("ALARM!")

    cv2.imshow("Security", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
