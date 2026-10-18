import cv2
import numpy as np

ref_path = "object.jpg"
video_path = "video.mp4"

ref = cv2.imread(ref_path, 0)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
kp1, des1 = sift.detectAndCompute(ref, None)

cap = cv2.VideoCapture(video_path)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)

    if des2 is not None and len(des2) > 2:
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                good.append(pair[0])

        if len(good) >= 10:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = ref.shape
                corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                box = cv2.perspectiveTransform(corners, H)
                frame = cv2.polylines(frame, [np.int32(box)], True, (0, 255, 0), 3)
                cv2.putText(frame, "Object", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
