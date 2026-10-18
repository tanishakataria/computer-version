import numpy as np
import cv2

data = np.load("camera_calibration.npz")
K = data["camera_matrix"]
dist = data["dist_coeffs"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("webcam open nahi hui")

ok, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
x, y, rw, rh = roi

while True:
    ok, frame = cap.read()
    if not ok:
        break

    undistorted = cv2.undistort(frame, K, dist, None, new_K)
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
        undistorted = cv2.resize(undistorted, (w, h))

    cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(undistorted, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live feed", np.hstack([frame, undistorted]))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
