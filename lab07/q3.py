import glob
import numpy as np
import cv2

pattern = (9, 6)
square_size = 1.0

objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []
size = None

files = sorted(glob.glob("calib_images/*.jpg") + glob.glob("calib_images/*.png"))
if not files:
    raise SystemExit("calib_images folder mein chessboard ki images rakho")

for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        print("ok:", f)
    else:
        print("corners nahi mile:", f)

if len(objpoints) < 5:
    raise SystemExit("kam az kam 5 achi images chahiye")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

total = 0
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    total += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)

print("RMS error:", ret)
print("Mean reprojection error:", total / len(objpoints))
print("Camera matrix:")
print(K)
print("Distortion coefficients:")
print(dist)

np.savez("camera_calibration.npz", camera_matrix=K, dist_coeffs=dist)
print("camera_calibration.npz save ho gayi")
