import numpy as np
import cv2

img = cv2.imread("chalk_art.jpg")
if img is None:
    img = np.full((400, 500, 3), 180, np.uint8)
    pts = np.array([[150, 80], [350, 80], [475, 360], [25, 360]])
    cv2.fillPoly(img, [pts], (0, 200, 255))
    cv2.putText(img, "ART", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

h, w = img.shape[:2]

src = [[150, 80], [350, 80], [475, 360], [25, 360]]

side = 400
dst = [[0, 0], [side, 0], [side, side], [0, side]]

A = []
B = []
for (x, y), (xn, yn) in zip(src, dst):
    A.append([x, y, 1, 0, 0, 0, -x * xn, -y * xn])
    B.append(xn)
    A.append([0, 0, 0, x, y, 1, -x * yn, -y * yn])
    B.append(yn)

h_vals = np.linalg.solve(np.array(A, dtype=float), np.array(B, dtype=float))
H = np.array([[h_vals[0], h_vals[1], h_vals[2]],
              [h_vals[3], h_vals[4], h_vals[5]],
              [h_vals[6], h_vals[7], 1]])
print("Perspective matrix:")
print(np.round(H, 4))

H_inv = np.linalg.inv(H)
out = np.zeros((side, side, 3), np.uint8)

for y in range(side):
    for x in range(side):
        p = H_inv @ np.array([x, y, 1])
        px = int(round(p[0] / p[2]))
        py = int(round(p[1] / p[2]))
        if 0 <= px < w and 0 <= py < h:
            out[y, x] = img[py, px]

cv2.imwrite("task8_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Birds-eye view", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
