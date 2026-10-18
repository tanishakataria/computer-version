import os
import numpy as np
import cv2

if os.path.exists("left.jpg") and os.path.exists("right.jpg"):
    img1 = cv2.imread("left.jpg")
    img2 = cv2.imread("right.jpg")

    pts1 = [[520, 100], [540, 300], [600, 250], [580, 50]]
    pts2 = [[20, 105], [40, 305], [100, 255], [80, 55]]
else:
    big = np.zeros((300, 700, 3), np.uint8)
    for i in range(300):
        big[i, :, 0] = int(255 * i / 300)
    for j in range(700):
        big[:, j, 1] = int(255 * j / 700)
    cv2.putText(big, "STADIUM", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
    img1 = big[:, :450].copy()
    img2 = big[:, 250:].copy()
    pts1 = [[300, 50], [400, 50], [400, 250], [300, 250]]
    pts2 = [[50, 50], [150, 50], [150, 250], [50, 250]]

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

A = []
B = []
for (x, y), (xn, yn) in zip(pts2, pts1):
    A.append([x, y, 1, 0, 0, 0, -x * xn, -y * xn])
    B.append(xn)
    A.append([0, 0, 0, x, y, 1, -x * yn, -y * yn])
    B.append(yn)
hv = np.linalg.solve(np.array(A, dtype=float), np.array(B, dtype=float))
H = np.array([[hv[0], hv[1], hv[2]],
              [hv[3], hv[4], hv[5]],
              [hv[6], hv[7], 1]])
print("Homography matrix:")
print(np.round(H, 4))

corners2 = np.array([[0, 0, 1], [w2, 0, 1], [w2, h2, 1], [0, h2, 1]]).T
c = H @ corners2
c = c[:2] / c[2]
all_x = list(c[0]) + [0, w1]
all_y = list(c[1]) + [0, h1]
min_x = int(np.floor(min(all_x)))
min_y = int(np.floor(min(all_y)))
max_x = int(np.ceil(max(all_x)))
max_y = int(np.ceil(max(all_y)))
W = max_x - min_x
Hh = max_y - min_y

T = np.array([[1, 0, -min_x],
              [0, 1, -min_y],
              [0, 0, 1]])
M_inv = np.linalg.inv(T @ H)

pano = np.zeros((Hh, W, 3), np.uint8)
for y in range(Hh):
    for x in range(W):
        p = M_inv @ np.array([x, y, 1])
        px = int(round(p[0] / p[2]))
        py = int(round(p[1] / p[2]))
        if 0 <= px < w2 and 0 <= py < h2:
            pano[y, x] = img2[py, px]

ox = -min_x
oy = -min_y
pano[oy:oy + h1, ox:ox + w1] = img1

cv2.imwrite("task9_panorama.png", pano)
cv2.imshow("Panorama", pano)
cv2.waitKey(0)
cv2.destroyAllWindows()
