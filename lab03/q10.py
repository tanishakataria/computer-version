import os
import numpy as np
import cv2

painting = cv2.imread("painting.png")
if painting is None:
    painting = np.full((300, 400, 3), 255, np.uint8)
    cv2.circle(painting, (200, 150), 100, (0, 0, 255), -1)
    cv2.putText(painting, "ART", (140, 165), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

wall = cv2.imread("wall.jpg")
if wall is None:
    wall = np.full((500, 700, 3), 200, np.uint8)
    cv2.polylines(wall, [np.array([[300, 120], [560, 160], [560, 400], [300, 430]])],
                  True, (40, 40, 40), 6)

ph, pw = painting.shape[:2]
wh, ww = wall.shape[:2]

frame = [[300, 120], [560, 160], [560, 400], [300, 430]]

S = np.array([[0.5, 0, 0],
              [0, 0.5, 0],
              [0, 0, 1]])

t = np.radians(5)
c = np.cos(t)
s = np.sin(t)
Rg = np.array([[c, -s, 280],
               [s,  c, 110],
               [0,  0, 1]])

corners = np.array([[0, 0, 1], [pw, 0, 1], [pw, ph, 1], [0, ph, 1]]).T
moved = Rg @ S @ corners
moved = (moved[:2] / moved[2]).T

A = []
B = []
for (x, y), (xn, yn) in zip(moved, frame):
    A.append([x, y, 1, 0, 0, 0, -x * xn, -y * xn])
    B.append(xn)
    A.append([0, 0, 0, x, y, 1, -x * yn, -y * yn])
    B.append(yn)
hv = np.linalg.solve(np.array(A, dtype=float), np.array(B, dtype=float))
H = np.array([[hv[0], hv[1], hv[2]],
              [hv[3], hv[4], hv[5]],
              [hv[6], hv[7], 1]])

total = H @ Rg @ S
print("Final matrix:")
print(np.round(total, 4))

total_inv = np.linalg.inv(total)
result = wall.copy()

for y in range(wh):
    for x in range(ww):
        p = total_inv @ np.array([x, y, 1])
        px = int(round(p[0] / p[2]))
        py = int(round(p[1] / p[2]))
        if 0 <= px < pw and 0 <= py < ph:
            result[y, x] = painting[py, px]

cv2.imwrite("task10_output.png", result)
cv2.imshow("Forged", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
