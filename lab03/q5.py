import numpy as np
import cv2

img = cv2.imread("chip.png")
if img is None:
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (200, 160), (0, 0, 0), -1)
    cv2.putText(img, "CHIP", (105, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

h, w = img.shape[:2]

angle = -90
tx = 30
ty = -20

theta = np.radians(angle)
c = np.cos(theta)
s = np.sin(theta)
cx, cy = w / 2, h / 2

R = np.array([[c, -s, 0],
              [s,  c, 0],
              [0,  0, 1]])
T = np.array([[1, 0, tx],
              [0, 1, ty],
              [0, 0, 1]])
C1 = np.array([[1, 0, cx],
               [0, 1, cy],
               [0, 0, 1]])
C2 = np.array([[1, 0, -cx],
               [0, 1, -cy],
               [0, 0, 1]])

M = T @ C1 @ R @ C2
print("Rigid matrix:")
print(np.round(M, 3))

M_inv = np.linalg.inv(M)
out = np.zeros_like(img)

for y in range(h):
    for x in range(w):
        src = M_inv @ np.array([x, y, 1])
        px = int(round(src[0]))
        py = int(round(src[1]))
        if 0 <= px < w and 0 <= py < h:
            out[y, x] = img[py, px]

cv2.imwrite("task5_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Aligned", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
