import numpy as np
import cv2

img = cv2.imread("blueprint.png")
if img is None:
    img = np.full((150, 200, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 50), (180, 140), (0, 0, 0), 2)
    cv2.line(img, (20, 50), (100, 10), (0, 0, 0), 2)
    cv2.line(img, (100, 10), (180, 50), (0, 0, 0), 2)

h, w = img.shape[:2]

scale = 2.0
angle = -30
tx = 150
ty = 100

theta = np.radians(angle)
c = np.cos(theta)
s = np.sin(theta)

M = np.array([[scale * c, -scale * s, tx],
              [scale * s,  scale * c, ty],
              [0, 0, 1]])
print("Similarity matrix:")
print(np.round(M, 3))

M_inv = np.linalg.inv(M)

new_w = int(w * 2.5)
new_h = int(h * 2.5)
out = np.zeros((new_h, new_w, 3), np.uint8)

for y in range(new_h):
    for x in range(new_w):
        src = M_inv @ np.array([x, y, 1])
        px = int(round(src[0]))
        py = int(round(src[1]))
        if 0 <= px < w and 0 <= py < h:
            out[y, x] = img[py, px]

cv2.imwrite("task6_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Fixed blueprint", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
