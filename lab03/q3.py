import numpy as np
import cv2

img = cv2.imread("barcode.png")
if img is None:
    img = np.full((150, 300, 3), 255, np.uint8)
    for i in range(10, 290, 12):
        cv2.line(img, (i, 10), (i, 140), (0, 0, 0), 3)

h, w = img.shape[:2]
k = 0.4

Sh = np.array([[1, -k],
               [0, 1]])
print("Shear matrix:")
print(Sh)

shift = k * h
new_w = w + int(np.ceil(shift))
new_h = h

M = np.array([[1, -k, shift],
              [0, 1, 0],
              [0, 0, 1]])
M_inv = np.linalg.inv(M)

out = np.zeros((new_h, new_w, 3), np.uint8)
for y in range(new_h):
    for x in range(new_w):
        src = M_inv @ np.array([x, y, 1])
        px = int(round(src[0]))
        py = int(round(src[1]))
        if 0 <= px < w and 0 <= py < h:
            out[y, x] = img[py, px]

cv2.imwrite("task3_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Fixed", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
