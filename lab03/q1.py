# Task 1: The Tiny Fingerprint (Linear - Scale)
import numpy as np
import cv2

img = cv2.imread("fingerprint.png")
if img is None:
    # file na mile to test image bana lo
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 60, (0, 0, 0), 2)
    cv2.circle(img, (100, 100), 40, (0, 0, 0), 2)
    cv2.circle(img, (100, 100), 20, (0, 0, 0), 2)

h, w = img.shape[:2]
sx = 3   # 300% x-axis
sy = 3   # 300% y-axis

# 2x2 scaling matrix
S = np.array([[sx, 0],
              [0, sy]])
print("Scaling matrix:")
print(S)

# scale origin (0,0) se hota hai, is liye image corner ki taraf nikal jati hai.
# center par rakhne ke liye shift chahiye:
# x_new = sx*(x - cx) + cx  ->  tx = cx - sx*cx
cx = w / 2
cy = h / 2
tx = cx - sx * cx
ty = cy - sy * cy
print("Center rakhne ke liye shift:", tx, ty)

# 3x3 matrix (scale + centering shift)
M = np.array([[sx, 0, tx],
              [0, sy, ty],
              [0, 0, 1]])

# inverse mapping: output ke har pixel ke liye dekho original mein kahan se aaya
M_inv = np.linalg.inv(M)
out = np.zeros_like(img)

for y in range(h):
    for x in range(w):
        src = M_inv @ np.array([x, y, 1])
        sx_pix = int(round(src[0]))
        sy_pix = int(round(src[1]))
        if 0 <= sx_pix < w and 0 <= sy_pix < h:
            out[y, x] = img[sy_pix, sx_pix]

cv2.imwrite("task1_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Scaled 300%", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
