import numpy as np
import cv2
 
img = cv2.imread("map.png")
if img is None:
    img = np.full((300, 400, 3), 220, np.uint8)
    cv2.putText(img, "X", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
 
h, w = img.shape[:2]
 
tx = 150
ty = 80
 
T = np.array([[1, 0, tx],
              [0, 1, ty],
              [0, 0, 1]])
print("Translation matrix:")
print(T)
 
T_inv = np.linalg.inv(T)
 
new_w = w + tx
new_h = h + ty
out = np.zeros((new_h, new_w, 3), np.uint8)
 
for y in range(new_h):
    for x in range(new_w):
        src = T_inv @ np.array([x, y, 1])
        px = int(round(src[0]))
        py = int(round(src[1]))
        if 0 <= px < w and 0 <= py < h:
            out[y, x] = img[py, px]
 
cv2.imwrite("task4_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Shifted", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
