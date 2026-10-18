import numpy as np
import cv2

img = cv2.imread("photo.png")
if img is None:
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (360, 260), (0, 0, 0), 3)
    cv2.putText(img, "PHOTO", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

h, w = img.shape[:2]

def apply(image, M, out_w, out_h):
    M_inv = np.linalg.inv(M)
    res = np.zeros((out_h, out_w, 3), np.uint8)
    ih, iw = image.shape[:2]
    for y in range(out_h):
        for x in range(out_w):
            src = M_inv @ np.array([x, y, 1])
            px = int(round(src[0]))
            py = int(round(src[1]))
            if 0 <= px < iw and 0 <= py < ih:
                res[y, x] = image[py, px]
    return res

G = np.array([[0.8, 0.3, 40],
              [-0.2, 0.6, 30],
              [0, 0, 1]])
glitched = apply(img, G, w + 100, h + 100)
cv2.imwrite("task7_glitched.png", glitched)

end_pts = [[50, 50], [300, 60], [100, 250]]
start_pts = []
for (x, y) in end_pts:
    p = G @ np.array([x, y, 1])
    start_pts.append([p[0], p[1]])

A = []
B = []
for (x, y), (xn, yn) in zip(start_pts, end_pts):
    A.append([x, y, 1, 0, 0, 0])
    B.append(xn)
    A.append([0, 0, 0, x, y, 1])
    B.append(yn)

p = np.linalg.solve(np.array(A, dtype=float), np.array(B, dtype=float))
M = np.array([[p[0], p[1], p[2]],
              [p[3], p[4], p[5]],
              [0, 0, 1]])
print("Affine matrix (glitched -> fixed):")
print(np.round(M, 3))

fixed = apply(glitched, M, w, h)
cv2.imwrite("task7_fixed.png", fixed)
cv2.imshow("Glitched", glitched)
cv2.imshow("Fixed", fixed)
cv2.waitKey(0)
cv2.destroyAllWindows()
