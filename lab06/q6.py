import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "road.jpg"

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

h, w = edges.shape
mask = np.zeros_like(edges)
roi = np.array([[(0, h), (w // 2, int(h * 0.6)), (w, h)]], dtype=np.int32)
cv2.fillPoly(mask, roi, 255)
masked = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)

out = img.copy()
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if abs(y2 - y1) < 10:
            continue
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 4)

plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.title("Lane Detection")
plt.axis("off")
plt.show()
