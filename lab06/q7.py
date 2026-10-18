import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "coins.jpg"

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                           param1=100, param2=40, minRadius=15, maxRadius=80)

count = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0]:
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        count += 1

print("Total coins:", count)
cv2.putText(img, "Coins: %d" % count, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
