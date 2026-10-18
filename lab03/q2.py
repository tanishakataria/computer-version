# Task 2: The Dizzy Satellite (Linear - Rotation)
import numpy as np
import cv2

img = cv2.imread("satellite.png")
if img is None:
    # file na mile to test image bana lo
    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (10, 10), (290, 190), (0, 0, 0), 3)
    cv2.putText(img, "CITY", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

h, w = img.shape[:2]

angle = 45                      # degree
theta = np.radians(angle)       # sin/cos radians mein lete hain
c = np.cos(theta)
s = np.sin(theta)

# 2x2 rotation matrix
R = np.array([[c, -s],
              [s,  c]])
print("Rotation matrix:")
print(R)

# naye image ka size manually:
# rotate hone ke baad corners bahar nikalte hain, is liye canvas bada chahiye
new_w = int(np.ceil(abs(w * c) + abs(h * s)))
new_h = int(np.ceil(abs(w * s) + abs(h * c)))
print("Original size:", w, "x", h)
print("New size:", new_w, "x", new_h)

# rotation center ke around chahiye (origin ke around nahi):
# 1) original center ko origin par lao  2) rotate karo  3) naye canvas ke center par rakho
cx, cy = w / 2, h / 2
ncx, ncy = new_w / 2, new_h / 2

T1 = np.array([[1, 0, -cx],
               [0, 1, -cy],
               [0, 0, 1]])
R3 = np.array([[c, -s, 0],
               [s,  c, 0],
               [0,  0, 1]])
T2 = np.array([[1, 0, ncx],
               [0, 1, ncy],
               [0, 0, 1]])

M = T2 @ R3 @ T1
M_inv = np.linalg.inv(M)

out = np.zeros((new_h, new_w, 3), np.uint8)

# inverse mapping
for y in range(new_h):
    for x in range(new_w):
        src = M_inv @ np.array([x, y, 1])
        px = int(round(src[0]))
        py = int(round(src[1]))
        if 0 <= px < w and 0 <= py < h:
            out[y, x] = img[py, px]

cv2.imwrite("task2_output.png", out)
cv2.imshow("Original", img)
cv2.imshow("Rotated back", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
