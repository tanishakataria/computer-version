import cv2
import numpy as np
import matplotlib.pyplot as plt

image_paths = ["p1.jpg", "p2.jpg", "p3.jpg"]


def stitch_two(left, right):
    sift = cv2.SIFT_create()
    g1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print("good matches:", len(good))

    pts_right = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_left = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)

    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    result = cv2.warpPerspective(right, H, (w1 + w2, max(h1, h2)))
    result[0:h1, 0:w1] = left

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.max(axis=0) > 0)[0]
    return result[:, :cols[-1] + 1]


imgs = [cv2.imread(p) for p in image_paths]
pano = imgs[0]
for nxt in imgs[1:]:
    pano = stitch_two(pano, nxt)

cv2.imwrite("panorama.jpg", pano)
plt.figure(figsize=(12, 5))
plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
plt.title("Panorama")
plt.axis("off")
plt.show()
