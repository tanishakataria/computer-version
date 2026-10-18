import cv2
import numpy as np
import matplotlib.pyplot as plt

scene_path = "lab.jpg"
assets = {"Monitor": "monitor.jpg", "Keyboard": "keyboard.jpg"}

scene = cv2.imread(scene_path)
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
kp2, des2 = sift.detectAndCompute(scene_gray, None)

inventory = {}
for name, p in assets.items():
    ref = cv2.imread(p, 0)
    kp1, des1 = sift.detectAndCompute(ref, None)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
            good.append(pair[0])

    if len(good) >= 10:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            h, w = ref.shape
            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            box = cv2.perspectiveTransform(corners, H)
            scene = cv2.polylines(scene, [np.int32(box)], True, (0, 255, 0), 3)
            x, y = np.int32(box[0][0])
            cv2.putText(scene, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            inventory[name] = "Found"
            continue
    inventory[name] = "Not found"

print("Inventory:")
for k, v in inventory.items():
    print(" ", k, "->", v)

plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
plt.title("Asset Tracking (SIFT)")
plt.axis("off")
plt.show()
