def find_with_sift(ref_gray, scene_gray, sift, bf, min_matches=10):
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_gray, None)
    if des1 is None or des2 is None:
        return None
 
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
            good.append(pair[0])
 
    if len(good) < min_matches:
        return None
 
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None
 
    h, w = ref_gray.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H)
 
 
def asset_tracking(scene_path, asset_paths):
    # asset_paths = {"Monitor": "monitor.jpg", "Keyboard": "keyboard.jpg"}
    scene = cv2.imread(scene_path)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
 
    inventory = {}
    for name, p in asset_paths.items():
        ref = cv2.imread(p, 0)
        box = find_with_sift(ref, scene_gray, sift, bf)
        if box is not None:
            scene = cv2.polylines(scene, [np.int32(box)], True, (0, 255, 0), 3)
            x, y = np.int32(box[0][0])
            cv2.putText(scene, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            inventory[name] = "Found"
        else:
            inventory[name] = "Not found"
 
    print("Inventory:")
    for k, v in inventory.items():
        print(" ", k, "->", v)
 
    plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    plt.title("Asset Tracking (SIFT)")
    plt.axis("off")
    plt.show()
