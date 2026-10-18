def screen_detection(path, expected_screens=6):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=60, maxLineGap=10)
 
    # sirf horizontal aur vertical lines rakhni hain (screen ke borders)
    mask = np.zeros_like(gray)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170 or 80 < angle < 100:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
 
    # lines ke andar jo closed regions bane wo screens hain
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    found = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 3000 or area > 0.5 * img.shape[0] * img.shape[1]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        roi = gray[y:y + h, x:x + w]
        # bright hai to screen on, dark hai to off
        status = "ON" if roi.mean() > 90 else "OFF"
        color = (0, 255, 0) if status == "ON" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, status, (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        found += 1
 
    print("Screens found:", found)
    if found < expected_screens:
        print("Anomaly: %d screen(s) missing!" % (expected_screens - found))
 
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Screen Detection")
    plt.axis("off")
    plt.show()
 
