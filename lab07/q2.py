import numpy as np
import cv2


def apply_perspective_transform(img, src_pts, dst_pts, out_size):
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    result = cv2.warpPerspective(img, M, out_size)
    return result, M


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


img = cv2.imread("document.jpg")
if img is None:
    page = np.full((500, 350, 3), 255, np.uint8)
    cv2.rectangle(page, (5, 5), (344, 494), (0, 0, 0), 3)
    for i in range(8):
        cv2.putText(page, "Line of text " + str(i + 1), (30, 70 + i * 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    flat_corners = np.float32([[0, 0], [349, 0], [349, 499], [0, 499]])
    tilted_corners = np.float32([[150, 80], [520, 40], [560, 520], [90, 450]])
    Mt = cv2.getPerspectiveTransform(flat_corners, tilted_corners)
    img = cv2.warpPerspective(page, Mt, (640, 600), borderValue=(120, 120, 120))
    tilted = tilted_corners
else:
    tilted = np.float32([[150, 80], [520, 40], [560, 520], [90, 450]])

src = order_points(tilted)
tl, tr, br, bl = src

width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

dst = np.float32([[0, 0],
                  [width - 1, 0],
                  [width - 1, height - 1],
                  [0, height - 1]])

result, M = apply_perspective_transform(img, src, dst, (width, height))
print("Perspective matrix:")
print(M)

marked = img.copy()
cv2.polylines(marked, [src.astype(int)], True, (0, 255, 0), 3)

cv2.imwrite("q2_output.png", result)
cv2.imshow("Tilted view (source)", marked)
cv2.imshow("Head-on view (destination)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
