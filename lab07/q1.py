import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_image():
    img = cv2.imread("image.jpg")
    if img is None:
        img = np.full((300, 400, 3), 255, np.uint8)
        for i in range(0, 400, 40):
            cv2.line(img, (i, 0), (i, 300), (200, 200, 200), 1)
        for j in range(0, 300, 40):
            cv2.line(img, (0, j), (400, j), (200, 200, 200), 1)
        cv2.rectangle(img, (20, 20), (380, 280), (0, 0, 0), 3)
        cv2.putText(img, "LAB 07", (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 0, 0), 5)
    return img


def show(images, titles, name, cols=3):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (im, t) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(name)
    plt.show()


def part_a():
    img = make_image()
    h, w = img.shape[:2]

    T = np.float32([[1, 0, 60],
                    [0, 1, 40]])
    translated = cv2.warpAffine(img, T, (w + 60, h + 40))

    R = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
    rotated = cv2.warpAffine(img, R, (w, h))

    S = np.float32([[0.5, 0, 0],
                    [0, 0.5, 0]])
    scaled = cv2.warpAffine(img, S, (w // 2, h // 2))

    k = 0.4
    Sh = np.float32([[1, k, 0],
                     [0, 1, 0]])
    sheared = cv2.warpAffine(img, Sh, (int(w + k * h), h))

    show([img, translated, rotated, scaled, sheared],
         ["Original", "Translation", "Rotation 30", "Scaling 0.5", "Shear 0.4"],
         "q1a_2d_transforms.png")


def cube_points():
    pts = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    pts -= 0.5
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(pts[i] != pts[j]) == 1:
                edges.append((i, j))
    return pts, edges


def part_b():
    pts, edges = cube_points()

    ax_, ay_, az_ = np.radians([30, 40, 20])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax_), -np.sin(ax_)],
                   [0, np.sin(ax_), np.cos(ax_)]])
    Ry = np.array([[np.cos(ay_), 0, np.sin(ay_)],
                   [0, 1, 0],
                   [-np.sin(ay_), 0, np.cos(ay_)]])
    Rz = np.array([[np.cos(az_), -np.sin(az_), 0],
                   [np.sin(az_), np.cos(az_), 0],
                   [0, 0, 1]])

    rotated = pts @ (Rz @ Ry @ Rx).T
    moved = pts + np.array([2, 1, 3])

    fig = plt.figure(figsize=(15, 5))
    for i, (p, title) in enumerate([(pts, "Original cube"),
                                    (rotated, "Rotated about x, y, z"),
                                    (moved, "Moved by (2, 1, 3)")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for a, b in edges:
            ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], [p[a, 2], p[b, 2]], "b-")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="r")
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_zlim(-1, 4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("q1b_3d_transform.png")
    plt.show()


def part_c():
    x, y = 3, 4
    p = np.array([x, y, 1])
    print("2D point:", (x, y))
    print("Homogeneous form:", p)

    T = np.array([[1, 0, 5],
                  [0, 1, 2],
                  [0, 0, 1]])
    print("Translation by (5, 2) using 3x3 matrix:")
    print(T @ p)

    M2 = np.array([[1, 0],
                   [0, 1]])
    print("A 2x2 matrix can only map (0,0) to (0,0), so it cannot translate:")
    print(M2 @ np.array([0, 0]))

    scaled_point = np.array([6, 8, 2])
    print("Scaled homogeneous point:", scaled_point)
    print("Converted back to 2D:", scaled_point[:2] / scaled_point[2])

    print()
    print("Extra coordinate is used because:")
    print("1. Translation becomes a matrix multiplication like rotation and scaling.")
    print("2. Many transformations can be combined into one 3x3 matrix.")
    print("3. Projective transformations (perspective) need the divide by w.")
    print("4. Points at infinity can be represented with w = 0.")


def part_d():
    h, w = 300, 400
    img = np.full((h, w, 3), 255, np.uint8)
    for i in range(0, w, 40):
        cv2.line(img, (i, 0), (i, h), (0, 0, 0), 1)
    for j in range(0, h, 40):
        cv2.line(img, (0, j), (w, j), (0, 0, 0), 1)
    cv2.circle(img, (200, 150), 60, (0, 0, 255), 3)

    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst = np.float32([[40, 30], [330, 70], [10, 260]])
    M = cv2.getAffineTransform(src, dst)
    print("Affine matrix:")
    print(M)

    out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

    line_pts = np.array([[10, 10, 1], [110, 60, 1], [210, 110, 1]], dtype=float).T
    new_pts = M @ line_pts
    v1 = new_pts[:, 1] - new_pts[:, 0]
    v2 = new_pts[:, 2] - new_pts[:, 0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    print("Collinearity check after transform (should be ~0):", round(cross, 6))

    par1 = M[:, :2] @ np.array([1, 0])
    par2 = M[:, :2] @ np.array([5, 0])
    print("Parallel lines stay parallel:", np.isclose(par1[0] * par2[1] - par1[1] * par2[0], 0))
    print("Straight lines remain straight, parallel lines stay parallel.")
    print("Angles and lengths do not stay the same.")

    show([img, out], ["Original", "Affine transformed"], "q1d_affine.png", cols=2)


def part_e():
    book = cv2.imread("book.jpg")
    if book is None:
        flat = np.full((400, 300, 3), (60, 120, 200), np.uint8)
        cv2.rectangle(flat, (10, 10), (290, 390), (255, 255, 255), 4)
        cv2.putText(flat, "BOOK", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        flat_pts = np.float32([[0, 0], [299, 0], [299, 399], [0, 399]])
        tilt_pts = np.float32([[120, 60], [420, 20], [470, 440], [90, 380]])
        Mt = cv2.getPerspectiveTransform(flat_pts, tilt_pts)
        book = cv2.warpPerspective(flat, Mt, (560, 480), borderValue=(255, 255, 255))
        src = tilt_pts
    else:
        src = np.float32([[120, 60], [420, 20], [470, 440], [90, 380]])

    W, H = 300, 400
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])

    Hm = cv2.getPerspectiveTransform(src, dst)
    print("Homography matrix:")
    print(Hm)
    flat_out = cv2.warpPerspective(book, Hm, (W, H))

    show([book, flat_out], ["Tilted book", "Corrected (flat)"], "q1e_homography.png", cols=2)


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
