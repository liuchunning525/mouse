import os
import cv2
import json
import argparse
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Return points ordered as:
    [top-left, top-right, bottom-right, bottom-left]
    """
    pts = pts.astype(np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def detect_black_dots(image: np.ndarray,
                      min_area: int = 20,
                      max_area: int = 5000,
                      min_circularity: float = 0.35):
    """
    Detect black dots on white background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary inverse: black dots -> white blobs
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(c, True)
        if peri < 1e-6:
            continue

        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < min_circularity:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-8:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        x, y, w, h = cv2.boundingRect(c)

        candidates.append({
            "contour": c,
            "center": (float(cx), float(cy)),
            "area": float(area),
            "circularity": float(circularity),
            "bbox": (int(x), int(y), int(w), int(h))
        })

    return candidates, binary


def select_best_four_dots(candidates):
    """
    If more than 4 blobs are found, pick the 4 that most likely form the work area.
    Strategy:
    - prefer 4 with similar area
    - prefer larger convex hull area
    """
    if len(candidates) < 4:
        return None

    if len(candidates) == 4:
        return candidates

    # Brute force combinations of 4
    from itertools import combinations

    best_group = None
    best_score = -1e18

    for group in combinations(candidates, 4):
        pts = np.array([g["center"] for g in group], dtype=np.float32)
        hull = cv2.convexHull(pts.astype(np.float32))
        hull_area = cv2.contourArea(hull)

        areas = np.array([g["area"] for g in group], dtype=np.float32)
        area_std = float(np.std(areas))
        area_mean = float(np.mean(areas)) + 1e-6

        # prefer large spread and similar sizes
        score = hull_area - 30.0 * (area_std / area_mean)

        if score > best_score:
            best_score = score
            best_group = list(group)

    return best_group


def build_work_area_from_points(ordered_pts: np.ndarray, padding: int, image_shape):
    """
    Build axis-aligned ROI from 4 ordered corners, with optional padding.
    """
    xs = ordered_pts[:, 0]
    ys = ordered_pts[:, 1]

    x1 = max(0, int(np.floor(np.min(xs))) - padding)
    y1 = max(0, int(np.floor(np.min(ys))) - padding)
    x2 = min(image_shape[1] - 1, int(np.ceil(np.max(xs))) + padding)
    y2 = min(image_shape[0] - 1, int(np.ceil(np.max(ys))) + padding)

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "width": int(x2 - x1),
        "height": int(y2 - y1)
    }


def draw_result(image: np.ndarray, ordered_pts: np.ndarray, roi: dict, candidates: list):
    vis = image.copy()

    # draw all candidate centers
    for i, c in enumerate(candidates):
        cx, cy = c["center"]
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (255, 0, 255), -1)

    # draw polygon
    poly = ordered_pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=3)

    labels = ["TL", "TR", "BR", "BL"]
    for label, pt in zip(labels, ordered_pts):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(vis, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # draw ROI
    cv2.rectangle(vis, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), (255, 255, 0), 2)
    cv2.putText(vis, f"ROI: ({roi['x1']}, {roi['y1']}) - ({roi['x2']}, {roi['y2']})",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return vis


def save_outputs(output_dir, image_path, vis, binary, result):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    vis_path = os.path.join(output_dir, f"{base}_work_area_vis.jpg")
    mask_path = os.path.join(output_dir, f"{base}_work_area_mask.png")
    json_path = os.path.join(output_dir, f"{base}_work_area.json")

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(mask_path, binary)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return vis_path, mask_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image with 4 black dots")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--padding", type=int, default=20, help="ROI padding in pixels")
    parser.add_argument("--min_area", type=int, default=20, help="Minimum dot area")
    parser.add_argument("--max_area", type=int, default=5000, help="Maximum dot area")
    parser.add_argument("--min_circularity", type=float, default=0.35, help="Minimum circularity")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    candidates, binary = detect_black_dots(
        image=image,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity
    )

    if len(candidates) < 4:
        raise RuntimeError(f"Only found {len(candidates)} valid black dots. Need 4.")

    selected = select_best_four_dots(candidates)
    if selected is None:
        raise RuntimeError("Failed to select 4 black dots.")

    pts = np.array([c["center"] for c in selected], dtype=np.float32)
    ordered_pts = order_points(pts)

    roi = build_work_area_from_points(ordered_pts, padding=args.padding, image_shape=image.shape)
    vis = draw_result(image, ordered_pts, roi, selected)

    result = {
        "image": os.path.basename(args.image),
        "corners": {
            "top_left": [float(ordered_pts[0][0]), float(ordered_pts[0][1])],
            "top_right": [float(ordered_pts[1][0]), float(ordered_pts[1][1])],
            "bottom_right": [float(ordered_pts[2][0]), float(ordered_pts[2][1])],
            "bottom_left": [float(ordered_pts[3][0]), float(ordered_pts[3][1])]
        },
        "roi": roi
    }

    vis_path, mask_path, json_path = save_outputs(args.output_dir, args.image, vis, binary, result)

    print("=== Work Area Detection Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(vis_path)
    print(mask_path)
    print(json_path)


if __name__ == "__main__":
    main()