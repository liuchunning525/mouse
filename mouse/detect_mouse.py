import cv2
import numpy as np
import json
import os
import argparse


def normalize_vector(x, y):
    norm = np.sqrt(x * x + y * y)
    if norm < 1e-8:
        return 0.0, -1.0
    return x / norm, y / norm


def preprocess_for_mouse_mask(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 35, 110)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    closed = cv2.erode(closed, kernel, iterations=1)

    return closed


def contour_score(contour: np.ndarray, image_shape):
    area = cv2.contourArea(contour)
    if area < 1000:
        return -1e9

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    if rect_area <= 0:
        return -1e9

    extent = area / rect_area
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return -1e9

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)

    rr = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rr
    if rw < 1 or rh < 1:
        return -1e9

    long_side = max(rw, rh)
    short_side = min(rw, rh)
    aspect_ratio = long_side / short_side if short_side > 0 else 999.0

    if aspect_ratio < 1.1 or aspect_ratio > 4.8:
        return -1e9

    score = 0.0
    score += area * 0.01
    score += extent * 120.0
    score += circularity * 40.0
    score += (1.0 - abs(aspect_ratio - 1.8)) * 35.0

    return score


def find_best_mouse_contour(mask: np.ndarray, image_shape):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    best_contour = None
    best_score = -1e18

    for c in contours:
        s = contour_score(c, image_shape)
        if s > best_score:
            best_score = s
            best_contour = c

    return best_contour


def compute_pca_axis(contour: np.ndarray):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)

    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    ux, uy = float(eigvecs[0, 0]), float(eigvecs[1, 0])
    vx, vy = float(eigvecs[0, 1]), float(eigvecs[1, 1])

    ux, uy = normalize_vector(ux, uy)
    vx, vy = normalize_vector(vx, vy)

    if uy > 0:
        ux, uy = -ux, -uy
        vx, vy = -vx, -vy

    return mean, (ux, uy), (vx, vy), eigvals


def extract_mouse_features(image: np.ndarray, real_length_mm=None, real_width_mm=None):
    vis = image.copy()

    mask = preprocess_for_mouse_mask(image)
    contour = find_best_mouse_contour(mask, image.shape)

    if contour is None:
        raise RuntimeError("Mouse contour not found.")

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    x, y, bw, bh = cv2.boundingRect(contour)

    # 用 minAreaRect 的中心，不用质心
    rr = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rr
    box = cv2.boxPoints(rr).astype(np.int32)

    long_side = float(max(rw, rh))
    short_side = float(min(rw, rh))

    pca_center, major_axis, minor_axis, eigvals = compute_pca_axis(contour)

    aspect_ratio = long_side / short_side if short_side > 1e-8 else None
    extent = area / float(bw * bh) if bw * bh > 0 else None
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 1e-8 else None

    mm_per_pixel_length = None
    mm_per_pixel_width = None

    if real_length_mm is not None and long_side > 1e-8:
        mm_per_pixel_length = float(real_length_mm) / long_side

    if real_width_mm is not None and short_side > 1e-8:
        mm_per_pixel_width = float(real_width_mm) / short_side

    # draw
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
    cv2.drawContours(vis, [box], -1, (0, 255, 255), 2)

    center = (int(round(cx)), int(round(cy)))
    cv2.circle(vis, center, 5, (0, 0, 255), -1)

    half_long = int(round(long_side * 0.5))
    half_short = int(round(short_side * 0.5))

    ux, uy = major_axis
    vx, vy = minor_axis

    major_p1 = (int(round(cx - ux * half_long)), int(round(cy - uy * half_long)))
    major_p2 = (int(round(cx + ux * half_long)), int(round(cy + uy * half_long)))
    minor_p1 = (int(round(cx - vx * half_short)), int(round(cy - vy * half_short)))
    minor_p2 = (int(round(cx + vx * half_short)), int(round(cy + vy * half_short)))

    cv2.line(vis, major_p1, major_p2, (255, 0, 255), 2)
    cv2.line(vis, minor_p1, minor_p2, (0, 165, 255), 2)

    cv2.putText(vis, f"Center: ({int(round(cx))}, {int(round(cy))})",
                (x, max(20, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"L={long_side:.1f}px W={short_side:.1f}px",
                (x, max(45, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    features = {
        "contour_area_px2": area,
        "contour_perimeter_px": perimeter,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(bw),
        "bbox_h": int(bh),

        "center_x": float(cx),
        "center_y": float(cy),

        "mouse_length_px": long_side,
        "mouse_width_px": short_side,

        "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else None,
        "extent": float(extent) if extent is not None else None,
        "circularity": float(circularity) if circularity is not None else None,

        "major_axis_x": float(major_axis[0]),
        "major_axis_y": float(major_axis[1]),
        "minor_axis_x": float(minor_axis[0]),
        "minor_axis_y": float(minor_axis[1]),

        "real_length_mm": float(real_length_mm) if real_length_mm is not None else None,
        "real_width_mm": float(real_width_mm) if real_width_mm is not None else None,
        "mm_per_pixel_length": mm_per_pixel_length,
        "mm_per_pixel_width": mm_per_pixel_width
    }

    return features, vis, mask, contour


def save_outputs(features, vis, mask, output_dir, image_path):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    vis_path = os.path.join(output_dir, f"{base}_mouse_vis.jpg")
    mask_path = os.path.join(output_dir, f"{base}_mouse_mask.png")
    json_path = os.path.join(output_dir, f"{base}_mouse_features.json")

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(mask_path, mask)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    return vis_path, mask_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input mouse-only image")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--real_length_mm", type=float, default=None, help="Real mouse length in mm")
    parser.add_argument("--real_width_mm", type=float, default=None, help="Real mouse width in mm")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    features, vis, mask, _ = extract_mouse_features(
        image=image,
        real_length_mm=args.real_length_mm,
        real_width_mm=args.real_width_mm
    )

    vis_path, mask_path, json_path = save_outputs(features, vis, mask, args.output_dir, args.image)

    print("=== Mouse Detection Result ===")
    for k, v in features.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(vis_path)
    print(mask_path)
    print(json_path)


if __name__ == "__main__":
    main()