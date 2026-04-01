import os
import json
import math
import argparse
import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_vector(x, y):
    n = math.sqrt(x * x + y * y)
    if n < 1e-8:
        return 0.0, -1.0
    return x / n, y / n


# =========================
# Hand mask
# =========================

def build_hand_mask(image_shape, hand_json, dilate_kernel=11, dilate_iter=1):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if hand_json.get("num_hands", 0) == 0:
        return mask

    for hand in hand_json.get("hands", []):
        pts = []
        for lm in hand.get("landmarks", []):
            pts.append([int(lm["x_px"]), int(lm["y_px"])])

        if len(pts) < 3:
            continue

        pts = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    if dilate_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask


# =========================
# Edge extraction
# =========================

def extract_edges(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    return edges


def remove_hand_from_edges(edges, hand_mask):
    out = edges.copy()
    out[hand_mask > 0] = 0
    return out


def collect_edge_points(edge_mask, x_offset=0, y_offset=0):
    ys, xs = np.where(edge_mask > 0)
    if len(xs) == 0:
        return None
    pts = np.stack([xs + x_offset, ys + y_offset], axis=1).astype(np.float32)
    return pts


# =========================
# Pose refinement helpers
# =========================

def get_roi_from_ref_pose(ref_pose, image_shape, margin_ratio=0.35, min_margin=40):
    cx = ref_pose["center_x"]
    cy = ref_pose["center_y"]
    L = ref_pose["mouse_length_px"]
    W = ref_pose["mouse_width_px"]

    half_w = max(L, W) * 0.5
    margin = max(int(max(L, W) * margin_ratio), min_margin)

    x1 = max(0, int(round(cx - half_w - margin)))
    y1 = max(0, int(round(cy - half_w - margin)))
    x2 = min(image_shape[1], int(round(cx + half_w + margin)))
    y2 = min(image_shape[0], int(round(cy + half_w + margin)))

    return x1, y1, x2, y2


def estimate_translation_from_edges(ref_edge_pts, cur_edge_pts):
    """
    Simple translation estimate from average edge locations.
    """
    if ref_edge_pts is None or cur_edge_pts is None:
        return 0.0, 0.0

    ref_mean = np.mean(ref_edge_pts, axis=0)
    cur_mean = np.mean(cur_edge_pts, axis=0)

    dx = float(cur_mean[0] - ref_mean[0])
    dy = float(cur_mean[1] - ref_mean[1])

    return dx, dy


def estimate_small_rotation_from_edges(cur_edge_pts, ref_axis):
    """
    Estimate only a small rotation from visible edges.
    If points are insufficient, keep reference axis.
    """
    if cur_edge_pts is None or len(cur_edge_pts) < 20:
        return ref_axis

    mean = np.mean(cur_edge_pts, axis=0)
    centered = cur_edge_pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(eigvals)[::-1]
    axis = eigvecs[:, order[0]]

    ux, uy = float(axis[0]), float(axis[1])
    ux, uy = normalize_vector(ux, uy)

    rx, ry = ref_axis
    if ux * rx + uy * ry < 0:
        ux, uy = -ux, -uy

    return ux, uy


def build_current_pose(ref_pose, dx, dy, new_major_axis=None):
    cx = ref_pose["center_x"] + dx
    cy = ref_pose["center_y"] + dy

    if new_major_axis is None:
        major_x = ref_pose["major_axis_x"]
        major_y = ref_pose["major_axis_y"]
    else:
        major_x, major_y = new_major_axis
        major_x, major_y = normalize_vector(major_x, major_y)

    # perpendicular axis
    minor_x = -major_y
    minor_y = major_x

    pose = {
        "center_x": float(cx),
        "center_y": float(cy),
        "major_axis_x": float(major_x),
        "major_axis_y": float(major_y),
        "minor_axis_x": float(minor_x),
        "minor_axis_y": float(minor_y),
        "mouse_length_px": float(ref_pose["mouse_length_px"]),
        "mouse_width_px": float(ref_pose["mouse_width_px"]),
        "real_length_mm": ref_pose.get("real_length_mm"),
        "real_width_mm": ref_pose.get("real_width_mm"),
        "mm_per_pixel_length": ref_pose.get("mm_per_pixel_length"),
        "mm_per_pixel_width": ref_pose.get("mm_per_pixel_width"),
        "dx_from_ref": float(dx),
        "dy_from_ref": float(dy)
    }

    return pose


def draw_pose(vis, pose, color_center=(0, 0, 255), color_major=(255, 0, 255), color_minor=(0, 165, 255), label="current"):
    cx = pose["center_x"]
    cy = pose["center_y"]
    ux = pose["major_axis_x"]
    uy = pose["major_axis_y"]
    vx = pose["minor_axis_x"]
    vy = pose["minor_axis_y"]
    L = pose["mouse_length_px"]
    W = pose["mouse_width_px"]

    half_L = int(round(L * 0.5))
    half_W = int(round(W * 0.5))

    center = (int(round(cx)), int(round(cy)))
    major_p1 = (int(round(cx - ux * half_L)), int(round(cy - uy * half_L)))
    major_p2 = (int(round(cx + ux * half_L)), int(round(cy + uy * half_L)))
    minor_p1 = (int(round(cx - vx * half_W)), int(round(cy - vy * half_W)))
    minor_p2 = (int(round(cx + vx * half_W)), int(round(cy + vy * half_W)))

    cv2.circle(vis, center, 5, color_center, -1)
    cv2.line(vis, major_p1, major_p2, color_major, 2)
    cv2.line(vis, minor_p1, minor_p2, color_minor, 2)

    cv2.putText(vis, f"{label} center", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_center, 2)


# =========================
# Main refinement pipeline
# =========================

def refine_mouse_pose_with_hand(ref_image_bgr, cur_image_bgr, ref_pose, hand_json):
    vis = cur_image_bgr.copy()

    # 1. ROI around reference pose
    x1, y1, x2, y2 = get_roi_from_ref_pose(ref_pose, cur_image_bgr.shape)
    ref_roi = ref_image_bgr[y1:y2, x1:x2].copy()
    cur_roi = cur_image_bgr[y1:y2, x1:x2].copy()

    # 2. hand mask on current image
    hand_mask_full = build_hand_mask(cur_image_bgr.shape, hand_json, dilate_kernel=11, dilate_iter=1)
    hand_mask_roi = hand_mask_full[y1:y2, x1:x2]

    # 3. edges
    ref_edges = extract_edges(ref_roi)
    cur_edges = extract_edges(cur_roi)

    # remove hand from current edges only
    cur_edges_no_hand = remove_hand_from_edges(cur_edges, hand_mask_roi)

    ref_pts = collect_edge_points(ref_edges, x_offset=x1, y_offset=y1)
    cur_pts = collect_edge_points(cur_edges_no_hand, x_offset=x1, y_offset=y1)

    # 4. translation estimate
    dx, dy = estimate_translation_from_edges(ref_pts, cur_pts)

    # clamp translation to avoid crazy jumps
    dx = max(-40.0, min(40.0, dx))
    dy = max(-40.0, min(40.0, dy))

    # 5. orientation estimate (small correction only)
    ref_axis = (ref_pose["major_axis_x"], ref_pose["major_axis_y"])
    est_axis = estimate_small_rotation_from_edges(cur_pts, ref_axis)

    # Blend with ref axis to avoid over-rotation
    blended_x = 0.8 * ref_axis[0] + 0.2 * est_axis[0]
    blended_y = 0.8 * ref_axis[1] + 0.2 * est_axis[1]
    blended_x, blended_y = normalize_vector(blended_x, blended_y)

    current_pose = build_current_pose(ref_pose, dx, dy, new_major_axis=(blended_x, blended_y))

    # 6. visualization
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # overlay hand mask
    overlay = vis.copy()
    overlay[hand_mask_full > 0] = (0, 0, 180)
    vis = cv2.addWeighted(vis, 0.88, overlay, 0.12, 0)

    # draw reference pose in yellow-ish
    draw_pose(
        vis,
        ref_pose,
        color_center=(0, 255, 255),
        color_major=(0, 255, 255),
        color_minor=(0, 200, 200),
        label="ref"
    )

    # draw current pose in standard colors
    draw_pose(
        vis,
        current_pose,
        color_center=(0, 0, 255),
        color_major=(255, 0, 255),
        color_minor=(0, 165, 255),
        label="current"
    )

    cv2.putText(vis, f"dx={dx:.1f}, dy={dy:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    debug = {
        "hand_mask_full": hand_mask_full,
        "ref_edges": ref_edges,
        "cur_edges_no_hand": cur_edges_no_hand
    }

    return current_pose, vis, debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", required=True, help="Reference mouse-only image")
    parser.add_argument("--cur_image", required=True, help="Current hand-on-mouse image")
    parser.add_argument("--ref_pose", required=True, help="Reference mouse pose json")
    parser.add_argument("--hand_json", required=True, help="Current hand landmark json")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--output_name", default="mouse_pose_current.json", help="Output pose filename")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    ref_image = cv2.imread(args.ref_image)
    cur_image = cv2.imread(args.cur_image)

    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found: {args.ref_image}")
    if cur_image is None:
        raise FileNotFoundError(f"Current image not found: {args.cur_image}")
    if not os.path.exists(args.ref_pose):
        raise FileNotFoundError(f"Reference pose not found: {args.ref_pose}")
    if not os.path.exists(args.hand_json):
        raise FileNotFoundError(f"Hand json not found: {args.hand_json}")

    ref_pose = load_json(args.ref_pose)
    hand_json = load_json(args.hand_json)

    current_pose, vis, debug = refine_mouse_pose_with_hand(
        ref_image_bgr=ref_image,
        cur_image_bgr=cur_image,
        ref_pose=ref_pose,
        hand_json=hand_json
    )

    base = os.path.splitext(os.path.basename(args.cur_image))[0]

    pose_path = os.path.join(args.output_dir, args.output_name)
    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_refined_vis.jpg")
    hand_mask_path = os.path.join(args.output_dir, f"{base}_hand_mask_small.png")
    ref_edges_path = os.path.join(args.output_dir, f"{base}_ref_edges.png")
    cur_edges_path = os.path.join(args.output_dir, f"{base}_cur_edges_no_hand.png")

    save_json(pose_path, current_pose)
    cv2.imwrite(vis_path, vis)
    cv2.imwrite(hand_mask_path, debug["hand_mask_full"])
    cv2.imwrite(ref_edges_path, debug["ref_edges"])
    cv2.imwrite(cur_edges_path, debug["cur_edges_no_hand"])

    print("=== Refined Mouse Pose ===")
    for k, v in current_pose.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(pose_path)
    print(vis_path)
    print(hand_mask_path)
    print(ref_edges_path)
    print(cur_edges_path)


if __name__ == "__main__":
    main()