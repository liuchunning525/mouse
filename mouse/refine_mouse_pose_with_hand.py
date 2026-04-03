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


def merge_roi(work_area_roi, ref_pose, extra_margin=50):
    wx1 = work_area_roi["x1"]
    wy1 = work_area_roi["y1"]
    wx2 = work_area_roi["x2"]
    wy2 = work_area_roi["y2"]

    cx = ref_pose["center_x"]
    cy = ref_pose["center_y"]
    L = ref_pose["mouse_length_px"]
    W = ref_pose["mouse_width_px"]

    radius = max(L, W) * 0.75 + extra_margin

    x1 = max(wx1, int(round(cx - radius)))
    y1 = max(wy1, int(round(cy - radius)))
    x2 = min(wx2, int(round(cx + radius)))
    y2 = min(wy2, int(round(cy + radius)))

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "width": int(x2 - x1),
        "height": int(y2 - y1)
    }


def extract_mouse_template(ref_image_bgr, ref_pose, padding=10):
    cx = ref_pose["center_x"]
    cy = ref_pose["center_y"]
    L = ref_pose["mouse_length_px"]
    W = ref_pose["mouse_width_px"]

    # 模板裁得更紧，只包住鼠标主体
    half_x = W * 0.65 + padding
    half_y = L * 0.55 + padding

    x1 = max(0, int(round(cx - half_x)))
    y1 = max(0, int(round(cy - half_y)))
    x2 = min(ref_image_bgr.shape[1], int(round(cx + half_x)))
    y2 = min(ref_image_bgr.shape[0], int(round(cy + half_y)))

    template = ref_image_bgr[y1:y2, x1:x2].copy()

    info = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "center_x_local": float(cx - x1),
        "center_y_local": float(cy - y1)
    }

    return template, info

def extract_tail_template(ref_image_bgr, ref_pose, padding_x=20, padding_y=20):
    """
    Extract a small template around the mouse tail / cable connection area.
    """
    cx = ref_pose["center_x"]
    cy = ref_pose["center_y"]
    L = ref_pose["mouse_length_px"]
    W = ref_pose["mouse_width_px"]

    x1 = max(0, int(round(cx - W * 0.25 - padding_x)))
    x2 = min(ref_image_bgr.shape[1], int(round(cx + W * 0.25 + padding_x)))

    y1 = max(0, int(round(cy + L * 0.25 - padding_y)))
    y2 = min(ref_image_bgr.shape[0], int(round(cy + L * 0.55 + padding_y)))

    patch = ref_image_bgr[y1:y2, x1:x2].copy()

    info = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "anchor_x_local": float(cx - x1),
        "anchor_y_local": float(cy - y1)
    }
    return patch, info


def extract_wheel_template(ref_image_bgr, ref_pose, padding_x=20, padding_y=20):
    """
    Extract a small template around the mouse wheel / center line region.
    """
    cx = ref_pose["center_x"]
    cy = ref_pose["center_y"]
    L = ref_pose["mouse_length_px"]
    W = ref_pose["mouse_width_px"]

    x1 = max(0, int(round(cx - W * 0.18 - padding_x)))
    x2 = min(ref_image_bgr.shape[1], int(round(cx + W * 0.18 + padding_x)))

    y1 = max(0, int(round(cy - L * 0.10 - padding_y)))
    y2 = min(ref_image_bgr.shape[0], int(round(cy + L * 0.18 + padding_y)))

    patch = ref_image_bgr[y1:y2, x1:x2].copy()

    info = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "anchor_x_local": float(cx - x1),
        "anchor_y_local": float(cy - y1)
    }
    return patch, info


def safe_match_template(search_bgr, template_bgr, hand_mask_search=None):
    """
    Gray template matching with simple hand suppression.
    Returns score and top-left match location.
    """
    search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    search_proc = search_gray.copy().astype(np.float32)

    if hand_mask_search is not None:
        search_proc[hand_mask_search > 0] = 255.0

    search_proc = np.clip(search_proc, 0, 255).astype(np.uint8)

    if template_gray.shape[0] > search_proc.shape[0] or template_gray.shape[1] > search_proc.shape[1]:
        scale_h = search_proc.shape[0] / template_gray.shape[0]
        scale_w = search_proc.shape[1] / template_gray.shape[1]
        scale = min(scale_h, scale_w) * 0.9

        new_w = max(20, int(template_gray.shape[1] * scale))
        new_h = max(20, int(template_gray.shape[0] * scale))
        template_gray = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    result = cv2.matchTemplate(search_proc, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return float(max_val), (int(max_loc[0]), int(max_loc[1])), template_gray.shape[1], template_gray.shape[0]


def masked_template_match(search_bgr, template_bgr, hand_mask_search=None):
    search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    search_proc = search_gray.copy()

    if hand_mask_search is not None:
        search_proc = search_proc.astype(np.float32)
        search_proc[hand_mask_search > 0] = 255.0
        search_proc = np.clip(search_proc, 0, 255).astype(np.uint8)

    if template_gray.shape[0] > search_proc.shape[0] or template_gray.shape[1] > search_proc.shape[1]:
        raise RuntimeError("Template is larger than search image.")

    result = cv2.matchTemplate(search_proc, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return float(max_val), (int(max_loc[0]), int(max_loc[1])), result


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


def estimate_small_rotation_from_edges(cur_edge_pts, ref_axis):
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


def build_current_pose(ref_pose, center_x, center_y, new_major_axis=None, match_score=None):
    if new_major_axis is None:
        major_x = ref_pose["major_axis_x"]
        major_y = ref_pose["major_axis_y"]
    else:
        major_x, major_y = normalize_vector(new_major_axis[0], new_major_axis[1])

    minor_x = -major_y
    minor_y = major_x

    dx = float(center_x - ref_pose["center_x"])
    dy = float(center_y - ref_pose["center_y"])

    pose = {
        "center_x": float(center_x),
        "center_y": float(center_y),
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
        "dx_from_ref": dx,
        "dy_from_ref": dy,
        "match_score": float(match_score) if match_score is not None else None
    }

    return pose

def contour_list_to_array(contour_list):
    """
    [[x,y], ...] -> np.ndarray [N,2]
    """
    return np.array(contour_list, dtype=np.float32)


def angle_from_axis(axis_x, axis_y):
    return math.atan2(axis_y, axis_x)


def transform_reference_contour(ref_pose, current_pose, reference_contour_list):
    """
    Transform reference contour from reference mouse pose to current mouse pose.
    """
    ref_pts = contour_list_to_array(reference_contour_list)

    # reference pose
    rcx = ref_pose["center_x"]
    rcy = ref_pose["center_y"]
    rux = ref_pose["major_axis_x"]
    ruy = ref_pose["major_axis_y"]
    rL = ref_pose["mouse_length_px"]
    rW = ref_pose["mouse_width_px"]

    # current pose
    ccx = current_pose["center_x"]
    ccy = current_pose["center_y"]
    cux = current_pose["major_axis_x"]
    cuy = current_pose["major_axis_y"]
    cL = current_pose["mouse_length_px"]
    cW = current_pose["mouse_width_px"]

    # move points to reference center
    pts = ref_pts - np.array([[rcx, rcy]], dtype=np.float32)

    # rotate reference major axis to x-axis
    theta_ref = angle_from_axis(rux, ruy)
    R_ref_inv = np.array([
        [ math.cos(-theta_ref), -math.sin(-theta_ref)],
        [ math.sin(-theta_ref),  math.cos(-theta_ref)]
    ], dtype=np.float32)

    pts_local = pts @ R_ref_inv.T

    # anisotropic scaling
    scale_x = cW / rW if abs(rW) > 1e-8 else 1.0
    scale_y = cL / rL if abs(rL) > 1e-8 else 1.0

    # 注意：主轴方向对应 length，副轴方向对应 width
    # 在 local 坐标中，x 近似副轴，y 近似主轴
    pts_local[:, 0] *= scale_x
    pts_local[:, 1] *= scale_y

    # rotate local contour to current axis
    theta_cur = angle_from_axis(cux, cuy)
    R_cur = np.array([
        [ math.cos(theta_cur), -math.sin(theta_cur)],
        [ math.sin(theta_cur),  math.cos(theta_cur)]
    ], dtype=np.float32)

    pts_out = pts_local @ R_cur.T

    # move to current center
    pts_out += np.array([[ccx, ccy]], dtype=np.float32)

    return pts_out


def draw_transformed_contour(vis, pts, color=(0, 255, 0), thickness=3):
    contour = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(vis, [contour], -1, color, thickness)

def draw_pose(vis, pose, color_center, color_major, color_minor, label):
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


def refine_mouse_pose_with_hand(ref_image_bgr, cur_image_bgr, ref_pose, work_area_roi, hand_json):
    vis = cur_image_bgr.copy()

    refine_roi = merge_roi(work_area_roi, ref_pose, extra_margin=50)
    rx1, ry1, rx2, ry2 = refine_roi["x1"], refine_roi["y1"], refine_roi["x2"], refine_roi["y2"]

    search_img = cur_image_bgr[ry1:ry2, rx1:rx2].copy()

    hand_mask_full = build_hand_mask(cur_image_bgr.shape, hand_json, dilate_kernel=11, dilate_iter=1)
    hand_mask_search = hand_mask_full[ry1:ry2, rx1:rx2]

    # --- new: tail + wheel local matching ---
    tail_template, tail_info = extract_tail_template(ref_image_bgr, ref_pose, padding_x=20, padding_y=20)
    wheel_template, wheel_info = extract_wheel_template(ref_image_bgr, ref_pose, padding_x=20, padding_y=20)

    tail_score, tail_loc, tail_w, tail_h = safe_match_template(
        search_bgr=search_img,
        template_bgr=tail_template,
        hand_mask_search=hand_mask_search
    )

    wheel_score, wheel_loc, wheel_w, wheel_h = safe_match_template(
        search_bgr=search_img,
        template_bgr=wheel_template,
        hand_mask_search=hand_mask_search
    )

    # matched top-left in full image
    tail_x_full = rx1 + tail_loc[0]
    tail_y_full = ry1 + tail_loc[1]

    wheel_x_full = rx1 + wheel_loc[0]
    wheel_y_full = ry1 + wheel_loc[1]

    # infer center from each anchor
    center_from_tail_x = tail_x_full + tail_info["anchor_x_local"]
    center_from_tail_y = tail_y_full + tail_info["anchor_y_local"]

    center_from_wheel_x = wheel_x_full + wheel_info["anchor_x_local"]
    center_from_wheel_y = wheel_y_full + wheel_info["anchor_y_local"]

    # weighted fusion
    total_score = max(1e-6, tail_score + wheel_score)
    center_x = (tail_score * center_from_tail_x + wheel_score * center_from_wheel_x) / total_score
    center_y = (tail_score * center_from_tail_y + wheel_score * center_from_wheel_y) / total_score

    match_score = max(tail_score, wheel_score)

    # small angle correction from wheel region
    patch = cur_image_bgr[wheel_y_full:wheel_y_full + wheel_h, wheel_x_full:wheel_x_full + wheel_w].copy()
    hand_mask_patch = hand_mask_full[wheel_y_full:wheel_y_full + wheel_h, wheel_x_full:wheel_x_full + wheel_w]

    patch_edges = extract_edges(patch)
    patch_edges_no_hand = remove_hand_from_edges(patch_edges, hand_mask_patch)
    patch_pts = collect_edge_points(patch_edges_no_hand, x_offset=wheel_x_full, y_offset=wheel_y_full)

    ref_axis = (ref_pose["major_axis_x"], ref_pose["major_axis_y"])
    est_axis = estimate_small_rotation_from_edges(patch_pts, ref_axis)

    blended_x = 0.9 * ref_axis[0] + 0.1 * est_axis[0]
    blended_y = 0.9 * ref_axis[1] + 0.1 * est_axis[1]
    blended_x, blended_y = normalize_vector(blended_x, blended_y)

    current_pose = build_current_pose(
        ref_pose=ref_pose,
        center_x=center_x,
        center_y=center_y,
        new_major_axis=(blended_x, blended_y),
        match_score=match_score
    )

    transformed_contour = None
    if "reference_contour" in ref_pose and ref_pose["reference_contour"]:
        transformed_contour = transform_reference_contour(
            ref_pose=ref_pose,
            current_pose=current_pose,
            reference_contour_list=ref_pose["reference_contour"]
        )
        current_pose["transformed_contour"] = [
            [float(p[0]), float(p[1])] for p in transformed_contour
        ]

    # visualization
    cv2.rectangle(vis, (work_area_roi["x1"], work_area_roi["y1"]),
                  (work_area_roi["x2"], work_area_roi["y2"]), (255, 255, 0), 2)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

    # tail / wheel matched patch
    cv2.rectangle(vis, (tail_x_full, tail_y_full), (tail_x_full + tail_w, tail_y_full + tail_h), (255, 0, 0), 2)
    cv2.rectangle(vis, (wheel_x_full, wheel_y_full), (wheel_x_full + wheel_w, wheel_y_full + wheel_h), (0, 255, 0), 2)

    overlay = vis.copy()
    overlay[hand_mask_full > 0] = (0, 0, 180)
    vis = cv2.addWeighted(vis, 0.88, overlay, 0.12, 0)

    draw_pose(
        vis, ref_pose,
        color_center=(0, 255, 255),
        color_major=(0, 255, 255),
        color_minor=(0, 200, 200),
        label="ref"
    )

    draw_pose(
        vis, current_pose,
        color_center=(0, 0, 255),
        color_major=(255, 0, 255),
        color_minor=(0, 165, 255),
        label="current"
    )

    if transformed_contour is not None:
        draw_transformed_contour(vis, transformed_contour, color=(0, 255, 0), thickness=3)

    cv2.putText(vis, f"dx={current_pose['dx_from_ref']:.1f}, dy={current_pose['dy_from_ref']:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, f"tail={tail_score:.3f}, wheel={wheel_score:.3f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    debug = {
        "hand_mask_full": hand_mask_full,
        "tail_template": tail_template,
        "wheel_template": wheel_template,
        "patch_edges_no_hand": patch_edges_no_hand,
        "refine_roi": refine_roi
    }

    return current_pose, vis, debug

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", required=True, help="Mouse-only reference image")
    parser.add_argument("--cur_image", required=True, help="Current hand-on-mouse image")
    parser.add_argument("--ref_pose", required=True, help="Reference mouse pose json")
    parser.add_argument("--work_area_json", required=True, help="Work area json")
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
    if not os.path.exists(args.work_area_json):
        raise FileNotFoundError(f"Work area json not found: {args.work_area_json}")
    if not os.path.exists(args.hand_json):
        raise FileNotFoundError(f"Hand json not found: {args.hand_json}")

    ref_pose = load_json(args.ref_pose)
    work_area = load_json(args.work_area_json)
    hand_json = load_json(args.hand_json)

    work_area_roi = work_area["roi"]

    current_pose, vis, debug = refine_mouse_pose_with_hand(
        ref_image_bgr=ref_image,
        cur_image_bgr=cur_image,
        ref_pose=ref_pose,
        work_area_roi=work_area_roi,
        hand_json=hand_json
    )

    base = os.path.splitext(os.path.basename(args.cur_image))[0]

    pose_path = os.path.join(args.output_dir, args.output_name)
    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_refined_vis.jpg")
    hand_mask_path = os.path.join(args.output_dir, f"{base}_hand_mask_small.png")
    tail_template_path = os.path.join(args.output_dir, f"{base}_tail_template.jpg")
    wheel_template_path = os.path.join(args.output_dir, f"{base}_wheel_template.jpg")
    patch_edges_path = os.path.join(args.output_dir, f"{base}_patch_edges_no_hand.png")

    save_json(pose_path, current_pose)
    cv2.imwrite(vis_path, vis)
    cv2.imwrite(hand_mask_path, debug["hand_mask_full"])
    cv2.imwrite(tail_template_path, debug["tail_template"])
    cv2.imwrite(wheel_template_path, debug["wheel_template"])
    cv2.imwrite(patch_edges_path, debug["patch_edges_no_hand"])

    print("=== Refined Mouse Pose (Template Match) ===")
    print(json.dumps(current_pose, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(pose_path)
    print(vis_path)
    print(hand_mask_path)
    print(tail_template_path)
    print(wheel_template_path)
    print(patch_edges_path)


if __name__ == "__main__":
    main()