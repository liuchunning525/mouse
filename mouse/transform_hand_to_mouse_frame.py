import os
import json
import math
import argparse
import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_vector(x, y):
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-8:
        return 0.0, -1.0
    return x / norm, y / norm


def transform_point_to_mouse_frame(px, py, mouse_template):
    """
    Convert image pixel coordinates into mouse-local normalized coordinates.

    Mouse frame:
    - origin: mouse center
    - a-axis: along mouse major axis (front/back)
    - b-axis: along mouse minor axis (left/right)

    Returns:
        a_px, b_px, a_norm, b_norm
    """
    cx = mouse_template["center_x"]
    cy = mouse_template["center_y"]

    ux = mouse_template["major_axis_x"]
    uy = mouse_template["major_axis_y"]

    vx = mouse_template["minor_axis_x"]
    vy = mouse_template["minor_axis_y"]

    L = mouse_template["mouse_length_px"]
    W = mouse_template["mouse_width_px"]

    dx = px - cx
    dy = py - cy

    a_px = dx * ux + dy * uy
    b_px = dx * vx + dy * vy

    a_norm = a_px / L if L > 1e-8 else 0.0
    b_norm = b_px / W if W > 1e-8 else 0.0

    return a_px, b_px, a_norm, b_norm


def add_mouse_frame_to_hand_data(hand_data, mouse_template):
    result = {
        "num_hands": hand_data.get("num_hands", 0),
        "mouse_template": mouse_template,
        "hands": []
    }

    for hand in hand_data.get("hands", []):
        new_hand = {
            "hand_index": hand["hand_index"],
            "handedness": hand["handedness"],
            "handedness_score": hand["handedness_score"],
            "bbox": hand["bbox"],
            "landmarks_mouse_frame": []
        }

        for lm in hand["landmarks"]:
            x_px = lm["x_px"]
            y_px = lm["y_px"]

            a_px, b_px, a_norm, b_norm = transform_point_to_mouse_frame(
                x_px, y_px, mouse_template
            )

            new_hand["landmarks_mouse_frame"].append({
                "id": lm["id"],
                "x_px": x_px,
                "y_px": y_px,
                "x_norm": lm["x_norm"],
                "y_norm": lm["y_norm"],
                "z_norm": lm["z_norm"],
                "a_px": a_px,
                "b_px": b_px,
                "a_norm": a_norm,
                "b_norm": b_norm
            })

        result["hands"].append(new_hand)

    return result


def draw_mouse_frame_overlay(image, mouse_template):
    vis = image.copy()

    cx = int(round(mouse_template["center_x"]))
    cy = int(round(mouse_template["center_y"]))

    ux = mouse_template["major_axis_x"]
    uy = mouse_template["major_axis_y"]
    vx = mouse_template["minor_axis_x"]
    vy = mouse_template["minor_axis_y"]

    L = mouse_template["mouse_length_px"]
    W = mouse_template["mouse_width_px"]

    half_L = int(round(L * 0.5))
    half_W = int(round(W * 0.5))

    major_p1 = (int(cx - ux * half_L), int(cy - uy * half_L))
    major_p2 = (int(cx + ux * half_L), int(cy + uy * half_L))

    minor_p1 = (int(cx - vx * half_W), int(cy - vy * half_W))
    minor_p2 = (int(cx + vx * half_W), int(cy + vy * half_W))

    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(vis, major_p1, major_p2, (255, 0, 255), 2)   # major axis
    cv2.line(vis, minor_p1, minor_p2, (0, 165, 255), 2)   # minor axis

    cv2.putText(vis, "Mouse Center", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return vis


def draw_hand_points_in_mouse_frame(image, transformed_data):
    vis = image.copy()

    for hand in transformed_data.get("hands", []):
        for lm in hand["landmarks_mouse_frame"]:
            x = int(lm["x_px"])
            y = int(lm["y_px"])
            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)

            label = f"{lm['id']}"
            cv2.putText(vis, label, (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return vis


def print_key_landmarks(transformed_data):
    key_ids = [0, 4, 8, 12, 16, 20]  # wrist, thumb, index, middle, ring, pinky tip

    print("=== Hand Landmarks in Mouse Frame ===")
    for hand in transformed_data.get("hands", []):
        print(f"\nHand {hand['hand_index']} ({hand['handedness']}):")
        for lm in hand["landmarks_mouse_frame"]:
            if lm["id"] in key_ids:
                print(
                    f"  id={lm['id']:2d} | "
                    f"a_norm={lm['a_norm']:.3f}, "
                    f"b_norm={lm['b_norm']:.3f}, "
                    f"z_norm={lm['z_norm']:.3f}"
                )


def save_outputs(output_dir, image_path, transformed_data, vis_image):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base}_hand_mouse_frame.json")
    vis_path = os.path.join(output_dir, f"{base}_hand_mouse_frame_vis.jpg")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)

    cv2.imwrite(vis_path, vis_image)

    return json_path, vis_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Original image path")
    parser.add_argument("--mouse_template", type=str, required=True, help="Path to mouse_template.json")
    parser.add_argument("--hand_json", type=str, required=True, help="Path to *_hand_landmarks.json")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.mouse_template):
        raise FileNotFoundError(f"Mouse template not found: {args.mouse_template}")
    if not os.path.exists(args.hand_json):
        raise FileNotFoundError(f"Hand json not found: {args.hand_json}")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    mouse_template = load_json(args.mouse_template)
    hand_data = load_json(args.hand_json)

    transformed_data = add_mouse_frame_to_hand_data(hand_data, mouse_template)

    vis = draw_mouse_frame_overlay(image, mouse_template)
    vis = draw_hand_points_in_mouse_frame(vis, transformed_data)

    print_key_landmarks(transformed_data)

    json_path, vis_path = save_outputs(
        args.output_dir, args.image, transformed_data, vis
    )

    print("\nSaved:")
    print(json_path)
    print(vis_path)


if __name__ == "__main__":
    main()