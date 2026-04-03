import os
import json
import argparse
import cv2

from detect_mouse_in_work_area import detect_mouse_in_roi, load_json


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def contour_to_list(contour):
    """
    Convert OpenCV contour [N,1,2] to plain python list [[x,y], ...]
    """
    pts = contour.reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in pts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Mouse-only reference image")
    parser.add_argument("--work_area_json", required=True, help="Work area json from detect_work_area.py")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--real_length_mm", type=float, default=None, help="Real mouse length in mm")
    parser.add_argument("--real_width_mm", type=float, default=None, help="Real mouse width in mm")
    parser.add_argument("--output_name", default="mouse_pose_ref.json", help="Output pose filename")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    work_area = load_json(args.work_area_json)
    roi = work_area["roi"]

    features, vis, mask, contour = detect_mouse_in_roi(
        image=image,
        roi=roi,
        real_length_mm=args.real_length_mm,
        real_width_mm=args.real_width_mm
    )

    pose = {
        "source_image": os.path.basename(args.image),
        "work_area_json": os.path.basename(args.work_area_json),

        "center_x": features["center_x"],
        "center_y": features["center_y"],

        "major_axis_x": features["major_axis_x"],
        "major_axis_y": features["major_axis_y"],
        "minor_axis_x": features["minor_axis_x"],
        "minor_axis_y": features["minor_axis_y"],

        "mouse_length_px": features["mouse_length_px"],
        "mouse_width_px": features["mouse_width_px"],

        "real_length_mm": features["real_length_mm"],
        "real_width_mm": features["real_width_mm"],
        "mm_per_pixel_length": features["mm_per_pixel_length"],
        "mm_per_pixel_width": features["mm_per_pixel_width"],

        "roi": features["roi"],

        # 新增：参考轮廓
        "reference_contour": contour_to_list(contour)
    }

    base = os.path.splitext(os.path.basename(args.image))[0]

    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_vis.jpg")
    mask_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_mask.png")
    pose_path = os.path.join(args.output_dir, args.output_name)

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(mask_path, mask)
    save_json(pose_path, pose)

    print("=== Locked Mouse Pose (Work Area + Contour) ===")
    print(json.dumps(pose, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(vis_path)
    print(mask_path)
    print(pose_path)


if __name__ == "__main__":
    main()