import os
import json
import argparse
import cv2

from detect_mouse import extract_mouse_features


def save_mouse_pose(output_path, pose):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pose, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Mouse-only reference image")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--real_length_mm", type=float, default=None, help="Real mouse length in mm")
    parser.add_argument("--real_width_mm", type=float, default=None, help="Real mouse width in mm")
    parser.add_argument("--output_name", default="mouse_pose_ref.json", help="Output pose filename")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    features, vis, mask, _ = extract_mouse_features(
        image=image,
        real_length_mm=args.real_length_mm,
        real_width_mm=args.real_width_mm
    )

    pose = {
        "source_image": os.path.basename(args.image),
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
        "mm_per_pixel_width": features["mm_per_pixel_width"]
    }

    base = os.path.splitext(os.path.basename(args.image))[0]
    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_vis.jpg")
    mask_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_mask.png")
    pose_path = os.path.join(args.output_dir, args.output_name)

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(mask_path, mask)
    save_mouse_pose(pose_path, pose)

    print("=== Locked Mouse Pose ===")
    for k, v in pose.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(vis_path)
    print(mask_path)
    print(pose_path)


if __name__ == "__main__":
    main()