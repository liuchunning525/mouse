import cv2
import json
import os
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # green


def draw_hand_landmarks_on_bgr_image(image, detection_result):
    annotated = image.copy()
    h, w = image.shape[:2]

    if detection_result is None or not detection_result.hand_landmarks:
        return annotated

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):

        # 画点
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

        # 画骨架（简单连线）
        connections = [
            (0,1),(1,2),(2,3),(3,4),      # thumb
            (0,5),(5,6),(6,7),(7,8),      # index
            (0,9),(9,10),(10,11),(11,12), # middle
            (0,13),(13,14),(14,15),(15,16), # ring
            (0,17),(17,18),(18,19),(19,20)  # pinky
        ]

        for c in connections:
            p1 = hand_landmarks[c[0]]
            p2 = hand_landmarks[c[1]]

            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)

            cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 标注左右手
        label = "Unknown"
        if idx < len(detection_result.handedness):
            if len(detection_result.handedness[idx]) > 0:
                label = detection_result.handedness[idx][0].category_name

        # 找一个点写文字
        base = hand_landmarks[0]
        text_x = int(base.x * w)
        text_y = int(base.y * h) - 10

        cv2.putText(
            annotated,
            label,
            (text_x, max(20, text_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return annotated


def normalized_to_pixel(landmark, image_width, image_height):
    """
    Convert normalized landmark to pixel coordinates.
    """
    x_px = int(round(landmark.x * image_width))
    y_px = int(round(landmark.y * image_height))
    return x_px, y_px


def compute_hand_bbox(hand_landmarks, image_width, image_height):
    xs = []
    ys = []

    for lm in hand_landmarks:
        x_px, y_px = normalized_to_pixel(lm, image_width, image_height)
        xs.append(x_px)
        ys.append(y_px)

    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(image_width - 1, max(xs))
    y_max = min(image_height - 1, max(ys))

    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "width": int(x_max - x_min),
        "height": int(y_max - y_min)
    }


def extract_hand_data(detection_result, image_width, image_height):
    """
    Convert MediaPipe detection result into JSON-friendly dict.
    """
    result_dict = {
        "num_hands": 0,
        "hands": []
    }

    if detection_result is None or not detection_result.hand_landmarks:
        return result_dict

    result_dict["num_hands"] = len(detection_result.hand_landmarks)

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_info = {}

        # handedness
        handedness_label = "Unknown"
        handedness_score = None
        if idx < len(detection_result.handedness) and len(detection_result.handedness[idx]) > 0:
            handedness_label = detection_result.handedness[idx][0].category_name
            handedness_score = float(detection_result.handedness[idx][0].score)

        hand_info["hand_index"] = idx
        hand_info["handedness"] = handedness_label
        hand_info["handedness_score"] = handedness_score

        # bbox
        hand_info["bbox"] = compute_hand_bbox(hand_landmarks, image_width, image_height)

        # landmarks
        landmarks_list = []
        for lm_idx, lm in enumerate(hand_landmarks):
            x_px, y_px = normalized_to_pixel(lm, image_width, image_height)
            landmarks_list.append({
                "id": lm_idx,
                "x_norm": float(lm.x),
                "y_norm": float(lm.y),
                "z_norm": float(lm.z),
                "x_px": int(x_px),
                "y_px": int(y_px)
            })

        hand_info["landmarks"] = landmarks_list

        # Useful key points
        wrist = landmarks_list[0]
        index_tip = landmarks_list[8]
        middle_tip = landmarks_list[12]
        ring_tip = landmarks_list[16]
        pinky_tip = landmarks_list[20]
        thumb_tip = landmarks_list[4]

        hand_info["keypoints_summary"] = {
            "wrist": wrist,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
            "middle_tip": middle_tip,
            "ring_tip": ring_tip,
            "pinky_tip": pinky_tip
        }

        result_dict["hands"].append(hand_info)

    return result_dict


def create_detector(model_path, num_hands=2, min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5, min_tracking_confidence=0.5):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


def save_outputs(image_path, output_dir, hand_data, vis_image):
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base}_hand_landmarks.json")
    vis_path = os.path.join(output_dir, f"{base}_hand_vis.jpg")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hand_data, f, indent=2, ensure_ascii=False)

    cv2.imwrite(vis_path, vis_image)

    return json_path, vis_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="hand_landmarker.task", help="Path to hand_landmarker.task")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--num_hands", type=int, default=2, help="Maximum number of hands to detect")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            f"Please download hand_landmarker.task and put it in the working directory "
            f"or specify --model path/to/hand_landmarker.task"
        )

    bgr_image = cv2.imread(args.image)
    if bgr_image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    detector = create_detector(args.model, num_hands=args.num_hands)
    detection_result = detector.detect(mp_image)

    image_height, image_width = bgr_image.shape[:2]
    hand_data = extract_hand_data(detection_result, image_width, image_height)
    vis_image = draw_hand_landmarks_on_bgr_image(bgr_image, detection_result)

    json_path, vis_path = save_outputs(args.image, args.output_dir, hand_data, vis_image)

    print("=== Hand Landmark Extraction Result ===")
    print(f"num_hands: {hand_data['num_hands']}")

    for hand in hand_data["hands"]:
        print(f"\nHand {hand['hand_index']}:")
        print(f"  handedness: {hand['handedness']}")
        print(f"  handedness_score: {hand['handedness_score']}")
        print(f"  bbox: {hand['bbox']}")

    print("\nSaved:")
    print(json_path)
    print(vis_path)


if __name__ == "__main__":
    main()