import json
import argparse


# 关键点ID
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


def get_landmark(hand, lm_id):
    for lm in hand["landmarks_mouse_frame"]:
        if lm["id"] == lm_id:
            return lm
    return None


def compute_features(hand):
    wrist = get_landmark(hand, WRIST)
    index_tip = get_landmark(hand, INDEX_TIP)
    middle_tip = get_landmark(hand, MIDDLE_TIP)
    ring_tip = get_landmark(hand, RING_TIP)
    pinky_tip = get_landmark(hand, PINKY_TIP)

    if None in [wrist, index_tip, middle_tip, ring_tip, pinky_tip]:
        return None

    # ===== 核心特征 =====

    # 1. 手掌位置（越大越靠后）
    palm_position = wrist["a_norm"]

    # 2. 手指平均前伸程度（越小越靠前）
    finger_forward = (
        index_tip["a_norm"] +
        middle_tip["a_norm"] +
        ring_tip["a_norm"] +
        pinky_tip["a_norm"]
    ) / 4.0

    # 3. 手指“弯曲程度”（用 wrist 到 tip 距离近似）
    def dist(lm):
        dx = lm["a_norm"] - wrist["a_norm"]
        dy = lm["b_norm"] - wrist["b_norm"]
        return (dx**2 + dy**2) ** 0.5

    finger_spread = (
        dist(index_tip) +
        dist(middle_tip) +
        dist(ring_tip) +
        dist(pinky_tip)
    ) / 4.0

    return {
        "palm_position": palm_position,
        "finger_forward": finger_forward,
        "finger_spread": finger_spread
    }


def classify_grip(features):
    if features is None:
        return "unknown"

    palm = features["palm_position"]
    forward = features["finger_forward"]
    spread = features["finger_spread"]

    # ===== 简单规则分类 =====

    # fingertip：手掌很靠后 + 手指很靠前
    if palm > 0.2 and forward < -0.1:
        return "fingertip"

    # claw：手掌靠后 + 手指不太前 + 弯曲明显
    if palm > 0.1 and forward > -0.05 and spread < 0.25:
        return "claw"

    # palm：手掌不太靠后 + 手指比较平
    return "palm"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="hand_mouse_frame.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["num_hands"] == 0:
        print("No hands detected")
        return

    for hand in data["hands"]:
        features = compute_features(hand)
        grip = classify_grip(features)

        print("\n=== Grip Classification ===")
        print(f"Hand: {hand['handedness']}")
        print(f"Palm position (a_norm): {features['palm_position']:.3f}")
        print(f"Finger forward: {features['finger_forward']:.3f}")
        print(f"Finger spread: {features['finger_spread']:.3f}")
        print(f"Predicted grip: {grip}")


if __name__ == "__main__":
    main()