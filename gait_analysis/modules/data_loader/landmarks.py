"""Model names and the gait-relevant canonical landmark set."""

MODELS = ["POSE", "SIMPLE_HOLISTIC", "HOLISTIC"]

# Lower-body + shoulders: the landmarks Module 2 needs. Present in all 3 models.
GAIT_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
