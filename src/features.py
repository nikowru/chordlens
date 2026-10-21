"""
Feature engineering: turn 21 MediaPipe hand landmarks into one feature vector.

This module is the "data preparation" step of CRISP-DM and the reason the
project is classical machine learning rather than deep learning: we never show
the classifier raw pixels. Instead we hand it a short, geometric description of
the hand shape. The module depends only on NumPy, so it can be reasoned about
on paper and reused unchanged if the MediaPipe API is ever swapped.

THE THREE FEATURE GROUPS (explain these in this order to a panel)
-----------------------------------------------------------------
1. Normalised landmark coordinates (60 values)
   Raw MediaPipe coordinates describe *where the hand is in the picture*, not
   *what shape it makes*. Two problems follow: moving your hand to the left of
   the frame changes every number, and standing closer to the camera scales
   every number. The classifier would then have to learn each chord shape
   separately for every position and distance, which it cannot do from a small
   dataset.
   We remove both nuisances geometrically:
     - translation: subtract the wrist (landmark 0) from all 21 points, so the
       wrist becomes the origin. Position in the frame no longer matters.
     - scale: divide by the wrist-to-middle-finger-MCP distance (landmark 0 to
       landmark 9), which is a stable proxy for hand size in the image.
       Distance from the camera, and the player's own hand size, no longer
       matter.
   The wrist is dropped afterwards because it is exactly (0, 0, 0) by
   construction, and a constant column carries no information. That leaves
   20 landmarks x 3 coordinates = 60 features.

2. Pairwise fingertip-to-fingertip distances (10 values)
   The five fingertips are landmarks 4, 8, 12, 16 and 20, which gives
   C(5, 2) = 10 pairs. These distances say directly how spread out or bunched
   together the fingers are, which is what separates, say, open C (fingers
   fanned across three frets) from a barre (fingertips stacked in a line).
   The same information is implicitly present in group 1, but only as a
   difference the model would have to discover for itself; stating it
   explicitly makes the classes easier to separate with a linear or RBF
   boundary.

3. Finger joint angles (15 values)
   For each of the five fingers we take its chain of landmarks starting at the
   wrist and measure the angle at its three interior joints (for the index
   finger: the angles at MCP, PIP and DIP). Each angle is the angle between the
   two bone vectors meeting at that joint.
   Angles are invariant to translation, scale *and* rotation, so they stay
   meaningful even when the guitar neck is tilted - which it always is, and
   differently in every public dataset photo. They capture how far each finger
   is curled, which is how a guitarist describes a chord shape in the first
   place.

Total: 60 + 10 + 15 = 85 features, in the fixed order given by FEATURE_NAMES.

A NOTE ON ASPECT RATIO
----------------------
MediaPipe returns x and y normalised to image width and height separately, so
on a 1280x720 frame one unit of x spans 1280 pixels but one unit of y spans
only 720. Distances and angles computed from those numbers come out stretched.
Passing aspect_ratio = width / height rescales x back into the same units as y,
making the geometry isotropic and features from webcam frames comparable with
features from public images of a different shape.
"""

from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices (see the MediaPipe Hands landmark diagram)
# ---------------------------------------------------------------------------

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_FINGER_MCP = 9  # the knuckle used as the hand-size reference point

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# Landmark chains from the wrist out to each fingertip.
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}

# Anatomical names of the three interior joints of each chain. Used only to make
# the feature names readable in the CSV and in figures.
JOINT_NAMES = {
    "thumb": ("cmc", "mcp", "ip"),
    "index": ("mcp", "pip", "dip"),
    "middle": ("mcp", "pip", "dip"),
    "ring": ("mcp", "pip", "dip"),
    "pinky": ("mcp", "pip", "dip"),
}

FINGERTIPS = {name: chain[-1] for name, chain in FINGER_CHAINS.items()}

# Guards against dividing by a hand size of zero on a degenerate detection.
_EPSILON = 1e-8


def _build_feature_names():
    """Build the fixed list of feature column names, in output order."""
    names = []
    # Group 1: normalised coordinates, wrist excluded.
    for index in range(NUM_LANDMARKS):
        if index == WRIST:
            continue
        names += [f"lm{index}_x", f"lm{index}_y", f"lm{index}_z"]
    # Group 2: fingertip-to-fingertip distances.
    for first, second in combinations(FINGER_NAMES, 2):
        names.append(f"tipdist_{first}_{second}")
    # Group 3: joint angles.
    for finger in FINGER_NAMES:
        for joint in JOINT_NAMES[finger]:
            names.append(f"angle_{finger}_{joint}")
    return names


FEATURE_NAMES = _build_feature_names()
NUM_FEATURES = len(FEATURE_NAMES)  # 85


def landmarks_to_array(hand_landmarks):
    """
    Convert a MediaPipe NormalizedLandmarkList into a plain (21, 3) array.

    This is the only function that knows the shape of the MediaPipe result
    object, which keeps the rest of the module pure NumPy. If the project is
    ever moved to the newer MediaPipe Tasks API, only this function changes.
    """
    return np.array(
        [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
        dtype=np.float64,
    )


def normalise_landmarks(landmarks, aspect_ratio=1.0):
    """
    Translate landmarks to a wrist-centred origin and scale by hand size.

    Returns a (21, 3) array in which landmark 0 is exactly the origin and the
    distance from the wrist to the middle-finger MCP is exactly 1. This is the
    translation- and scale-invariant representation described in group 1 above.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"expected {NUM_LANDMARKS} landmarks of (x, y, z), got {points.shape}"
        )

    points = points.copy()
    # Undo MediaPipe's per-axis normalisation so x, y and z share a unit.
    points[:, 0] *= float(aspect_ratio)

    # Translation invariance: the wrist becomes the origin.
    points -= points[WRIST]

    # Scale invariance: one "hand size" becomes one unit.
    hand_size = np.linalg.norm(points[MIDDLE_FINGER_MCP])
    return points / max(hand_size, _EPSILON)


def _joint_angle(point_a, point_b, point_c):
    """
    Angle in radians at point_b, between the bones b->a and b->c.

    Returned in [0, pi]: a straight finger gives an angle near pi, a fully
    curled joint gives a small angle.
    """
    bone_one = point_a - point_b
    bone_two = point_c - point_b
    norms = np.linalg.norm(bone_one) * np.linalg.norm(bone_two)
    if norms < _EPSILON:
        return 0.0
    # Clipping guards against floating point values a hair outside [-1, 1].
    cosine = np.clip(np.dot(bone_one, bone_two) / norms, -1.0, 1.0)
    return float(np.arccos(cosine))


def extract_features(landmarks, aspect_ratio=1.0):
    """
    Turn 21 hand landmarks into the 85-element feature vector.

    Pure function: same landmarks in, same vector out, no state and no I/O.

    Parameters
    ----------
    landmarks : array-like, shape (21, 3)
        The MediaPipe hand landmarks as (x, y, z). Use landmarks_to_array() to
        obtain this from a MediaPipe result.
    aspect_ratio : float
        Source image width / height, used to make the geometry isotropic.
        See the module docstring.

    Returns
    -------
    numpy.ndarray, shape (85,), dtype float64
        Values in the order described by FEATURE_NAMES.
    """
    points = normalise_landmarks(landmarks, aspect_ratio=aspect_ratio)

    # Group 1: normalised coordinates, flattened, wrist dropped (it is all zeros).
    coordinates = np.delete(points, WRIST, axis=0).ravel()

    # Group 2: distances between every pair of fingertips.
    tip_distances = [
        np.linalg.norm(points[FINGERTIPS[first]] - points[FINGERTIPS[second]])
        for first, second in combinations(FINGER_NAMES, 2)
    ]

    # Group 3: three joint angles per finger.
    angles = []
    for finger in FINGER_NAMES:
        chain = FINGER_CHAINS[finger]
        for position in range(1, 4):  # the three interior joints of the chain
            angles.append(
                _joint_angle(
                    points[chain[position - 1]],
                    points[chain[position]],
                    points[chain[position + 1]],
                )
            )

    features = np.concatenate([coordinates, np.array(tip_distances), np.array(angles)])
    assert features.shape == (NUM_FEATURES,), "feature vector length changed"
    return features
