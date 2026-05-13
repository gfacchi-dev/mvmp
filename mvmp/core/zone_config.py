"""
Zone-based camera configuration for deterministic multi-view landmark detection.

Each of the 478 MediaPipe landmarks is assigned to exactly one of 5 zones.
After auto-align the face faces +Z:
  yaw  > 0  →  camera at +X  →  sees face's −X = subject's LEFT
  yaw  < 0  →  camera at −X  →  sees face's +X = subject's RIGHT
  pitch > 0  →  camera at +Y  →  looks downward, sees forehead
  pitch < 0  →  camera at −Y  →  looks upward, sees chin
"""
from __future__ import annotations
import numpy as np

N_LM = 478

# ── Zone IDs ─────────────────────────────────────────────────────────────
FRONT         = 0
LEFT_QUARTER  = 1   # subject's left  (camera yaw = +35°)
RIGHT_QUARTER = 2   # subject's right (camera yaw = −35°)
UPPER         = 3   # forehead / top  (camera pitch = +15°)
LOWER         = 4   # chin / jaw      (camera pitch = −15°)

# (yaw_deg, pitch_deg) for each zone camera
ZONE_CAMERAS: dict[int, tuple[float, float]] = {
    FRONT:         (  0.0,   0.0),
    LEFT_QUARTER:  (+35.0,   0.0),
    RIGHT_QUARTER: (-35.0,   0.0),
    UPPER:         (  0.0, +15.0),
    LOWER:         (  0.0, -15.0),
}

ZONE_NAMES = {
    FRONT:         "front",
    LEFT_QUARTER:  "left_quarter",
    RIGHT_QUARTER: "right_quarter",
    UPPER:         "upper",
    LOWER:         "lower",
}


# ── Landmark → Zone mapping (dict[int, int]) ─────────────────────────────

_ZONE_MAP: dict[int, int] = {}

# front (223 landmarks)
_ZONE_MAP.update({i: FRONT for i in [
    0, 1, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 57, 59, 61, 62, 64, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 98,
    100, 101, 102, 106, 114, 115, 122, 126, 128, 129, 131, 134, 142, 146, 164, 165, 166, 167, 168, 174,
    178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 191, 193, 194, 195, 196, 197, 198, 200, 201, 202,
    203, 204, 205, 206, 207, 209, 212, 216, 217, 218, 219, 220, 235, 236, 237, 239, 240, 244, 245, 248,
    266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 287, 289, 291, 292, 294,
    302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
    322, 324, 325, 327, 329, 330, 331, 335, 343, 344, 351, 355, 357, 358, 360, 363, 371, 375, 376, 391,
    392, 393, 399, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 415, 416, 417, 418, 419, 420,
    421, 422, 423, 424, 425, 426, 427, 429, 432, 433, 434, 436, 437, 438, 439, 440, 455, 456, 457, 459,
    460, 464, 465,
]})

# left_quarter (94 landmarks)
_ZONE_MAP.update({i: LEFT_QUARTER for i in [
    249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 284, 286, 288, 298,
    301, 323, 332, 339, 340, 341, 342, 345, 346, 347, 348, 349, 350, 352, 353, 356, 359, 361, 362, 364,
    365, 366, 367, 368, 369, 372, 373, 374, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
    390, 394, 395, 397, 398, 400, 401, 413, 414, 430, 431, 435, 441, 442, 443, 444, 445, 446, 447, 448,
    449, 450, 451, 452, 453, 454, 463, 466, 467, 473, 474, 475, 476, 477,
]})

# right_quarter (99 landmarks)
_ZONE_MAP.update({i: RIGHT_QUARTER for i in [
    7, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 54, 56, 58, 68,
    71, 93, 103, 110, 111, 112, 113, 116, 117, 118, 119, 120, 121, 123, 124, 127, 130, 132, 133, 135,
    136, 137, 138, 139, 140, 143, 144, 145, 147, 149, 150, 153, 154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 169, 170, 172, 173, 176, 177, 187, 189, 190, 192, 210, 211, 213, 214, 215, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 243, 246, 247, 468, 469, 470, 471, 472,
]})

# upper (32 landmarks)
_ZONE_MAP.update({i: UPPER for i in [
    10, 46, 52, 53, 55, 63, 65, 66, 67, 69, 70, 104, 105, 107, 108, 109, 151, 276, 282, 283,
    285, 293, 295, 296, 297, 299, 300, 333, 334, 336, 337, 338,
]})

# lower (30 landmarks)
_ZONE_MAP.update({i: LOWER for i in [
    2, 19, 20, 60, 94, 97, 99, 125, 141, 148, 152, 171, 175, 199, 208, 238, 241, 242, 250, 290,
    326, 328, 354, 370, 377, 396, 428, 458, 461, 462,
]})


# ── Derived data structures ───────────────────────────────────────────────

def _build_zones() -> tuple[np.ndarray, dict[int, list[int]]]:
    zones = np.full(N_LM, FRONT, dtype=np.int32)
    for lm_idx, z in _ZONE_MAP.items():
        zones[lm_idx] = z

    zone_lms: dict[int, list[int]] = {
        z: [i for i in range(N_LM) if zones[i] == z]
        for z in sorted(ZONE_CAMERAS)
    }
    return zones, zone_lms


LANDMARK_ZONES, ZONE_LANDMARKS = _build_zones()
