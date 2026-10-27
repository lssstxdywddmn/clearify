"""Laplacian-variance blur screening; not a semantic visibility classifier."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def assess(gray, low=30.0, high=100.0, roi=None):
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
        raise ValueError("Thresholds must satisfy 0 <= low < high and be finite.")
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Input must be a two-dimensional uint8 grayscale image.")
    if roi is not None:
        x1, y1, x2, y2 = roi
        height, width = gray.shape
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("ROI must be inside the original image: x1 y1 x2 y2.")
        gray = gray[y1:y2, x1:x2]
    if min(gray.shape) < 3:
        raise ValueError("The evaluated image/ROI must be at least 3 by 3 pixels.")

    # Four-neighbour Laplacian; ignore the one-pixel boundary.
    response = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)[1:-1, 1:-1]
    score = float(response.var())
    if score >= high:
        label = "清晰"
    elif score >= low:
        label = "细节不清"
    else:
        label = "主体不可辨"

    reasons = []
    # This is an initial review heuristic, not a published visibility threshold.
    if float(gray.std()) < 5:
        reasons.append("低对比度或低纹理：低分可能并非模糊")
    if score < low:
        reasons.append("第三类仅表示严重模糊候选，尚未确认主体语义可辨性")
    return {
        "label": label,
        "laplacian_variance": score,
        "thresholds": {"low": low, "high": high},
        "threshold_status": "unvalidated_on_business_images",
        "scope": "roi" if roi is not None else "whole_image",
        "evaluated_size_wh": [gray.shape[1], gray.shape[0]],
        "semantic_visibility_assessed": False,
        "requires_review": bool(reasons),
        "review_reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--low", type=float, default=30.0)
    parser.add_argument("--high", type=float, default=100.0)
    args = parser.parse_args()
    try:
        encoded = np.fromfile(args.image, dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError("Image file is empty.")
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Image could not be decoded.")
        result = assess(gray, args.low, args.high, args.roi)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
