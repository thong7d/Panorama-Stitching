#!/usr/bin/env python3
"""Validate V0 homographies on real images against an independent OpenCV model.

Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
         Dao Quoc Tuan (23120392)
External sources: OpenCV SIFT/BFMatcher/findHomography API documentation.
No external implementation code was copied. See THIRD_PARTY_NOTICES.md.

Real scenes have no ground-truth homography, so this script does not label the
OpenCV estimate as truth. It reports (1) projected-grid disagreement with an
independent SIFT + RANSAC estimate and (2) V0 consistency after normalizing
coordinates across 720p, 1080p and 4K.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


DATASETS = {
    "site1": "site1_indoor",
    "site2": "site2_outdoor",
    "site3": "site3_custom",
    "challenging": "challenging",
}
RESOLUTIONS = ("small", "medium", "large")


def _project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((points, np.ones(len(points))))
    mapped_h = (H @ points_h.T).T
    valid = np.abs(mapped_h[:, 2]) > 1e-12
    mapped = np.full((len(points), 2), np.nan, dtype=np.float64)
    mapped[valid] = mapped_h[valid, :2] / mapped_h[valid, 2:3]
    return mapped


def _pixel_grid(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0.1 * width, 0.9 * width, 9)
    ys = np.linspace(0.1 * height, 0.9 * height, 5)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float64)


def _grid_disagreement_pct(
    first: np.ndarray, second: np.ndarray, width: int, height: int
) -> float:
    points = _pixel_grid(width, height)
    p1 = _project(first, points)
    p2 = _project(second, points)
    valid = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    if valid.sum() < 4:
        return math.inf
    rmse = np.sqrt(np.mean(np.sum((p1[valid] - p2[valid]) ** 2, axis=1)))
    return float(100.0 * rmse / math.hypot(width, height))


def _normalized_homography(H: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.diag([width, height, 1.0])
    normalized = np.linalg.inv(scale) @ H @ scale
    return normalized / normalized[2, 2]


def _reference_homography(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray | None, int, int, str]:
    cv2.setRNGSeed(42)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=4000)
        norm = cv2.NORM_L2
        method = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=5000, fastThreshold=20)
        norm = cv2.NORM_HAMMING
        method = "ORB"

    key_a, desc_a = detector.detectAndCompute(left, None)
    key_b, desc_b = detector.detectAndCompute(right, None)
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return None, 0, 0, method

    pairs = cv2.BFMatcher(norm).knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None, len(good), 0, method

    points_a = np.float32([key_a[m.queryIdx].pt for m in good])
    points_b = np.float32([key_b[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(
        points_a,
        points_b,
        cv2.RANSAC,
        3.0,
        maxIters=10000,
        confidence=0.999,
    )
    if H is None or mask is None or not np.isfinite(H).all():
        return None, len(good), 0, method
    H = H / H[2, 2]
    return H, len(good), int(mask.sum()), method


def validate(
    dataset_root: Path, metrics_root: Path
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    normalized_by_site: dict[str, list[tuple[str, np.ndarray]]] = {}

    for output_name, dataset_name in DATASETS.items():
        for resolution in RESOLUTIONS:
            metrics_path = metrics_root / output_name / f"metrics_{resolution}.json"
            left_path = dataset_root / dataset_name / f"left_{resolution}.jpg"
            right_path = dataset_root / dataset_name / f"right_{resolution}.jpg"
            if not metrics_path.is_file() or not left_path.is_file() or not right_path.is_file():
                continue

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            H_v0 = np.asarray(metrics["homography_a_to_b"], dtype=np.float64).reshape(3, 3)
            H_v0 /= H_v0[2, 2]
            left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                continue
            height, width = left.shape
            H_ref, matches, inliers, method = _reference_homography(left, right)
            disagreement = (
                _grid_disagreement_pct(H_v0, H_ref, width, height)
                if H_ref is not None
                else math.inf
            )
            record = {
                "site": output_name,
                "resolution": resolution,
                "reference_method": f"OpenCV {method} + findHomography",
                "reference_matches": matches,
                "reference_inliers": inliers,
                "v0_reference_grid_disagreement_pct_diagonal": disagreement,
                "reference_available": H_ref is not None,
            }
            records.append(record)
            normalized_by_site.setdefault(output_name, []).append(
                (resolution, _normalized_homography(H_v0, width, height))
            )

    consistency: list[dict] = []
    for site, items in normalized_by_site.items():
        if len(items) < 2:
            continue
        disagreements = []
        unit_w, unit_h = 1000, 1000
        unit_scale = np.diag([unit_w, unit_h, 1.0])
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                first = unit_scale @ items[i][1] @ np.linalg.inv(unit_scale)
                second = unit_scale @ items[j][1] @ np.linalg.inv(unit_scale)
                disagreements.append(
                    _grid_disagreement_pct(first, second, unit_w, unit_h)
                )
        consistency.append(
            {
                "site": site,
                "resolutions": [name for name, _ in items],
                "max_cross_resolution_grid_disagreement_pct_diagonal": max(disagreements),
            }
        )
    return records, consistency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--metrics-root", type=Path, default=Path("output"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    records, consistency = validate(args.dataset_root, args.metrics_root)
    if not records:
        print("[Error] No matching real-image metrics were found.")
        return 1

    print("site\tresolution\tref matches/inliers\tV0-vs-ref grid disagreement (% diagonal)")
    for row in records:
        disagreement = row["v0_reference_grid_disagreement_pct_diagonal"]
        value = f"{disagreement:.3f}" if math.isfinite(disagreement) else "N/A"
        print(
            f"{row['site']}\t{row['resolution']}\t"
            f"{row['reference_matches']}/{row['reference_inliers']}\t{value}"
        )
    print("\nCross-resolution V0 consistency (normalized coordinates):")
    for row in consistency:
        print(
            f"{row['site']}: "
            f"{row['max_cross_resolution_grid_disagreement_pct_diagonal']:.3f}% diagonal"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"comparisons": records, "consistency": consistency}, indent=2),
            encoding="utf-8",
        )
        print(f"[OK] Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
