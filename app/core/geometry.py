from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Sequence[float]
BBox = Sequence[float]


def to_bbox(points: Iterable[Point]) -> List[float]:
    arr = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = float(point[0]), float(point[1])
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def point_in_bbox(point: Point, bbox: BBox) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, bbox)
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_coverage(inner: BBox, outer: BBox) -> float:
    ix1, iy1, ix2, iy2 = map(float, inner)
    ox1, oy1, ox2, oy2 = map(float, outer)
    x1, y1 = max(ix1, ox1), max(iy1, oy1)
    x2, y2 = min(ix2, ox2), min(iy2, oy2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if area <= 0:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / area


def polygon_iou(polygon: Sequence[Point], bbox: BBox) -> float:
    try:
        from shapely.geometry import Polygon, box as shapely_box
    except Exception:
        return bbox_iou(to_bbox(polygon), bbox)

    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) < 3:
        return bbox_iou(to_bbox(polygon), bbox)
    try:
        poly = Polygon(pts)
        rect = shapely_box(*map(float, bbox))
        if not poly.is_valid:
            poly = poly.buffer(0)
        union = poly.union(rect).area
        return poly.intersection(rect).area / union if union > 0 else 0.0
    except Exception:
        return bbox_iou(to_bbox(polygon), bbox)


def describe_relation(point: Point, bbox: BBox, inside: bool) -> str:
    if inside:
        return "inside"
    px, py = float(point[0]), float(point[1])
    x1, y1, x2, y2 = map(float, bbox)
    if x1 <= px <= x2:
        return "below" if py > y2 else "above"
    if y1 <= py <= y2:
        return "right" if px > x2 else "left"
    return "near"


def nms_bboxes(items: List[dict], iou_threshold: float = 0.9) -> List[dict]:
    ordered = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    keep: List[dict] = []
    for item in ordered:
        bbox = item.get("bbox")
        if not bbox:
            continue
        if any(bbox_iou(bbox, kept["bbox"]) > iou_threshold for kept in keep):
            continue
        keep.append(item)
    return keep


def clamp_bbox(bbox: BBox, width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = map(float, bbox)
    nx1, ny1 = max(0.0, min(x1, x2)), max(0.0, min(y1, y2))
    nx2, ny2 = min(float(width), max(x1, x2)), min(float(height), max(y1, y2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


def rotate_point_back(
    point: Point,
    detected_angle: int,
    original_width: int,
    original_height: int,
) -> List[float]:
    x, y = float(point[0]), float(point[1])
    if detected_angle == 90:
        return [original_width - y, x]
    if detected_angle == 180:
        return [original_width - x, original_height - y]
    if detected_angle == 270:
        return [y, original_height - x]
    return [x, y]


def transform_polygon_back(
    polygon: Sequence[Point],
    detected_angle: int,
    original_width: int,
    original_height: int,
) -> List[List[float]]:
    return [
        rotate_point_back(point, detected_angle, original_width, original_height)
        for point in polygon
    ]


def transform_bbox_back(
    bbox: BBox,
    detected_angle: int,
    original_width: int,
    original_height: int,
) -> List[float]:
    corners = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
    ]
    restored = transform_polygon_back(corners, detected_angle, original_width, original_height)
    result = to_bbox(restored)
    clamped = clamp_bbox(result, original_width, original_height)
    return clamped or result
