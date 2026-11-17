from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from ..core.geometry import (
    bbox_center,
    bbox_coverage,
    describe_relation,
    point_in_bbox,
    polygon_iou,
)


REGION_LABELS = {
    "image": "IMAGE",
    "chart": "CHART",
    "table": "TABLE",
    "header_image": "HEADER_IMAGE",
    "footer_image": "FOOTER_IMAGE",
}


def group_texts_into_lines(
    texts: Sequence[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """按视觉行聚类文本项，返回从上到下、行内从左到右的分组。"""
    if not texts:
        return []

    items: List[Dict[str, Any]] = []
    for index, text in enumerate(texts):
        x1, y1, x2, y2 = map(float, text["bbox"])
        items.append(
            {
                "index": index,
                "text": text,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "height": max(1.0, y2 - y1),
            }
        )

    items.sort(key=lambda item: (item["cy"], item["x1"]))
    lines: List[Dict[str, Any]] = []
    for item in items:
        matched = None
        for line in lines:
            tolerance = 0.6 * min(item["height"], line["height"])
            overlap = min(item["y2"], line["y2"]) - max(item["y1"], line["y1"])
            if abs(item["cy"] - line["cy"]) <= tolerance or overlap > 0.4 * min(
                item["height"], line["height"]
            ):
                matched = line
                break
        if matched is None:
            lines.append(
                {
                    "cy": item["cy"],
                    "height": item["height"],
                    "y1": item["y1"],
                    "y2": item["y2"],
                    "items": [item],
                }
            )
        else:
            matched["items"].append(item)
            matched["cy"] = sum(entry["cy"] for entry in matched["items"]) / len(
                matched["items"]
            )
            matched["height"] = max(entry["height"] for entry in matched["items"])
            matched["y1"] = min(entry["y1"] for entry in matched["items"])
            matched["y2"] = max(entry["y2"] for entry in matched["items"])

    lines.sort(key=lambda line: line["y1"])
    grouped: List[List[Dict[str, Any]]] = []
    for line in lines:
        line["items"].sort(key=lambda item: item["x1"])
        grouped.append([item["text"] for item in line["items"]])
    return grouped


def sort_texts_reading_order(texts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """摊平视觉行并给每个文本项标注所属行号，供排版文本换行使用。"""
    ordered: List[Dict[str, Any]] = []
    for line_index, group in enumerate(group_texts_into_lines(texts)):
        for text in group:
            ordered.append({**text, "line": line_index})
    return ordered


def _needs_space(left: str, right: str) -> bool:
    if not left or not right or left.isspace() or right.isspace():
        return False
    return left.isascii() and right.isascii() and (left.isalnum() or right.isalnum())


def _join_line_text(texts: Sequence[Dict[str, Any]]) -> List[str]:
    """把同一视觉行的文本拼成一行，并保留文本内部已有的换行符。"""
    joined = ""
    for text in texts:
        part = str(text["text"]).strip()
        if not part:
            continue
        if joined and _needs_space(joined[-1], part[0]):
            joined += " "
        joined += part
    return [line.strip() for line in joined.split("\n") if line.strip()]


class MergeService:
    def build_regions(self, detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"r{i + 1}",
                "type": item["type"],
                "bbox": [float(v) for v in item["bbox"]],
                "score": float(item["score"]),
            }
            for i, item in enumerate(detections)
        ]

    def build_texts(
        self,
        detections: Sequence[Dict[str, Any]],
        regions: Sequence[Dict[str, Any]],
        association_mode: str = "hybrid",
        text_in_region: str = "annotate",
    ) -> List[Dict[str, Any]]:
        texts: List[Dict[str, Any]] = []
        for i, item in enumerate(detections):
            polygon = [[float(x), float(y)] for x, y in item["polygon"]]
            bbox = [float(v) for v in item["bbox"]]
            region_ref = self._match_region(bbox, polygon, regions, association_mode)

            if text_in_region == "exclude" and region_ref and region_ref["relation"] == "inside":
                continue

            text_item: Dict[str, Any] = {
                "id": f"t{i + 1}",
                "text": item["text"],
                "score": float(item["score"]),
                "bbox": bbox,
                "polygon": polygon,
            }
            if text_in_region == "annotate" and region_ref is not None:
                text_item["region"] = region_ref
            texts.append(text_item)

        return sort_texts_reading_order(texts)

    def _match_region(
        self,
        bbox: Sequence[float],
        polygon: Sequence[Sequence[float]],
        regions: Sequence[Dict[str, Any]],
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        if not regions:
            return None
        center = bbox_center(bbox)
        candidates: List[tuple[float, Dict[str, Any], bool]] = []
        for region in regions:
            rbbox = region["bbox"]
            inside = point_in_bbox(center, rbbox)
            iou = polygon_iou(polygon, rbbox)
            coverage = bbox_coverage(bbox, rbbox)
            if mode == "center":
                score = 1.0 if inside else 0.0
            elif mode == "iou":
                score = max(iou, coverage)
            else:
                score = 1.0 if inside else max(iou, coverage) * 0.9
            if score >= 0.15:
                candidates.append((score, region, inside))

        if not candidates:
            return None
        score, region, inside = max(candidates, key=lambda item: item[0])
        return {
            "id": region["id"],
            "type": region["type"],
            "bbox": list(region["bbox"]),
            "relation": describe_relation(center, region["bbox"], inside),
        }

    def build_markdown(
        self,
        regions: Sequence[Dict[str, Any]],
        texts: Sequence[Dict[str, Any]],
        include_texts: bool = True,
    ) -> str:
        region_ids = {region["id"] for region in regions}
        texts_by_region: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        standalone_texts: List[Dict[str, Any]] = []

        for text in texts:
            ref = text.get("region") or {}
            if (
                include_texts
                and ref.get("relation") == "inside"
                and ref.get("id") in region_ids
            ):
                texts_by_region[ref["id"]].append(text)
            else:
                standalone_texts.append(text)

        blocks: List[Dict[str, Any]] = []
        for region in regions:
            label = REGION_LABELS.get(region["type"], region["type"].upper())
            bbox = ", ".join(f"{value:.0f}" for value in region["bbox"])
            lines = [f"[{label} {region['id']}] bbox=[{bbox}]"]

            if include_texts:
                for group in self._group_by_line(texts_by_region.get(region["id"], [])):
                    lines.extend(f"  {line}" for line in _join_line_text(group))

            blocks.append(
                {
                    "kind": "region",
                    "y": region["bbox"][1],
                    "x": region["bbox"][0],
                    "content": "\n".join(lines),
                }
            )

        if include_texts:
            for group in self._group_by_line(standalone_texts):
                blocks.append(
                    {
                        "kind": "text",
                        "y": min(text["bbox"][1] for text in group),
                        "x": min(text["bbox"][0] for text in group),
                        "content": "\n".join(_join_line_text(group)),
                    }
                )

        blocks.sort(key=lambda item: (item["y"], item["x"]))
        parts: List[str] = []
        previous_kind: Optional[str] = None
        for block in blocks:
            if parts:
                separator = (
                    "\n\n"
                    if block["kind"] == "region" or previous_kind == "region"
                    else "\n"
                )
                parts.append(separator)
            parts.append(block["content"])
            previous_kind = block["kind"]
        return "".join(parts)

    @staticmethod
    def _group_by_line(texts: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not texts:
            return []
        if all(isinstance(text.get("line"), int) for text in texts):
            grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for text in texts:
                grouped[text["line"]].append(text)
            return [grouped[key] for key in sorted(grouped)]
        return group_texts_into_lines(texts)

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
