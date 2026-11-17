from app.services.merge_service import MergeService, sort_texts_reading_order


def sample_regions():
    return [
        {
            "id": "r1",
            "type": "image",
            "bbox": [100, 100, 300, 300],
            "score": 0.9,
        }
    ]


def sample_text():
    return [
        {
            "text": "图 1 系统架构",
            "score": 0.95,
            "bbox": [130, 180, 260, 210],
            "polygon": [[130, 180], [260, 180], [260, 210], [130, 210]],
        }
    ]


def test_text_is_annotated_with_region():
    service = MergeService()
    texts = service.build_texts(sample_text(), sample_regions())
    assert texts[0]["region"]["id"] == "r1"
    assert texts[0]["region"]["relation"] == "inside"


def test_text_in_region_exclude():
    service = MergeService()
    texts = service.build_texts(sample_text(), sample_regions(), text_in_region="exclude")
    assert texts == []


def test_markdown_contains_region_placeholder_and_bbox():
    service = MergeService()
    regions = service.build_regions(sample_regions())
    texts = service.build_texts(sample_text(), regions)
    markdown = service.build_markdown(regions, texts)
    assert "[IMAGE r1] bbox=[100, 100, 300, 300]" in markdown
    assert "图 1 系统架构" in markdown


def test_reading_order_groups_same_line_by_x():
    texts = [
        {"text": "丙", "bbox": [800, 42, 900, 112]},
        {"text": "甲", "bbox": [100, 34, 300, 120]},
        {"text": "乙", "bbox": [500, 40, 700, 115]},
        {"text": "下一行", "bbox": [100, 200, 300, 260]},
    ]
    ordered = sort_texts_reading_order(texts)
    assert [item["text"] for item in ordered] == ["甲", "乙", "丙", "下一行"]
    assert [item["line"] for item in ordered] == [0, 0, 0, 1]


def test_markdown_joins_same_visual_line_without_newline():
    service = MergeService()
    texts = [
        {"text": "甲", "bbox": [100, 34, 300, 120], "line": 0},
        {"text": "乙", "bbox": [500, 40, 700, 115], "line": 0},
        {"text": "下一行", "bbox": [100, 200, 300, 260], "line": 1},
    ]
    assert service.build_markdown([], texts) == "甲乙\n下一行"


def test_markdown_region_groups_same_visual_line():
    service = MergeService()
    regions = service.build_regions(sample_regions())
    texts = [
        {
            "text": "图",
            "bbox": [130, 180, 190, 210],
            "line": 0,
            "region": {"id": "r1", "relation": "inside"},
        },
        {
            "text": "1",
            "bbox": [200, 182, 260, 212],
            "line": 0,
            "region": {"id": "r1", "relation": "inside"},
        },
        {
            "text": "第二行",
            "bbox": [130, 240, 260, 270],
            "line": 1,
            "region": {"id": "r1", "relation": "inside"},
        },
    ]
    markdown = service.build_markdown(regions, texts)
    assert "  图1\n  第二行" in markdown


def test_markdown_keeps_internal_newline_and_ascii_spacing():
    service = MergeService()
    texts = [
        {"text": "第一行\n第二行", "bbox": [100, 34, 300, 120], "line": 0},
        {"text": "Hello", "bbox": [100, 200, 200, 260], "line": 1},
        {"text": "World", "bbox": [220, 200, 320, 260], "line": 1},
    ]
    assert service.build_markdown([], texts) == "第一行\n第二行\nHello World"
