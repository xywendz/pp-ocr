from app.core.geometry import bbox_center, bbox_iou, describe_relation, point_in_polygon, to_bbox


def test_to_bbox_and_center():
    polygon = [[10, 20], [30, 20], [30, 40], [10, 40]]
    assert to_bbox(polygon) == [10, 20, 30, 40]
    assert bbox_center([10, 20, 30, 40]) == (20.0, 30.0)


def test_bbox_iou():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_point_in_polygon():
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon([5, 5], polygon) is True
    assert point_in_polygon([15, 5], polygon) is False


def test_describe_relation():
    assert describe_relation([5, 5], [0, 0, 10, 10], True) == "inside"
    assert describe_relation([5, 15], [0, 0, 10, 10], False) == "below"
    assert describe_relation([15, 5], [0, 0, 10, 10], False) == "right"
