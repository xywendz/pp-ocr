from app.core.geometry import rotate_point_back, transform_bbox_back, transform_polygon_back


def test_rotate_point_back_90():
    assert rotate_point_back([10, 20], 90, 800, 500) == [780, 10]


def test_rotate_point_back_180():
    assert rotate_point_back([10, 20], 180, 800, 500) == [790, 480]


def test_rotate_point_back_270():
    assert rotate_point_back([10, 20], 270, 800, 500) == [20, 490]


def test_transform_polygon_back_90():
    polygon = [[0, 0], [100, 0], [100, 50], [0, 50]]
    restored = transform_polygon_back(polygon, 90, 800, 500)
    assert restored[0] == [800, 0]
    assert restored[2] == [750, 100]


def test_transform_bbox_back_90():
    bbox = [10, 20, 100, 80]
    assert transform_bbox_back(bbox, 90, 800, 500) == [720, 10, 780, 100]
