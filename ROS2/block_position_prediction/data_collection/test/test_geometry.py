from block_position_prediction.data_collection.geometry import (
    BLOCK_SIDE_TAXEL,
    SheetConfig,
    fixed_square_footprint,
    label_from_paper_position,
    normalize_yaw_mod90,
    yaw_mod90_vector,
)


def test_default_sensor_is_centered_on_landscape_a4():
    config = SheetConfig()

    assert config.sensor_origin_mm == (114.5, 85.0)
    assert config.sensor_rect_mm == (114.5, 85.0, 182.5, 125.0)
    assert round(config.sensor.right_margin_mm, 6) == 4.0
    assert round(config.sensor.bottom_margin_mm, 6) == 8.0


def test_sensor_mm_to_array_coordinates():
    sensor = SheetConfig().sensor

    assert sensor.sensor_mm_to_array(4.0, 4.0) == (0.0, 0.0)
    assert sensor.sensor_mm_to_taxel_center(4.0, 4.0) == (0.0, 0.0)
    assert sensor.taxel_center_to_sensor_mm(15.0, 7.0) == (64.0, 32.0)
    assert sensor.taxel_center_to_normalized(15.0, 7.0) == (1.0, 1.0)
    cm = sensor.taxel_center_to_cm_from_taxel0(6.0, 2.0)
    assert abs(cm[0] - 2.4) < 1e-8
    assert abs(cm[1] - 0.8) < 1e-8
    assert sensor.sensor_mm_to_array(64.0, 32.0) == (15.0, 7.0)
    assert sensor.array_contains(15.0, 7.0)
    assert not sensor.array_contains(16.0, 7.0)
    assert sensor.contains_taxel_sensor(-1.0, -1.0)
    assert sensor.contains_taxel_sensor(16.0, 9.0)
    assert not sensor.contains_taxel_sensor(16.1, 9.0)


def test_fixed_square_footprint_uses_taxel_center_frame():
    sensor = SheetConfig().sensor
    footprint = fixed_square_footprint((7.5, 3.5), yaw_rad=0.0, side_taxel=BLOCK_SIDE_TAXEL)

    assert footprint[0] == (4.5, 0.5)
    assert footprint[2] == (10.5, 6.5)
    assert sensor.footprint_fully_inside_sensor(footprint)
    assert not sensor.footprint_fully_inside_sensor(fixed_square_footprint((0.0, 0.0), side_taxel=BLOCK_SIDE_TAXEL))


def test_yaw_mod90_vector_represents_square_symmetry():
    yaw = normalize_yaw_mod90(3.141592653589793 / 2.0 + 0.25)
    vector = yaw_mod90_vector(yaw)

    assert abs(yaw - 0.25) < 1e-8
    assert abs(vector[0] * vector[0] + vector[1] * vector[1] - 1.0) < 1e-8


def test_label_from_paper_position_reports_inside_flags():
    config = SheetConfig()

    label = label_from_paper_position(config, 118.5, 89.0)
    assert label.sensor_mm == (4.0, 4.0)
    assert label.array_col_row == (0.0, 0.0)
    assert label.inside_sensor
    assert label.inside_array

    outside = label_from_paper_position(config, 100.0, 100.0)
    assert not outside.inside_sensor
    assert not outside.inside_array
