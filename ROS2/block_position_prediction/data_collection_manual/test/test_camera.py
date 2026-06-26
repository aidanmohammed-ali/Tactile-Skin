from block_position_prediction.data_collection_manual.camera import _should_use_process_capture, parse_camera_source


def test_url_camera_sources_use_process_isolation():
    assert _should_use_process_capture("http://example.test/video")
    assert _should_use_process_capture("rtsp://example.test/stream")
    assert not _should_use_process_capture(0)
    assert not _should_use_process_capture("/dev/video0")


def test_parse_camera_source_keeps_urls_as_strings():
    assert parse_camera_source("0") == 0
    assert parse_camera_source("http://example.test/video") == "http://example.test/video"
