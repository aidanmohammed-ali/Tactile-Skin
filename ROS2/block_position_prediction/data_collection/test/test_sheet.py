from block_position_prediction.data_collection.sheet import generate_calibration_sheet


def test_generate_pdf_calibration_sheet(tmp_path):
    output = tmp_path / "sheet.pdf"
    metadata = tmp_path / "sheet.json"

    generate_calibration_sheet(output, dpi=72, metadata_path=metadata)

    assert output.read_bytes().startswith(b"%PDF")
    assert metadata.exists()

