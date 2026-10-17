import pytest

from hackeval.ingestion import IngestionError, extract_zip, parse_document
from tests.helpers import make_pdf, make_pptx, make_zip


def test_extract_filters_and_blocks_zip_slip(tmp_path, settings):
    z = make_zip(tmp_path / "a.zip", {
        "team1/deck.pptx": make_pptx(), "doc.pdf": make_pdf(),
        "__MACOSX/._doc.pdf": b"x", "notes.txt": b"x", "../evil.pdf": make_pdf(),
    })
    res = extract_zip(z, tmp_path / "out", settings)
    assert sorted(f.original_name for f in res.files) == ["doc.pdf", "team1/deck.pptx"]
    assert any("unsafe" in s for s in res.skipped)
    assert any("unsupported" in s for s in res.skipped)
    assert not (tmp_path / "evil.pdf").exists()


def test_empty_and_invalid_zip(tmp_path, settings):
    with pytest.raises(IngestionError):
        extract_zip(make_zip(tmp_path / "e.zip", {"a.txt": b"x"}), tmp_path / "o", settings)
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(IngestionError):
        extract_zip(bad, tmp_path / "o2", settings)


def test_parse_pdf_pptx(tmp_path):
    p = tmp_path / "a.pdf"; p.write_bytes(make_pdf())
    q = tmp_path / "a.pptx"; q.write_bytes(make_pptx())
    assert "crop disease" in parse_document(p, "a.pdf", 10000).text
    d = parse_document(q, "a.pptx", 10000)
    assert "[Slide 1]" in d.text and "Demo Project" in d.text


def test_truncation_and_empty(tmp_path):
    p = tmp_path / "a.pdf"; p.write_bytes(make_pdf())
    assert parse_document(p, "a.pdf", 100).truncated
    e = tmp_path / "e.pdf"; e.write_bytes(make_pdf("  "))
    with pytest.raises(IngestionError):
        parse_document(e, "e.pdf", 1000)
