import io

import pytest
from docx import Document

from backend.app.services.parser import (
    EmptyDocument,
    ParseError,
    UnsupportedFileType,
    file_type,
    parse_content,
    parse_content_async,
)


def make_pdf(text: str | None, link: str | None = None) -> bytes:
    """Build a tiny, valid one-page PDF with a correct xref table."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
    annots = "/Annots [5 0 R] " if link else ""
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R {annots}"
        "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
    ]
    if link:
        objs.append(f"<< /Type /Annot /Subtype /Link /Rect [0 0 100 100] /A << /S /URI /URI ({link}) >> >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Root 1 0 R /Size {len(objs) + 1} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table:
        t = doc.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_text_and_markdown():
    assert parse_content("a.txt", b"hello world") == "hello world"
    assert parse_content("README.MD", "# Titel æøå".encode()) == "# Titel æøå"


def test_csv_with_sheet_preamble_and_semicolons():
    raw = b"Sheet: Users\nname;role\nAnna;admin\nBo;\n"
    text = parse_content("users.csv", raw)
    assert "Data from users.csv" in text
    assert "name: Anna | role: admin" in text
    assert "Bo" in text and "Sheet:" not in text


def test_csv_with_only_header_is_empty():
    with pytest.raises(EmptyDocument):
        parse_content("h.csv", b"a,b\n")


def test_pdf_text_and_links():
    text = parse_content("doc.pdf", make_pdf("Reset your password quickly", link="https://example.org/help"))
    assert "Reset your password quickly" in text
    assert "https://example.org/help" in text


def test_pdf_without_text_is_empty_document():
    with pytest.raises(EmptyDocument):
        parse_content("scan.pdf", make_pdf(None))


def test_corrupt_pdf_is_rejected():
    with pytest.raises(ParseError):
        parse_content("bad.pdf", b"%PDF-1.4 this is not a pdf")


def test_docx_paragraphs_and_tables():
    text = parse_content("d.docx", make_docx(["First paragraph", "Second"], [["cell-a", "cell-b"]]))
    assert "First paragraph" in text and "cell-a" in text and "cell-b" in text


def test_corrupt_and_empty_docx():
    with pytest.raises(ParseError):
        parse_content("bad.docx", b"not a zip")
    with pytest.raises(EmptyDocument):
        parse_content("empty.docx", make_docx([]))


def test_url_shortcut():
    text = parse_content("Power_BI.url", b"[InternetShortcut]\nURL=https://app.powerbi.com/report/1\n")
    assert "Power BI" in text and "https://app.powerbi.com/report/1" in text
    with pytest.raises(EmptyDocument):
        parse_content("none.url", b"[InternetShortcut]\n")


@pytest.mark.parametrize("name", ["slides.pptx", "archive.zip", "noext", "image.png"])
def test_unsupported_types(name):
    with pytest.raises(UnsupportedFileType):
        parse_content(name, b"data")


def test_whitespace_only_is_empty():
    with pytest.raises(EmptyDocument):
        parse_content("blank.txt", b"  \n\t ")


async def test_async_wrapper():
    assert await parse_content_async("a.txt", b"x") == "x"


def test_file_type():
    assert file_type("a.PDF") == "PDF" and file_type("noext") == "TXT"
