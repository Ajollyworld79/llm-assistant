"""Extract text from uploaded files. Failures raise instead of producing placeholder text."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import urllib.parse
from collections.abc import Callable
from pathlib import PurePosixPath

import pdfplumber
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".md", ".csv", ".pdf", ".docx", ".url")
_PDF_MIN_CHARS = 200


class ParseError(ValueError):
    """The file could not be read (corrupt or malformed)."""


class UnsupportedFileType(ParseError):
    """The file extension is not supported."""


class EmptyDocument(ParseError):
    """The file parsed but contained no extractable text."""


def file_type(filename: str) -> str:
    suffix = PurePosixPath(filename).suffix
    return suffix[1:].upper() if suffix else "TXT"


def _decode(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _parse_text(_: str, content: bytes) -> str:
    return _decode(content)


def _parse_csv(filename: str, content: bytes) -> str:
    lines = _decode(content).splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Sheet:"):
            start = i + 1
        elif ";" in line or "," in line:
            start = i
            break
    body = "\n".join(lines[start:])
    delimiter = ";" if ";" in body else ","
    try:
        rows = list(csv.reader(io.StringIO(body), delimiter=delimiter))
    except csv.Error as exc:
        raise ParseError("malformed CSV") from exc
    if not rows:
        return ""
    header = rows[0]
    out = [f"**Data from {filename}**", ""]
    for row in rows[1:]:
        pairs = [f"{header[i]}: {val}" for i, val in enumerate(row) if i < len(header) and val.strip()]
        if pairs:
            out.append(" | ".join(pairs))
    return "\n".join(out) if len(out) > 2 else ""


def _pdf_links(reader: PdfReader) -> list[str]:
    links: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            annotations = page.get("/Annots") or []
            for annotation in annotations:
                obj = annotation.get_object()
                action = obj.get("/A") if obj.get("/Subtype") == "/Link" else None
                if action is not None and action.get("/S") == "/URI" and "/URI" in action:
                    links.append(f"Link (page {number}): {action['/URI']}")
        except (AttributeError, TypeError, KeyError, ValueError):
            continue
    return links


def _parse_pdf(_: str, content: bytes) -> str:
    links: list[str] = []
    pypdf_text = ""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                logger.debug("pypdf failed on a page", exc_info=True)
        pypdf_text = "\n".join(p for p in pages if p.strip())
        links = _pdf_links(reader)
    except Exception as exc:
        logger.info("pypdf could not open the PDF: %s", type(exc).__name__)

    text = pypdf_text
    if len(pypdf_text) < _PDF_MIN_CHARS:
        # Designed layouts sometimes defeat pypdf; pdfplumber may recover more.
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                plumber = "\n".join(t for t in (p.extract_text() or "" for p in pdf.pages) if t.strip())
            if len(plumber) > len(text):
                text = plumber
        except Exception as exc:
            logger.info("pdfplumber could not open the PDF: %s", type(exc).__name__)
            if not text:
                raise ParseError("unreadable PDF") from exc

    if text.strip() and links:
        text += "\n\n**Links from PDF document:**\n" + "\n".join(f"- {link}" for link in links)
    return text


_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _parse_docx(_: str, content: bytes) -> str:
    try:
        doc = Document(io.BytesIO(content))
    except Exception as exc:
        raise ParseError("unreadable DOCX") from exc

    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

    targets = {rel.rId: rel.target_ref for rel in doc.part.rels.values() if "hyperlink" in rel.reltype}
    links = []
    for paragraph in doc.paragraphs:
        for hyperlink in paragraph._element.findall(f".//{_W_NS}hyperlink"):
            url = targets.get(hyperlink.get(f"{_R_NS}id") or "")
            label = "".join(node.text for node in hyperlink.iter() if node.text).strip()
            if url and label:
                links.append(f"- {label}: {url}")
    if links:
        parts.append("\n**Links from document:**\n" + "\n".join(links))

    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text.strip() for cell in row.cells if cell.text.strip())
    return "\n".join(parts)


def _parse_url(filename: str, content: bytes) -> str:
    url = next(
        (line.strip()[4:] for line in _decode(content).splitlines() if line.strip().upper().startswith("URL=")), ""
    )
    if not url:
        raise EmptyDocument("shortcut has no URL")
    name = filename.rsplit(".", 1)[0].replace("_", " ")
    domain = urllib.parse.urlparse(url).netloc
    out = f"**{name}**\n\nType: Internet Link / Shortcut\nDescription: This is a link to {name}.\n"
    if "powerbi.com" in domain:
        out += "This is a Power BI report link.\n"
    elif "sharepoint.com" in domain:
        out += "This is a SharePoint link.\n"
    return out + f"\nLink: [{name}]({url})\n\nKeywords: {name}, link, shortcut, URL\n"


_PARSERS: dict[str, Callable[[str, bytes], str]] = {
    ".txt": _parse_text,
    ".md": _parse_text,
    ".csv": _parse_csv,
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".url": _parse_url,
}


def parse_content(filename: str, content: bytes) -> str:
    """Return the text of ``content``; raise ParseError subclasses for unsupported/empty/corrupt files."""
    suffix = PurePosixPath(filename.lower()).suffix
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise UnsupportedFileType(f"unsupported file type: {suffix or 'none'}")
    text = parser(filename, content)
    if not text.strip():
        raise EmptyDocument("no extractable text")
    return text


async def parse_content_async(filename: str, content: bytes) -> str:
    return await asyncio.to_thread(parse_content, filename, content)
