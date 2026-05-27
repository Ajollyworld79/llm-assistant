import logging
import io
import csv
import os
import asyncio
from pypdf import PdfReader

# Setup logging
try:
    from backend.app.module.Functions_module import setup_logger
    logger = setup_logger()
except ImportError:
    try:
        from module.Functions_module import setup_logger
        logger = setup_logger()
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger = logging.getLogger(__name__)

class ParserService:
    """Service responsible for extracting text and links from uploaded files.

    This encapsulates the previous free-function `parse_content` and provides a
    single, testable interface for file parsing.
    """

    def __init__(self):
        self.logger = logger

    def parse_content(self, filename: str, content: bytes) -> str:

        lower = filename.lower()

        try:
            # Plain text and markdown
            if lower.endswith('.txt') or lower.endswith('.md'):
                return content.decode('utf-8', errors='replace')

            elif lower.endswith('.csv'):
                s = content.decode('utf-8', errors='replace')
                # Skip leading "Sheet:" preamble lines that some exports add
                lines = s.splitlines()
                csv_start = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('Sheet:'):
                        csv_start = i + 1
                    elif ';' in line or ',' in line:
                        csv_start = i
                        break
                csv_text = '\n'.join(lines[csv_start:])
                sep = ';' if ';' in csv_text else ','
                reader = csv.reader(io.StringIO(csv_text), delimiter=sep)
                rows = list(reader)
                if not rows:
                    return ''
                header = rows[0]
                out = [f"**Data from {filename}**", ""]
                for row in rows[1:]:
                    pairs = [f"{header[i]}: {val}" for i, val in enumerate(row) if i < len(header) and val.strip()]
                    if pairs:
                        out.append(' | '.join(pairs))
                return '\n'.join(out)

            # PDF parsing with link extraction.
            # Strategy: run BOTH pypdf and pdfplumber, pick whichever extracted
            # the most text. Modern CVs (Canva/Notion layouts) often defeat
            # pypdf silently — it returns partial text without raising, so the
            # old fallback path was never triggered.
            if lower.endswith('.pdf'):
                try:
                    pypdf_parts: list = []
                    plumber_parts: list = []
                    links = []

                    # --- pypdf attempt (also extracts links) ---
                    try:
                        reader = PdfReader(io.BytesIO(content))
                        for i, page in enumerate(reader.pages, start=1):
                            try:
                                page_text = page.extract_text() or ''
                                if page_text.strip():
                                    pypdf_parts.append(page_text)
                            except Exception:
                                pass

                            # Extract annotations/links if present
                            try:
                                annots_obj = page.get('/Annots')  # type: ignore[reportGeneralTypeIssues]
                                if annots_obj:
                                    try:
                                        for a in annots_obj:
                                            try:
                                                obj = a.get_object()
                                                if obj.get('/Subtype') == '/Link' and '/A' in obj:
                                                    action = obj['/A']
                                                    if action.get('/S') == '/URI' and '/URI' in action:
                                                        uri = action['/URI']
                                                        links.append(f"Link (side {i}): {uri}")
                                            except Exception:
                                                continue
                                    except TypeError:
                                        try:
                                            a = annots_obj
                                            obj = a.get_object()
                                            if obj.get('/Subtype') == '/Link' and '/A' in obj:
                                                action = obj['/A']
                                                if action.get('/S') == '/URI' and '/URI' in action:
                                                    uri = action['/URI']
                                                    links.append(f"Link (side {i}): {uri}")
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # --- pdfplumber attempt (often better on complex layouts) ---
                    try:
                        import pdfplumber
                        with pdfplumber.open(io.BytesIO(content)) as pdf:
                            for page in pdf.pages:
                                txt = page.extract_text() or ''
                                if txt.strip():
                                    plumber_parts.append(txt)
                    except Exception:
                        pass

                    pypdf_text = '\n'.join(p for p in pypdf_parts if p and p.strip())
                    plumber_text = '\n'.join(p for p in plumber_parts if p and p.strip())

                    # Pick whichever extracted more characters
                    if len(plumber_text) > len(pypdf_text):
                        combined = plumber_text
                        logger.info("PDF parse: chose pdfplumber (%d chars vs pypdf %d)", len(plumber_text), len(pypdf_text))
                    else:
                        combined = pypdf_text
                        logger.info("PDF parse: chose pypdf (%d chars vs pdfplumber %d)", len(pypdf_text), len(plumber_text))

                    if not combined.strip():
                        return f"[Simulated PDF parsing for {filename} - pdf parsing libraries missing or failed]"

                    if links:
                        combined += '\n\n**Links from PDF document:**\n'
                        for l in links:
                            combined += f"- {l}\n"

                    if combined.strip() == '':
                        return f"[Simulated PDF parsing for {filename} - no text extracted]"

                    return combined
                except Exception as e:
                    self.logger.exception('PDF parsing error: %s', e)
                    return f"[Simulated PDF parsing for {filename} - error]"

            # DOCX parsing with link extraction
            elif lower.endswith('.docx'):
                try:
                    from docx import Document

                    doc = Document(io.BytesIO(content))
                    parts = []

                    # Extract paragraphs
                    for para in doc.paragraphs:
                        if para.text and para.text.strip():
                            parts.append(para.text.strip())

                    # Extract hyperlinks using rels
                    try:
                        all_hyperlinks = {}
                        for rel in doc.part.rels.values():
                            if 'hyperlink' in rel.reltype:
                                all_hyperlinks[rel.rId] = rel.target_ref

                        all_document_links = []
                        for paragraph in doc.paragraphs:
                            try:
                                for hyperlink in paragraph._element.findall(
                                    './/{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hyperlink'
                                ):
                                    rel_id = hyperlink.get(
                                        '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
                                    )
                                    if rel_id in all_hyperlinks:
                                        url = all_hyperlinks[rel_id]
                                        link_text = ''.join(node.text for node in hyperlink.iter() if node.text).strip()
                                        if url and link_text:
                                            all_document_links.append((link_text, url))
                            except Exception:
                                pass

                        if all_document_links:
                            links_section = '\n\n**Links from document:**\n' + '\n'.join([f"- {text}: {url}" for text, url in all_document_links])
                            parts.append(links_section)
                    except Exception:
                        # Ignore link extraction failures
                        pass

                    # Tables
                    try:
                        for table in doc.tables:
                            for row in table.rows:
                                for cell in row.cells:
                                    cell_text = cell.text.strip()
                                    if cell_text:
                                        parts.append(cell_text)
                    except Exception:
                        pass

                    return '\n'.join(parts)
                except Exception:
                    return f"[Simulated DOCX parsing for {filename} - python-docx missing or failed]"

            # PPTX slides
            elif lower.endswith('.pptx'):
                try:
                    from pptx import Presentation
                    prs = Presentation(io.BytesIO(content))
                    text = ''
                    for slide_num, slide in enumerate(prs.slides, 1):
                        text += f"\n--- Slide {slide_num} ---\n"
                        for shape in slide.shapes:
                            shape_text = getattr(shape, "text", None)
                            if shape_text:
                                text += shape_text + "\n"
                    return text
                except Exception:
                    return f"[Simulated PPTX parsing for {filename} - python-pptx missing or failed]"

            # URL shortcut files (.url)
            elif lower.endswith('.url'):
                try:
                    text = content.decode('utf-8', errors='replace')
                    url = None
                    for line in text.splitlines():
                        line = line.strip()
                        if line.upper().startswith('URL='):
                            url = line[4:]
                            break

                    if url:
                        import urllib.parse
                        parsed = urllib.parse.urlparse(str(url))
                        domain = str(parsed.netloc) or 'Unknown domain'
                        filename_without_ext = filename.rsplit('.', 1)[0].replace('_', ' ')

                        content_out = f"**{filename_without_ext}**\n\n"
                        content_out += f"Type: Internet Link / Shortcut\n"
                        content_out += f"Description: This is a link to {filename_without_ext}.\n"

                        if 'powerbi.com' in domain:
                            content_out += "This is a Power BI report link.\n"
                        elif 'sharepoint.com' in domain:
                            content_out += "This is a SharePoint link.\n"

                        content_out += f"\nLink: [{filename_without_ext}]({url})\n"
                        content_out += f"\nKeywords: {filename_without_ext}, link, shortcut, URL\n"

                        return content_out
                    else:
                        return f"Internet shortcut: {filename}"

                except Exception as e:
                    self.logger.exception('Error parsing URL file: %s', e)
                    return f"URL shortcut: {filename}"

            # Fallback
            else:
                return f"[Simulated parsing for {filename} - demo mode]\nContent not extracted."

        except Exception as e:
            self.logger.exception('Error parsing content: %s', e)
            return f"[Error parsing {filename}, using fallback text.]"

# Create a single global parser instance for convenience
parser_service = ParserService()

async def parse_content(filename: str, content: bytes) -> str:
    """Compatibility wrapper that delegates to the shared ParserService.
    Runs the blocking parser in a separate thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(parser_service.parse_content, filename, content)
