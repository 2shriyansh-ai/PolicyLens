from __future__ import annotations

from pathlib import Path

from .models import Page


class DocumentReadError(RuntimeError):
    pass


def _extract_page_text(page) -> str:
    text = page.get_text("text").strip()
    structured_tables: list[str] = []
    try:
        for table in page.find_tables().tables:
            rows = table.extract()
            rendered_rows = [
                " | ".join((cell or "").replace("\n", " ").strip() for cell in row)
                for row in rows
            ]
            if rendered_rows:
                structured_tables.append("TABLE\n" + "\n".join(rendered_rows))
    except Exception:
        # Some PDFs have no usable layout information. Plain text still remains
        # available and table cases are surfaced in failure analysis.
        pass
    if structured_tables:
        text = text + "\n\n" + "\n\n".join(structured_tables)
    return text


def extract_pages(path: str | Path) -> list[Page]:
    """Extract pages while preserving the source name and one-based page number."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = source.read_text(encoding="utf-8")
        return [Page(document=source.name, number=1, text=text)]
    if suffix != ".pdf":
        raise DocumentReadError(f"Unsupported document type: {source.suffix}")

    try:
        import pymupdf  # type: ignore
    except ImportError as exc:
        raise DocumentReadError(
            "PDF support requires PyMuPDF. Install the project dependencies first."
        ) from exc

    try:
        with pymupdf.open(source) as document:
            pages = []
            for index, page in enumerate(document):
                text = _extract_page_text(page)
                if text:
                    pages.append(Page(document=source.name, number=index + 1, text=text))
            return pages
    except Exception as exc:
        raise DocumentReadError(f"Could not read {source.name}: {exc}") from exc


def extract_uploaded_pdf(name: str, data: bytes) -> list[Page]:
    try:
        import pymupdf  # type: ignore
    except ImportError as exc:
        raise DocumentReadError("PDF support requires PyMuPDF.") from exc

    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            pages = []
            for index, page in enumerate(document):
                text = _extract_page_text(page)
                if text:
                    pages.append(Page(document=name, number=index + 1, text=text))
            return pages
    except Exception as exc:
        raise DocumentReadError(f"Could not read {name}: {exc}") from exc
