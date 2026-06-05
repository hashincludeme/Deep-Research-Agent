from pathlib import Path
from typing import Optional

import config


def read_text_file(path: str, max_chars: Optional[int] = None) -> Optional[str]:
    """Reads a plain text or markdown file."""
    max_chars = max_chars or config.CONTENT_MAX_CHARS
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:max_chars]
    except OSError as e:
        print(f"  [reader] Could not read {path}: {e}")
        return None


def read_pdf(path: str, max_chars: Optional[int] = None) -> Optional[str]:
    """
    Reads a PDF and returns its text content.
    Requires pypdf (listed in requirements.txt).
    """
    max_chars = max_chars or config.CONTENT_MAX_CHARS
    try:
        import pypdf
    except ImportError:
        print("  [reader] pypdf not installed. Run: pip install pypdf")
        return None

    try:
        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)[:max_chars]
    except Exception as e:
        print(f"  [reader] PDF read error {path}: {e}")
        return None


def read_document(path: str, max_chars: Optional[int] = None) -> Optional[str]:
    """Dispatches to the correct reader based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path, max_chars)
    elif ext in (".txt", ".md", ".rst", ".csv"):
        return read_text_file(path, max_chars)
    else:
        print(f"  [reader] Unsupported file type: {ext}")
        return None
