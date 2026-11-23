from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import re
import logging
from .models import Document

logger = logging.getLogger(__name__)

# Fallback imports
HAS_BS4 = False
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    logger.warning("beautifulsoup4 not installed. HTML parsing will be limited.")

HAS_PYPDF = False
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    logger.warning("pypdf not installed. PDF parsing will be unavailable.")

HAS_DOCX = False
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    logger.warning("python-docx not installed. DOCX parsing will be unavailable.")


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> Document:
        pass

    def clean_text(self, text: str) -> str:
        # Preserve single newlines but remove multiple ones
        text = re.sub(r'[ \t]+', ' ', text) # Clean horizontal whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text) # Normalize multiple newlines
        text = text.strip()
        return text

class TextLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cleaned_content = self.clean_text(content)
        metadata = {
            "source": str(file_path),
            "doc_type": "txt",
            "file_name": file_path.name
        }
        return Document.create(cleaned_content, metadata)

class PDFLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        if not HAS_PYPDF:
            raise ImportError("pypdf is required to load PDF files.")

        reader = PdfReader(file_path)
        content = ""
        for page in reader.pages:
            content += page.extract_text() + "\n"

        cleaned_content = self.clean_text(content)
        metadata = {
            "source": str(file_path),
            "doc_type": "pdf",
            "file_name": file_path.name,
            "page_count": len(reader.pages)
        }
        return Document.create(cleaned_content, metadata)

class DocxLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        if not HAS_DOCX:
            raise ImportError("python-docx is required to load DOCX files.")

        doc = DocxDocument(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
        cleaned_content = self.clean_text(content)
        metadata = {
            "source": str(file_path),
            "doc_type": "docx",
            "file_name": file_path.name
        }
        return Document.create(cleaned_content, metadata)

class MarkdownLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cleaned_content = self.clean_text(content)
        metadata = {
            "source": str(file_path),
            "doc_type": "md",
            "file_name": file_path.name
        }
        return Document.create(cleaned_content, metadata)

class HTMLLoader(BaseLoader):
    def load(self, file_path: Path) -> Document:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if HAS_BS4:
            soup = BeautifulSoup(content, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            content = soup.get_text()
        else:
            # Simple regex fallback if BS4 is missing
            content = re.sub(r'<[^>]+>', ' ', content)

        cleaned_content = self.clean_text(content)
        metadata = {
            "source": str(file_path),
            "doc_type": "html",
            "file_name": file_path.name
        }
        return Document.create(cleaned_content, metadata)

class LoaderFactory:
    _loaders = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".docx": DocxLoader,
        ".md": MarkdownLoader,
        ".html": HTMLLoader
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:
        extension = file_path.suffix.lower()
        loader_cls = cls._loaders.get(extension)
        if not loader_cls:
            raise ValueError(f"Unsupported file extension: {extension}")
        return loader_cls()
