from __future__ import annotations

from pathlib import Path

from docx import Document


class DOCXLoader:
    """
    Loads Microsoft Word (.docx) files.
    """

    def load(
        self,
        path: str | Path,
    ) -> str:

        document = Document(path)

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                paragraphs.append(text)

        return "\n".join(paragraphs)