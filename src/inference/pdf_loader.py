from __future__ import annotations

from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image


class PDFLoader:

    def __init__(
        self,
        dpi: int = 300,
    ):

        self.dpi = dpi

    def load(
        self,
        pdf_path: str | Path,
    ) -> List[Image.Image]:

        return convert_from_path(
            pdf_path,
            dpi=self.dpi,
        )