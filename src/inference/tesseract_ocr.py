from __future__ import annotations

from PIL import Image
import pytesseract

from src.inference.base_ocr import BaseOCR


class TesseractOCR(BaseOCR):
    """
    Tesseract OCR backend.
    """

    def __init__(
        self,
        language="eng",
    ):

        self.language = language

    def extract(
        self,
        image: Image.Image,
    ) -> str:

        text = pytesseract.image_to_string(

            image,

            lang=self.language,

            config="--oem 3 --psm 6",

        )

        return text.strip()