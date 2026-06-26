from __future__ import annotations

from src.inference.tesseract_ocr import TesseractOCR
from src.inference.rapidocr_ocr import RapidOCREngine


class OCRFactory:

    @staticmethod
    def create(

        backend="rapidocr",

    ):

        backend = backend.lower()

        if backend == "rapidocr":

            return RapidOCREngine()

        if backend == "tesseract":

            return TesseractOCR()

        raise ValueError(
            f"Unknown OCR backend: {backend}"
        )