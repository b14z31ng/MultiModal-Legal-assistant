from __future__ import annotations

from PIL import Image

import pytesseract


class OCR:

    """
    OCR wrapper.

    Uses Tesseract for text extraction.
    """

    def __init__(

        self,

        language="eng",

    ):

        self.language = language

    ########################################################

    def extract(

        self,

        image: Image.Image,

    ) -> str:

        text = pytesseract.image_to_string(

            image,

            lang=self.language,

        )

        return text.strip()
    