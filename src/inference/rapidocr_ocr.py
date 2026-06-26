from __future__ import annotations

import numpy as np
from PIL import Image

from rapidocr_onnxruntime import RapidOCR

from src.inference.base_ocr import BaseOCR


class RapidOCREngine(BaseOCR):
    """
    RapidOCR backend.
    """

    def __init__(self):

        self.engine = RapidOCR()

    def extract(
        self,
        image: Image.Image,
    ) -> str:

        image = np.array(image)

        result, _ = self.engine(image)

        if result is None:
            return ""

        lines = []

        for item in result:

            # item = [box, text, confidence]
            if len(item) >= 2:
                lines.append(item[1])

        return "\n".join(lines).strip()