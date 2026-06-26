from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from PIL import Image


class BaseOCR(ABC):
    """
    Base interface for all OCR engines.
    """

    @abstractmethod
    def extract(
        self,
        image: Image.Image,
    ) -> str:
        """
        Extract text from image.
        """
        raise NotImplementedError