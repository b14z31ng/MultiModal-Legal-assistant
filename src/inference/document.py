from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass(slots=True)
class ProcessedDocument:
    """
    Standard document object shared by the
    entire inference pipeline.

    Created by:
        DocumentProcessor

    Consumed by:
        Predictor
        FastAPI
        Streamlit
    """

    ####################################################
    # Core Data
    ####################################################

    image: Image.Image

    text: str

    ####################################################
    # Metadata
    ####################################################

    source: str

    file_type: str

    page_count: int = 1

    ####################################################
    # Optional Metadata
    ####################################################

    ocr_engine: Optional[str] = None

    language: Optional[str] = None

    filename: Optional[str] = None