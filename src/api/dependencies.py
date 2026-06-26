from __future__ import annotations

from functools import lru_cache

from src.inference.predictor import Predictor
from src.inference.document_processor import DocumentProcessor


@lru_cache(maxsize=1)
def get_predictor():

    return Predictor()


@lru_cache(maxsize=1)
def get_processor():

    return DocumentProcessor()