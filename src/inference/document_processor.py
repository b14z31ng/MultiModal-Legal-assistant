from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.inference.document import ProcessedDocument
from src.inference.ocr_factory import OCRFactory
from src.inference.pdf_loader import PDFLoader
from src.inference.docx_loader import DOCXLoader

class DocumentProcessor:
    """
    Handles supported document types.

    Supported
    ---------
    - PDF
    - PNG
    - JPG
    - JPEG
    - TIFF
    - BMP
    - DOCX (coming later)
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    }

    PDF_EXTENSIONS = {
        ".pdf",
    }

    DOCX_EXTENSIONS = {
        ".docx",
    }

    def __init__(
        self,
    ):

        self.ocr = OCRFactory.create(
            backend="rapidocr",
        )

        self.pdf_loader = PDFLoader()
        self.docx_loader = DOCXLoader()

    ########################################################
    # File Type
    ########################################################

    def get_file_type(
        self,
        file_path,
    ):

        suffix = Path(file_path).suffix.lower()

        if suffix in self.IMAGE_EXTENSIONS:
            return "image"

        if suffix in self.PDF_EXTENSIONS:
            return "pdf"

        if suffix in self.DOCX_EXTENSIONS:
            return "docx"

        raise ValueError(
            f"Unsupported file type: {suffix}"
        )

    ########################################################
    # Image
    ########################################################

    def process_image(
        self,
        image_path,
    ):

        image = Image.open(
            image_path,
        ).convert(
            "RGB",
        )

        extracted_text = self.ocr.extract(
            image,
        )

        return ProcessedDocument(
            image=image,
            text=extracted_text,
            source=str(image_path),
            file_type="image",
            page_count=1,
            ocr_engine=type(self.ocr).__name__,
            filename=Path(image_path).name,
        )

    ########################################################
    # PDF
    ########################################################

    def process_pdf(
        self,
        pdf_path,
    ):

        pages = self.pdf_loader.load(
            pdf_path,
        )

        texts = []

        for page in pages:

            text = self.ocr.extract(
                page,
            )

            texts.append(
                text,
            )

        return ProcessedDocument(
            image=pages[0],
            text="\n\n".join(texts),
            source=str(pdf_path),
            file_type="pdf",
            page_count=len(pages),
            ocr_engine=type(self.ocr).__name__,
            filename=Path(pdf_path).name,
        )

    ########################################################
    # Main
    ########################################################
    def process_docx(
        self,
        docx_path,
    ):

        text = self.docx_loader.load(
            docx_path,
        )

        image = Image.new(
            "RGB",
            (384, 384),
            color="white",
        )

        return ProcessedDocument(
            image=image,
            text=text,
            source=str(docx_path),
            file_type="docx",
            page_count=1,
            ocr_engine=None,
            filename=Path(docx_path).name,
        )

    def process(
        self,
        file_path,
    ):

        file_type = self.get_file_type(
            file_path,
        )

        if file_type == "image":
            return self.process_image(
                file_path,
            )

        if file_type == "pdf":
            return self.process_pdf(
                file_path,
            )

        if file_type == "docx":

            return self.process_docx(
                file_path,
            )

        raise ValueError(
            f"Unsupported file type: {file_type}"
        )