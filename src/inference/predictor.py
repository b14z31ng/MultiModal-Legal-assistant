from __future__ import annotations

import torch

from torchvision import transforms
from transformers import AutoTokenizer

from src.inference.model_loader import load_multimodal_model
from src.inference.risk_engine import RiskEngine
from src.inference.document import ProcessedDocument

class Predictor:
    """
    Complete Multimodal Inference Pipeline.

    Responsibilities
    ----------------
    1. Receive processed document dictionary
    2. Tokenize extracted text
    3. Preprocess image
    4. Run Multimodal Model
    5. Run Risk Engine
    6. Return human-readable result
    """

    def __init__(

        self,

        device=None,

        image_size=384,

        tokenizer_name="answerdotai/ModernBERT-base",

    ):

        ####################################################
        # Device
        ####################################################

        if device is None:

            device = (

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        self.device = device

        ####################################################
        # Multimodal Model
        ####################################################

        self.model = load_multimodal_model(

            device=device,

        )

        ####################################################
        # Tokenizer
        ####################################################

        self.tokenizer = AutoTokenizer.from_pretrained(

            tokenizer_name,

            use_fast=True,

        )

        ####################################################
        # Risk Engine
        ####################################################

        self.risk_engine = RiskEngine()

        ####################################################
        # Image Transform
        ####################################################

        self.transform = transforms.Compose(

            [

                transforms.Resize(

                    (

                        image_size,

                        image_size,

                    )

                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[

                        0.485,

                        0.456,

                        0.406,

                    ],

                    std=[

                        0.229,

                        0.224,

                        0.225,

                    ],

                ),

            ]

        )

    ########################################################
    # Text Preprocessing
    ########################################################

    def preprocess_text(

        self,

        text,

    ):

        encoding = self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=512,

            return_tensors="pt",

        )

        return (

            encoding["input_ids"],

            encoding["attention_mask"],

        )

    ########################################################
    # Prediction
    ########################################################

    @torch.no_grad()

    def predict(

    self,

    document: ProcessedDocument,

):

        ####################################################
        # Validate Input
        ####################################################

        if not isinstance(

            document,

            ProcessedDocument,

        ):

            raise TypeError(

                "Expected ProcessedDocument."

            )

        ####################################################
        # Image
        ####################################################

        image = self.transform(

            document.image

        ).unsqueeze(

            0

        ).to(

            self.device

        )

        ####################################################
        # Text
        ####################################################

        input_ids, attention_mask = self.preprocess_text(

            document.text

        )

        input_ids = input_ids.to(

            self.device

        )

        attention_mask = attention_mask.to(

            self.device

        )

        ####################################################
        # Model Forward
        ####################################################

        prediction = self.model(

            image=image,

            input_ids=input_ids,

            attention_mask=attention_mask,

        )

        ####################################################
        # Human Readable Output
        ####################################################

        result = self.risk_engine.analyze(

            prediction,

        )

        ####################################################
        # Metadata
        ####################################################

        result["source"] = document.source

        result["file_type"] = document.file_type

        result["page_count"] = document.page_count

        result["ocr_engine"] = document.ocr_engine

        result["filename"] = document.filename

        return result