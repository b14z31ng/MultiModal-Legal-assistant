from __future__ import annotations

import json
from pathlib import Path

import torch


class RiskEngine:
    """
    Converts model outputs into
    human-readable risk analysis.
    """

    def __init__(

        self,

        clause_threshold=0.50,

        vision_mapping="data/processed/vision/class_mapping.json",

        clause_mapping="data/processed/clause_mapping.json",

    ):

        self.clause_threshold = clause_threshold

        ####################################################
        # Load Document Classes
        ####################################################

        vision_mapping = Path(vision_mapping)

        with open(
            vision_mapping,
            "r",
        ) as f:

            mapping = json.load(f)

        ####################################################
        # Convert
        #
        # {"invoice": 0}
        #
        # ->
        #
        # ["invoice", ...]
        ####################################################

        self.class_names = [None] * len(mapping)

        for name, index in mapping.items():

            self.class_names[index] = name

        ####################################################
        # Load Clause Mapping
        ####################################################

        ####################################################
        # Load Clause Mapping
        ####################################################

        clause_mapping = Path(clause_mapping)

        with open(
            clause_mapping,
            "r",
        ) as f:

            clause_json = json.load(f)

        ####################################################
        # Already stored as:
        #
        # {
        #   "0": "...",
        #   "1": "...",
        # }
        #
        ####################################################

        self.clause_names = [

            clause_json[str(i)]

            for i in range(len(clause_json))

        ]
        ####################################################
        # Convert
        ####################################################

    ####################################################
    # Risk Level
    ####################################################

    @staticmethod
    def risk_level(score):

        if score < 25:
            return "LOW"

        if score < 50:
            return "MEDIUM"

        if score < 75:
            return "HIGH"

        return "CRITICAL"

    ####################################################
    # Recommendation
    ####################################################

    @staticmethod
    def recommendation(level):

        table = {

            "LOW":
                "No major legal issues detected.",

            "MEDIUM":
                "Review highlighted clauses.",

            "HIGH":
                "Legal review recommended before signing.",

            "CRITICAL":
                "Do not sign without legal counsel.",

        }

        return table[level]

    ####################################################
    # Main
    ####################################################

    def analyze(

        self,

        prediction,

    ):

        ####################################################
        # Document
        ####################################################

        document_logits = prediction["document_logits"]

        document_prob = torch.softmax(

            document_logits,

            dim=1,

        )

        document_index = torch.argmax(

            document_prob,

            dim=1,

        ).item()

        ####################################################
        # Clauses
        ####################################################

        clause_prob = torch.sigmoid(

            prediction["clause_logits"]

        )[0]

        detected = []

        for i, probability in enumerate(clause_prob):

            if probability >= self.clause_threshold:

                detected.append(

                    {

                        "clause": self.clause_names[i],

                        "probability": round(
                            float(probability),
                            4,
                        ),

                    }

                )

        ####################################################
        # Risk
        ####################################################

        risk_probability = prediction[
            "risk_score"
        ].item()

        risk_score = round(

            risk_probability * 100,

            2,

        )

        level = self.risk_level(

            risk_score,

        )

        ####################################################
        # Output
        ####################################################

        return {

            "risk_score": risk_score,

            "risk_level": level,

            "document_type": self.class_names[document_index],

            "confidence": round(

                float(

                    document_prob.max()

                ) * 100,

                2,

            ),

            "high_risk_clauses": detected,

            "recommendation": self.recommendation(

                level,

            ),

        }