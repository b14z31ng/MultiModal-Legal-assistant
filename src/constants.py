from typing import Dict


DEFAULT_IMAGE_SIZE = 224

DEFAULT_MAX_LENGTH = 256

DEFAULT_THRESHOLD = 0.5


HIGH_RISK_CLAUSES = {
    "Termination For Convenience",
    "Non-Compete",
    "Revenue/Profit Sharing",
    "Most Favored Nation",
    "Change Of Control",
    "Unlimited/All-You-Can-Eat License",
}


SUPPORTED_VISION_MODELS: Dict[str, int] = {
    "resnet18": 512,
    "resnet50": 2048,
}


SUPPORTED_TEXT_MODELS: Dict[str, int] = {
    "distilbert-base-uncased": 768,
    "bert-base-uncased": 768,
    "nlpaueb/legal-bert-base-uncased": 768,
}