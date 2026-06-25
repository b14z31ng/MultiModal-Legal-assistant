from transformers import AutoTokenizer


class TextProcessor:
    """
    Shared tokenizer for multimodal training.
    """

    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.max_length = max_length

    def __call__(self, text):

        encoded = self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_tensors="pt",

        )

        return {

            "input_ids":
                encoded["input_ids"].squeeze(0),

            "attention_mask":
                encoded["attention_mask"].squeeze(0),

        }