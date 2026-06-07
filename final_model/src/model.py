from transformers import AutoModelForSequenceClassification

DEFAULT_MODEL = 'monologg/koelectra-base-v3-discriminator'
NUM_LABELS = 5


def build_model(num_labels: int = NUM_LABELS, model_name: str = DEFAULT_MODEL):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )


def load_model(model_path: str, num_labels: int = NUM_LABELS):
    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )
