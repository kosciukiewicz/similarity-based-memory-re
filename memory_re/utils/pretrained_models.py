from memory_re.settings import PRETRAINED_MODELS_DIR


def get_pretrained_model_name(model_name: str) -> str:
    if PRETRAINED_MODELS_DIR is not None and (PRETRAINED_MODELS_DIR / model_name).exists():
        # Return pretrained locally saved model
        return str(PRETRAINED_MODELS_DIR / model_name)
    else:
        # Return raw model name to be handled by transformers library
        return model_name
