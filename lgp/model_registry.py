MODELS = {}


def register_model(name, cls):
    global MODELS
    MODELS[name] = cls


def get_model(name):
    if name in MODELS:
        return MODELS[name]
    else:
        raise ValueError(f"Model with name {name} not present in model registry! Available models: {list(MODELS.keys())}")