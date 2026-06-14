from .gbm import make_gbm, gbm_fit_predict


def __getattr__(name):  # lazy import so torch is only needed when you use CNNs
    if name in ("CNN1D", "CNN2D"):
        from .cnn import CNN1D, CNN2D
        return {"CNN1D": CNN1D, "CNN2D": CNN2D}[name]
    if name == "ASTClassifier":
        from .ast import ASTClassifier
        return ASTClassifier
    raise AttributeError(name)
