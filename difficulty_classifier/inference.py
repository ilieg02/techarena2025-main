import joblib
from typing import Union, List

# Load and wrap the sklearn pipeline once (module-level cache)
_sk_model = None
def _load_rf_pipeline(pkl_path: str = "difficulty_classifier.pkl"):
    global _sk_model
    if _sk_model is None:
        _sk_model = joblib.load(pkl_path)
    return _sk_model

def is_complex(question: Union[str, List[str]], pkl_path: str = "difficulty_classifier.pkl") -> Union[bool, List[bool]]:
    """
    Returns True if question is classified as 'complex' (medium/hard merged), False otherwise.
    Accepts a single string or a list of strings.
    """
    model = _load_rf_pipeline(pkl_path)

    single = False
    if isinstance(question, str):
        single = True
        inputs = [question]
    else:
        inputs = list(question)

    preds = model.predict(inputs)  # preds are label strings, e.g., 'easy' or 'complex'
    results = [True if p == "complex" else False for p in preds]

    return results[0] if single else results
