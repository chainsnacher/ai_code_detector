"""Simple data labeler used by quick tests and components.

This stub provides a minimal `DataLabeler` class so imports and basic
labeling work if the real implementation is missing.
"""
from typing import Tuple


class DataLabeler:
    """Minimal labeler that marks code as 'ai' or 'human' using a naive rule."""

    def __init__(self):
        pass

    def label(self, text: str) -> Tuple[str, float]:
        """Return a tuple `(label, confidence)`.

        This is intentionally simple: if the text contains patterns common
        to AI-generated code (e.g., '/* generated */' or repeated boilerplate),
        return 'ai' with moderate confidence; otherwise return 'human'.
        """
        if not text or not text.strip():
            return 'unknown', 0.0

        lowered = text.lower()
        if 'generated' in lowered or 'assistant' in lowered or 'gpt' in lowered:
            return 'ai', 0.75

        # fallback to human with low confidence
        return 'human', 0.35


def label_text(text: str):
    dl = DataLabeler()
    return dl.label(text)
