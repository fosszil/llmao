
import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Convert scores into a probability distribution."""

    values = np.asarray(scores, dtype=np.float64)

    # Shifting scores prevents exponential overflow.
    shifted = values - np.max(values)
    weights = np.exp(shifted)

    return weights / weights.sum()


def choose_token(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_k: int = 40,
    rng: np.random.Generator | None = None,
) -> int:
    """Select a token using greedy or top-k sampling."""

    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and non-negative")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Temperature zero uses deterministic greedy decoding.
    if temperature == 0:
        return int(np.argmax(logits))

    if rng is None:
        rng = np.random.default_rng()

    # Keep only the K highest-scoring candidate tokens.
    k = min(top_k, len(logits))

    candidate_ids = np.argpartition(
        logits, -k
    )[-k:]

    candidate_logits = np.asarray(logits[candidate_ids], dtype=np.float64)

    # Temperature changes the distribution's sharpness.
    # Shift before scaling so tiny temperatures cannot overflow to +inf.
    # Very unlikely candidates may become -inf, which gives probability zero.
    with np.errstate(over="ignore"):
        scaled_logits = (candidate_logits - candidate_logits.max()) / temperature
    probabilities = softmax(scaled_logits)

    # Draw a random token using the computed probabilities.
    selected = rng.choice(
        candidate_ids,
        p=probabilities,
    )

    return int(selected)
