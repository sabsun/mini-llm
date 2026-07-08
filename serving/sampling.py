import torch
import torch.nn.functional as F


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Scale logits by temperature.

    Lower temperature -> more deterministic.
    Higher temperature -> more random.
    """

    if temperature <= 0:
        return logits

    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Penalize tokens that have already appeared.
    """

    if penalty == 1.0:
        return logits

    logits = logits.clone()

    unique_tokens = torch.unique(generated_tokens)

    logits[unique_tokens] /= penalty

    return logits


def top_k_filter(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Keep only the top-k logits.
    """

    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k)

    cutoff = values[-1]

    logits = logits.clone()

    logits[logits < cutoff] = float("-inf")

    return logits


def top_p_filter(
    logits: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """
    Nucleus (top-p) filtering.
    """

    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    probs = F.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(probs, dim=-1)

    remove = cumulative_probs > p

    remove[1:] = remove[:-1].clone()

    remove[0] = False

    sorted_logits[remove] = float("-inf")

    filtered = torch.full_like(logits, float("-inf"))

    filtered.scatter_(0, sorted_indices, sorted_logits)

    return filtered


def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> int:
    """
    Sample the next token from logits.
    """

    logits = apply_repetition_penalty(
        logits,
        generated_tokens,
        repetition_penalty,
    )

    if temperature == 0:
        return torch.argmax(logits).item()

    logits = apply_temperature(
        logits,
        temperature,
    )

    logits = top_k_filter(
        logits,
        top_k,
    )

    logits = top_p_filter(
        logits,
        top_p,
    )

    probs = F.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )

    return next_token.item()