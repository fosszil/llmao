from collections.abc import Callable, Sequence
from dataclasses import dataclass

History = Sequence[tuple[str, str]]


@dataclass(frozen=True)
class ContextPlan:
    """The tokens and budget for one inference request."""

    token_ids: list[int]
    context_limit: int
    output_reserve: int
    dropped_turns: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def remaining(self) -> int:
        return self.context_limit - self.prompt_tokens - self.output_reserve


def plan_context(
    *,
    question: str,
    history: History,
    context_limit: int,
    output_reserve: int,
    tokenize: Callable[[str], list[int]],
    format_prompt: Callable[[str, History], str],
) -> ContextPlan:
    """Build a prompt that fits inside the context window."""

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if output_reserve < 1:
        raise ValueError("Output reserve must be positive")

    if context_limit < 1:
        raise ValueError("Context limit must be positive")

    # We modify a copy so the original history is untouched.
    recent_history = list(history)
    dropped_turns = 0

    while True:
        # Reconstruct and tokenize the actual prompt after
        # each history change.
        prompt = format_prompt(question, recent_history)
        tokens = tokenize(prompt)

        total_required = len(tokens) + output_reserve

        if total_required <= context_limit:
            return ContextPlan(
                token_ids=tokens,
                context_limit=context_limit,
                output_reserve=output_reserve,
                dropped_turns=dropped_turns,
            )

        if not recent_history:
            raise ValueError(
                "Current question and output reserve exceed the context window"
            )

        # Remove one complete user/assistant exchange.
        recent_history.pop(0)
        dropped_turns += 1
