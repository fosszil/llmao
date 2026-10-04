from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
from llama_cpp import Llama, llama_cpp

from .sampling import choose_token, softmax
from .tokenization import ContextPlan, plan_context

# Each completed conversation turn contains a user question
# and the assistant's answer.
History = Sequence[tuple[str, str]]


class InferenceEngine:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Llama initializes the backend, loads the GGUF weights,
        # and creates the inference context.
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_batch=256,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )

        self.vocab = llama_cpp.llama_model_get_vocab(self.model.model)
        self.rng = np.random.default_rng()
        self.last_plan: ContextPlan | None = None

    def tokenize(self, text: str) -> list[int]:
        """Convert UTF-8 text into the model's token IDs."""
        return self.model.tokenize(
            text.encode("utf-8"),
            add_bos=False,
            special=True,
        )

    def get_logits(self) -> np.ndarray:
        """Copy the latest next-token scores from llama.cpp."""
        pointer = llama_cpp.llama_get_logits_ith(self.model.ctx, -1)

        if not pointer:
            raise RuntimeError("Unable to read model logits")

        return np.ctypeslib.as_array(
            pointer,
            shape=(self.model.n_vocab(),),
        ).copy()

    def format_prompt(
        self,
        question: str,
        history: History = (),
    ) -> str:
        """Format our conversation using Qwen chat markers."""
        parts = [
            "<|im_start|>system\n"
            "You are a helpful, concise local AI assistant."
            "<|im_end|>\n"
        ]

        for user_text, assistant_text in history:
            parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n")

        parts.append(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")

        return "".join(parts)


    def prepare_tokens(
        self,
        question: str,
        history: History,
        max_new_tokens: int,
    ) -> ContextPlan:
        """Create a prompt and reserve space for generation."""

        return plan_context(
            question=question,
            history=history,
            context_limit=self.model.n_ctx(),
            output_reserve=max_new_tokens,
            tokenize=self.tokenize,
            format_prompt=self.format_prompt,
        )


    def inspect(self, question: str):
        """Inspect the model's next-token predictions."""

        tokens = self.prepare_tokens(
            question, (), 1
        ).token_ids

        self.model.reset()
        self.model.eval(tokens)

        logits = self.get_logits()
        probabilities = softmax(logits)

        top_ids = np.argsort(logits)[-5:][::-1]
        candidates = []

        for token_id in top_ids:
            token_id = int(token_id)

            token_text = self.model.detokenize(
                [token_id],
                prev_tokens=tokens,
            ).decode("utf-8", errors="replace")

            candidates.append(
                (
                    token_id,
                    token_text,
                    float(logits[token_id]),
                    float(probabilities[token_id]),
                )
            )

        return len(tokens), candidates

    def generate(
        self,
        question: str,
        history: History = (),
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Generate and stream new text from the local model."""

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        self.last_plan = None

        plan = self.prepare_tokens(
            question,
            history,
            max_new_tokens,
        )

        self.last_plan = plan
        prompt_tokens = plan.token_ids

        # Begin a fresh evaluation for this prompt.
        self.model.reset()
        self.model.eval(prompt_tokens)

        generated_tokens = []
        displayed_text = ""

        for step in range(max_new_tokens):
            if should_stop is not None and should_stop():
                break

            # 1. Read scores for the next token.
            logits = self.get_logits()

            # 2. Choose a token using our own sampler.
            token_id = choose_token(
                logits,
                temperature=temperature,
                top_k=40,
                rng=self.rng,
            )

            # 3. Stop if the model produced an ending token.
            if llama_cpp.llama_vocab_is_eog(self.vocab, token_id):
                break

            generated_tokens.append(token_id)

            # 4. Detokenize the output generated so far.
            # This avoids decoding incomplete UTF-8 sequences.
            output_bytes = self.model.detokenize(
                generated_tokens,
                prev_tokens=prompt_tokens,
            )

            current_text = output_bytes.decode("utf-8", errors="ignore")

            # 5. Stream only the newly available characters.
            new_text = current_text[len(displayed_text) :]

            if new_text:
                yield new_text

            displayed_text = current_text

            # 6. Evaluate the chosen token to predict the next.
            if step < max_new_tokens - 1:
                self.model.eval([token_id])


    def inspect_tokens(
        self,
        text: str,
        special: bool = False,
    ) -> list[tuple[int, int, str, str]]:
        """Inspect token IDs, text previews, and raw bytes."""

        token_ids = self.model.tokenize(
            text.encode("utf-8"),
            add_bos=False,
            special=special,
        )

        results = []

        for index, token_id in enumerate(token_ids):
            # Decode this token independently for inspection.
            piece = self.model.detokenize(
                [token_id],
                special=True,
            )

            # Backslash escaping preserves evidence of byte
            # sequences that are not valid UTF-8 on their own.
            preview = piece.decode(
                "utf-8",
                errors="backslashreplace",
            )

            results.append((
                index,
                token_id,
                preview,
                piece.hex(" "),
            ))

        return results

    def round_trip(self, text: str) -> tuple[list[int], str]:
        """Tokenize normal text and reconstruct it."""

        ids = self.model.tokenize(
            text.encode("utf-8"),
            add_bos=False,
            special=False,
        )

        decoded_bytes = self.model.detokenize(ids)

        return ids, decoded_bytes.decode("utf-8")

    def close(self):
        """Release the native model and context resources."""
        self.model.close()
