
from pathlib import Path

import numpy as np
from llama_cpp import Llama, llama_cpp


class EmbeddingEngine:
    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        self.model = Llama(
            model_path=str(model_path),
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_CLS,
            n_ctx=512,
            n_batch=512,
            n_gpu_layers=0,
            verbose=False,
        )

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Turn the complete input into one normalized vector.
        vector = self.model.embed(
            text,
            normalize=True,
            truncate=False,
        )

        return np.asarray(vector, dtype=np.float32)

    def close(self):
        self.model.close()
