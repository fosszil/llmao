import argparse
from pathlib import Path

from .embeddings import EmbeddingEngine
from .inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Local LLM learning application")

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen.gguf"),
        help="Path to the GGUF model",
    )

    commands = parser.add_subparsers(dest="command")


    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/bge-small.gguf"),
        help="Path to the embedding GGUF model",
    )


    inspect_parser = commands.add_parser(
        "inspect",
        help="Inspect next-token predictions",
    )
    inspect_parser.add_argument("question")

    tokens_parser = commands.add_parser(
        "tokens",
        help="Inspect token IDs and byte pieces",
    )

    tokens_parser.add_argument(
        "text",
        help="Text to tokenize",
    )

    tokens_parser.add_argument(
        "--special",
        action="store_true",
        help="Recognize supported special token markers",
    )

    embed_parser = commands.add_parser(
        "embed",
        help="Turn text into an embedding vector",
    )
    embed_parser.add_argument("text")


    args = parser.parse_args()

    if args.command == "inspect":
        engine = InferenceEngine(args.model)

        try:
            count, candidates = engine.inspect(args.question)

            print(f"Prompt tokens: {count}")
            print("Top next-token predictions:\n")

            for token_id, text, logit, probability in candidates:
                print(
                    f"ID={token_id:<8} "
                    f"text={text!r:<16} "
                    f"logit={logit:8.3f} "
                    f"p={probability:.3%}"
                )
        finally:
            engine.close()

    elif args.command == "tokens":
        engine = InferenceEngine(args.model)

        try:
            rows = engine.inspect_tokens(
                args.text,
                special=args.special,
            )

            print(f"Characters: {len(args.text)}")
            print(f"UTF-8 bytes: {len(args.text.encode('utf-8'))}")
            print(f"Tokens: {len(rows)}\n")

            for index, token_id, preview, hex_bytes in rows:
                print(f"{index:3} | {token_id:8} | {preview!r:18} | {hex_bytes}")
        finally:
            engine.close()

    elif args.command == "embed":
        engine = EmbeddingEngine(args.embedding_model)

        try:
            vector = engine.embed(args.text)

            print(f"Dimensions: {len(vector)}")
            print(
                f"L2 norm: {float(vector @ vector) ** 0.5:.4f}"
            )
            print(
                f"First 8 values: {vector[:8].round(4)}"
            )
        finally:
            engine.close()

    else:
        from .ui import LabApp

        LabApp(model_path=args.model).run()


if __name__ == "__main__":
    main()
