
import argparse
from pathlib import Path

from .inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="Local LLM learning application"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/qwen.gguf"),
        help="Path to the GGUF model",
    )

    commands = parser.add_subparsers(dest="command")

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
            print(
                f"UTF-8 bytes: "
                f"{len(args.text.encode('utf-8'))}"
            )
            print(f"Tokens: {len(rows)}\n")

            for index, token_id, preview, hex_bytes in rows:
                print(
                    f"{index:3} | "
                    f"{token_id:8} | "
                    f"{preview!r:18} | "
                    f"{hex_bytes}"
                )
        finally:
            engine.close()

    else:
        from .ui import LabApp

        LabApp(model_path=args.model).run()


if __name__ == "__main__":
    main()
