
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

    else:
        from .ui import LabApp

        LabApp(model_path=args.model).run()


if __name__ == "__main__":
    main()
