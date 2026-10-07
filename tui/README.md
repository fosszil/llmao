# Local LLM Lab

A terminal app for experimenting with on-device AI. Built with Python, Textual, and llama.cpp, using Qwen for generation and BGE for embeddings.

## Features

- Local chat with streaming responses.
- RAG using BGE embeddings, vector similarity search, and Qwen to answer questions using retrieved context.
- Greedy decoding and temperature-controlled top-k sampling.
- Token inspection: IDs, text pieces, raw bytes, and next-token scores.
- Context budgeting using actual model tokens, with automatic trimming of older conversation turns.
- Generation cancellation with `Escape`.

Inference runs locally on the CPU using GGUF models, without cloud APIs.

## Setup

Requires Python 3.12+ and uv. From this directory:

```bash
uv sync --locked
```

Download compatible GGUF models separately and place them at:

```text
models/qwen.gguf
models/bge-small.gguf
```

Model weights are ignored by Git.

## CLI tools

```bash
uv run tui tokens "Hello, world!"
uv run tui inspect "What is a language model?"
uv run tui embed "A short piece of text."
```

## How it works

For chat, the app tokenizes the conversation, reserves space for a reply, and removes older exchanges when needed. llama.cpp evaluates the prompt, and a NumPy sampler selects each next token while streaming the decoded text.

For RAG, document passages are embedded with BGE. Vector similarity search selects passages relevant to the question and adds them to Qwen's prompt as context.

Chat uses a 2,048-token context window and generates up to 160 tokens per reply. Conversation history stays in memory for the session.

Source files are under `src/tui/`.
