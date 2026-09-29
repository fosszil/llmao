
from pathlib import Path
from threading import Event

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, Log, Static

from .inference import InferenceEngine


class LabApp(App):
    TITLE = "LLM Lab"

    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 3;
        padding: 1 2;
        background: $surface;
    }

    #chat {
        height: 1fr;
        border: round $primary;
    }

    #prompt {
        height: 3;
    }
    """

    BINDINGS = [
        ("escape", "stop", "Stop generation"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model_path: Path):
        super().__init__()

        self.model_path = model_path
        self.engine: InferenceEngine | None = None

        # Store completed conversation turns in memory.
        self.history: list[tuple[str, str]] = []

        self.response_parts: list[str] = []
        self.temperature = 0.0
        self.busy = True

        # A thread-safe signal used to stop generation.
        self.stop_event = Event()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Loading model...", id="status")
        yield Log(id="chat", auto_scroll=True)
        yield Input(
            placeholder="Ask your local model...",
            id="prompt",
            disabled=True,
        )
        yield Footer()

    def on_mount(self):
        self.load_engine()

    @work(thread=True)
    def load_engine(self):
        """Load the model without blocking the UI."""
        try:
            engine = InferenceEngine(self.model_path)
        except Exception as error:
            self.call_from_thread(
                self.model_failed, str(error)
            )
        else:
            self.call_from_thread(self.model_ready, engine)

    def model_ready(self, engine: InferenceEngine):
        self.engine = engine
        self.busy = False

        self.query_one("#status", Static).update(
            "Ready | temperature=0.0 | /help for commands"
        )

        self.query_one("#chat", Log).write_line(
            "Model loaded. Start chatting!\n"
        )

        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

    def model_failed(self, message: str):
        self.query_one("#status", Static).update(
            f"Model loading failed: {message}"
        )

    def on_input_submitted(self, event: Input.Submitted):
        """Handle a question or a built-in command."""

        if self.busy:
            return

        question = event.value.strip()
        event.input.value = ""

        if not question:
            return

        log = self.query_one("#chat", Log)

        if question == "/clear":
            self.history.clear()
            log.clear()
            log.write_line("Conversation cleared.")
            return

        if question == "/help":
            log.write_line(
                "Commands: /help, /clear, /temp NUMBER"
            )
            log.write_line(
                "Escape stops generation. Ctrl+Q quits."
            )
            return

        if question.startswith("/temp "):
            try:
                value = float(question.split()[1])

                if not 0 <= value <= 2:
                    raise ValueError

            except (ValueError, IndexError):
                log.write_line(
                    "Temperature must be between 0 and 2."
                )
                return

            self.temperature = value

            self.query_one("#status", Static).update(
                f"Ready | temperature={value}"
            )
            return

        if question.startswith("/"):
            log.write_line("Unknown command. Try /help.")
            return

        # Only one inference request may use this context at a time.
        self.busy = True
        event.input.disabled = True

        self.response_parts = []
        self.stop_event.clear()

        log.write(f"\nYOU: {question}\nAI: ")

        # Pass a snapshot of the previous conversation.
        self.generate_reply(question, list(self.history))

    @work(thread=True)
    def generate_reply(
        self,
        question: str,
        history: list[tuple[str, str]],
    ):
        """Run blocking inference in a background thread."""

        assert self.engine is not None

        error_message = None

        try:
            for chunk in self.engine.generate(
                question,
                history=history,
                temperature=self.temperature,
                should_stop=self.stop_event.is_set,
            ):
                # Widget updates must run on Textual's UI thread.
                self.call_from_thread(
                    self.append_chunk, chunk
                )

        except Exception as error:
            error_message = str(error)

        self.call_from_thread(
            self.finish_reply,
            question,
            error_message,
            self.stop_event.is_set(),
        )

    def append_chunk(self, chunk: str):
        """Display new text and remember it for conversation history."""

        self.response_parts.append(chunk)
        self.query_one("#chat", Log).write(chunk)

    def finish_reply(
        self,
        question: str,
        error_message: str | None,
        stopped: bool,
    ):
        log = self.query_one("#chat", Log)
        response = "".join(self.response_parts)

        if error_message:
            log.write(f"\nError: {error_message}")

        elif stopped:
            log.write("\n[Generation stopped]")

        else:
            # Only completed answers enter conversation history.
            if response:
                self.history.append((question, response))

        log.write("\n")

        self.busy = False

        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

        self.query_one("#status", Static).update(
            f"Ready | temperature={self.temperature}"
        )

    def action_stop(self):
        """Request cancellation without interrupting native code."""
        if self.busy and self.engine is not None:
            self.stop_event.set()
            self.query_one("#status", Static).update(
                "Stopping generation..."
            )

    def on_unmount(self):
        # Signal the worker to stop when the UI is shutting down.
        self.stop_event.set()
