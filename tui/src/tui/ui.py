
from pathlib import Path
from threading import Event

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Input, Log, Static, Footer

from .inference import InferenceEngine


class LabApp(App):
    CSS = """
    Log {
        height: 1fr;
        border: round $primary;
    }
    """

    BINDINGS = [
        ("escape", "stop", "Stop"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, model_path: Path):
        super().__init__()

        self.model_path = model_path
        self.engine = None
        self.history = []
        self.parts = []

        self.temp = 0.0
        self.busy = True
        self.stop_event = Event()

    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="status")
        yield Log(id="chat")
        yield Input(
            placeholder="Ask something...",
            disabled=True,
        )
        yield Footer()

    def on_mount(self):
        self.load_model()

    @work(thread=True)
    def load_model(self):
        try:
            engine = InferenceEngine(self.model_path)
            self.call_from_thread(
                self.model_ready, engine
            )
        except Exception as e:
            self.call_from_thread(
                self.show_error, str(e)
            )

    def model_ready(self, engine):
        self.engine = engine
        self.busy = False

        self.set_status("Ready")

        prompt = self.query_one(Input)
        prompt.disabled = False
        prompt.focus()

    def show_error(self, message):
        self.set_status(f"Error: {message}")

    def set_status(self, message):
        self.query_one(Static).update(
            f"{message} | Temp: {self.temp}"
        )

    def on_input_submitted(self, event):
        text = event.value.strip()
        event.input.value = ""

        if not text or self.busy:
            return

        if text.startswith("/"):
            self.handle_command(text)
            return

        self.busy = True
        event.input.disabled = True

        self.parts = []
        self.stop_event.clear()

        self.query_one(Log).write(
            f"\nYOU: {text}\nAI: "
        )

        self.generate(
            text,
            list(self.history),
            self.temp,
        )

    def handle_command(self, text):
        log = self.query_one(Log)

        if text == "/clear":
            self.history.clear()
            log.clear()

        elif text == "/help":
            log.write_line(
                "/clear, /temp 0.7, /help"
            )

        elif text.startswith("/temp "):
            try:
                value = float(text.split()[1])
                if not 0 <= value <= 2:
                    raise ValueError

                self.temp = value
                self.set_status("Ready")

            except (ValueError, IndexError):
                log.write_line("Use /temp 0 to 2")

        else:
            log.write_line("Unknown command")

    @work(thread=True)
    def generate(self, question, history, temp):
        error = None

        try:
            for chunk in self.engine.generate(
                question,
                history=history,
                temperature=temp,
                should_stop=self.stop_event.is_set,
            ):
                self.call_from_thread(
                    self.add_chunk, chunk
                )

        except Exception as e:
            error = str(e)

        self.call_from_thread(
            self.finish, question, error
        )

    def add_chunk(self, chunk):
        self.parts.append(chunk)
        self.query_one(Log).write(chunk)

    def finish(self, question, error):
        log = self.query_one(Log)

        if error:
            log.write(f"\nError: {error}")

        elif self.stop_event.is_set():
            log.write("\n[Stopped]")

        else:
            answer = "".join(self.parts)
            if answer:
                self.history.append(
                    (question, answer)
                )

        log.write("\n")

        self.busy = False
        self.set_status("Ready")

        prompt = self.query_one(Input)
        prompt.disabled = False
        prompt.focus()

    def action_stop(self):
        if self.busy and self.engine:
            self.stop_event.set()
            self.set_status("Stopping...")

    def on_unmount(self):
        self.stop_event.set()
