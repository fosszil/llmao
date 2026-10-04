from pathlib import Path
from threading import Event

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Input, Log, Static

from .inference import InferenceEngine


class LabApp(App):
    CSS = """
    Screen { background: $background; }
    #status { height: 1; color: $text-muted; padding: 0 2; }
    #chat { height: 1fr; padding: 1 2; }
    Input { margin: 0 1 1 1; border: none; background: $surface; }
    Input:focus { border: none; }
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
        yield Static("Loading model...", id="status")
        yield Log(id="chat", auto_scroll=True)
        yield Input(placeholder="Ask anything...", disabled=True)
        yield Footer()

    def on_mount(self):
        self.load_model()

    def status(self, message):
        self.query_one("#status", Static).update(f"{message}  |  Temp {self.temp}")

    @work(thread=True)
    def load_model(self):
        try:
            engine = InferenceEngine(self.model_path)
            self.call_from_thread(self.model_ready, engine)
        except Exception as e:
            self.call_from_thread(self.status, f"Error: {e}")

    def model_ready(self, engine):
        self.engine = engine
        self.busy = False
        self.status("Ready")
        self.query_one(Input).disabled = False
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted):
        raw = event.value.lstrip()
        text = raw.strip()
        event.input.value = ""

        if not text or self.busy:
            return

        if text.startswith("/"):
            self.command(text, raw)
            return

        self.busy = True
        event.input.disabled = True
        self.parts = []
        self.stop_event.clear()

        self.query_one(Log).write(f"\n> {text}\n\n")
        self.generate(text, list(self.history), self.temp)

    def command(self, text, raw):
        log = self.query_one(Log)

        if text == "/clear":
            self.history.clear()
            log.clear()

        elif text == "/help":
            log.write_line("/clear  /help  /temp 0.7  /tokens TEXT  /context")

        elif text.startswith("/temp "):
            try:
                value = float(text[len("/temp ") :])
                if not 0 <= value <= 2:
                    raise ValueError
                self.temp = value
                self.status("Ready")
            except ValueError:
                log.write_line("Temperature must be 0 to 2")

        elif raw.startswith("/tokens "):
            rows = self.engine.inspect_tokens(raw[len("/tokens ") :])
            log.write_line(f"\n{len(rows)} tokens")

            for index, token_id, piece, _ in rows:
                log.write_line(f"{index:3}  {token_id:8}  {piece!r}")

        elif text == "/context":
            limit = self.engine.model.n_ctx()
            prompt = self.engine.format_prompt("", self.history)
            used = len(self.engine.tokenize(prompt))
            remaining = limit - used - 160

            log.write_line(
                f"\nContext: {used}/{limit} tokens "
                f"| Available for next question: {max(0, remaining)}"
            )
            if remaining < 0:
                log.write_line("Old conversation turns will be trimmed for the next question.")

        else:
            log.write_line("Unknown command. Try /help")

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
                self.call_from_thread(self.add_chunk, chunk)
        except Exception as e:
            error = str(e)

        self.call_from_thread(self.finish, question, error)

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
                # Keep history consistent with the context planner.
                plan = self.engine.last_plan
                if plan and plan.dropped_turns:
                    del self.history[: plan.dropped_turns]
                    log.write(f"\n[{plan.dropped_turns} old turns removed]")

                self.history.append((question, answer))

        log.write("\n")
        self.busy = False
        self.status("Ready")
        self.query_one(Input).disabled = False
        self.query_one(Input).focus()

    def action_stop(self):
        if self.busy and self.engine:
            self.stop_event.set()
            self.status("Stopping...")

    def on_unmount(self):
        self.stop_event.set()
