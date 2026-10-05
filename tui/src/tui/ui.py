
from pathlib import Path
from threading import Event

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Input, Log, Static

from .embeddings import EmbeddingEngine
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

    def __init__(self, model_path: Path, embedding_model_path: Path):
        super().__init__()

        self.model_path = model_path
        self.embedding_model_path = embedding_model_path

        self.engine = None
        self.embedder = None
        self.history = []
        self.parts = []

        self.temp = 0.0
        self.busy = True
        self.embedding_active = False
        self.stop_event = Event()

    def compose(self) -> ComposeResult:
        yield Static("Loading model...", id="status")
        yield Log(id="chat", auto_scroll=True)
        yield Input(placeholder="Ask anything...", disabled=True)
        yield Footer()

    def on_mount(self):
        self.load_model()

    def status(self, message):
        self.query_one("#status", Static).update(
            f"{message} | Temp {self.temp}"
        )

    @work(thread=True)
    def load_model(self):
        try:
            engine = InferenceEngine(self.model_path)
            self.call_from_thread(self.model_ready, engine)
        except Exception as e:
            self.call_from_thread(self.status, f"Error: {e}")

    def model_ready(self, engine):
        self.engine = engine
        self.finish_job()

    def start_job(self, message):
        self.busy = True
        self.query_one(Input).disabled = True
        self.status(message)

    def finish_job(self):
        self.busy = False
        self.status("Ready")

        prompt = self.query_one(Input)
        prompt.disabled = False
        prompt.focus()

    def on_input_submitted(self, event: Input.Submitted):
        raw = event.value.lstrip()
        text = raw.strip()
        event.input.value = ""

        if not text or self.busy:
            return

        if text.startswith("/"):
            self.command(text, raw)
            return

        self.start_job("Generating...")
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
            log.write_line(
                "/clear  /help  /temp 0.7  /tokens TEXT"
                "  /context  /embed TEXT"
            )

        elif text.startswith("/temp "):
            try:
                value = float(text[len("/temp "):])
                if not 0 <= value <= 2:
                    raise ValueError

                self.temp = value
                self.status("Ready")
            except ValueError:
                log.write_line("Temperature must be 0 to 2")

        elif text.startswith("/tokens "):
            try:
                rows = self.engine.inspect_tokens(raw[len("/tokens "):])
                log.write_line(f"\n{len(rows)} tokens")

                for index, token_id, piece, _ in rows:
                    log.write_line(
                        f"{index:3}  {token_id:8}  {piece!r}"
                    )
            except Exception as e:
                log.write_line(f"Tokenization error: {e}")

        elif text == "/context":
            limit = self.engine.model.n_ctx()
            prompt = self.engine.format_prompt("", self.history)
            used = len(self.engine.tokenize(prompt))
            remaining = limit - used - 160

            log.write_line(
                f"\nContext: {used}/{limit} tokens"
                f" | Room for next question: {remaining}"
            )

            if remaining < 0:
                log.write_line(
                    "Older turns may need to be trimmed."
                )

        elif text == "/embed" or text.startswith("/embed "):
            source = raw[len("/embed"):].strip()

            if not source:
                log.write_line("Usage: /embed TEXT")
                return

            self.embedding_active = True
            self.start_job("Embedding...")
            self.create_embedding(source)

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
                plan = self.engine.last_plan

                if plan and plan.dropped_turns:
                    del self.history[:plan.dropped_turns]
                    log.write(
                        f"\n[{plan.dropped_turns} old turns removed]"
                    )

                self.history.append((question, answer))

        log.write("\n")
        self.finish_job()

    @work(thread=True)
    def create_embedding(self, text):
        try:
            # Load only on the first embedding request.
            if self.embedder is None:
                self.embedder = EmbeddingEngine(
                    self.embedding_model_path
                )

            vector = self.embedder.embed(text)

            result = (
                f"\nEmbedding: {text}\n"
                f"Dimensions: {len(vector)}\n"
                f"First 8: {vector[:8].round(4)}\n"
                f"Norm: {float(vector @ vector) ** 0.5:.4f}"
            )

        except Exception as e:
            result = f"Embedding error: {e}"

        self.call_from_thread(self.embedding_done, result)

    def embedding_done(self, result):
        self.query_one(Log).write_line(result)
        self.embedding_active = False
        self.finish_job()

    def action_stop(self):
        # Escape only stops text generation, not embeddings.
        if self.busy and self.engine and not self.embedding_active:
            self.stop_event.set()
            self.status("Stopping...")

    def on_unmount(self):
        self.stop_event.set()
