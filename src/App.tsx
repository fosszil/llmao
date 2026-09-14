import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleAsk() {
    if (!prompt.trim()) {
      return;
    }

    setIsLoading(true);
    setResponse("");

    try {
      const res = await invoke<string>("greet", {
        name: prompt,
      });

      setResponse(res);
      setPrompt("");
    } catch (error) {
      console.error(error);
      setResponse("Oops, something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="container">
      <header>
        <h1>Local LLM</h1>
        <p>Start chatting with your local LLM.</p>
      </header>

      <section>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            handleAsk();
          }}
        >
          <label htmlFor="prompt">Prompt</label>

          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.currentTarget.value)}
            placeholder="Ask a question..."
            rows={4}
            disabled={isLoading}
          />

          <button
            type="submit"
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? "Thinking..." : "Ask"}
          </button>
        </form>
      </section>

      <section aria-live="polite">
        {isLoading && <p>Thinking about your answer...</p>}

        {response && (
          <article>
            <strong>Assistant</strong>
            <p>{response}</p>
          </article>
        )}
      </section>
    </main>
  );
}

export default App;
