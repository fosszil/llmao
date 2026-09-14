import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleAsk() {
    setIsLoading(true);
    setResponse("");

    try {
      const res = await invoke<string>('greet', { name: prompt });
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
      <p>Start chatting with your local LLM</p>

      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          handleAsk();
        }}
      >
        <input
          id="greet-input"
          value={prompt}
          onChange={(e) => setPrompt(e.currentTarget.value)}
          placeholder="Ask a question"
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Thinking..." : "Ask"}
        </button>
      </form>
      <div className="response-box">
        {isLoading && <p>Thinking about your answer...</p>}
        {response && <p>{response}</p>}
      </div>
    </main>
  );
}

export default App;
