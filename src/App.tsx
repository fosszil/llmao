import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

type Message = {
  role: "User" | "AI";
  content: string;
}

function App() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  async function handleAsk() {
    const trimmedPrompt = prompt.trim();

    if (!prompt.trim()) {
      return;
    }

    const userMessage: Message = {
      role: "User",
      content: trimmedPrompt,
    };

    setMessages((currentMessages) => [...currentMessages, userMessage]);

    setIsLoading(true);
    setPrompt("");

    try {
      const res = await invoke<string>("greet", {
        name: trimmedPrompt,
      });

      const assistantMessage: Message = {
        role: "AI",
        content: res,
      };

      setMessages((currentMessages) => [...currentMessages, assistantMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage: Message = {
        role: "AI",
        content: "Oops, something went wrong.",
      };
      setMessages((currentMessages) => [...currentMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="container">
      <header>
        <h1>Llmao</h1>
        <p>Start chatting with your local LLM.</p>
      </header>

      <section aria-live="polite">
        {messages.length === 0 && (
          <p>No messages yet. Start a conversation below.</p>
        )}

        {messages.map((message, index) => (
          <article key={index}>
            <strong>
              {message.role == "User" ? "You" : "AI"}
            </strong>
            <p>{message.content}</p>
          </article>
        ))}

        {isLoading && <p>Thinking about your answer...</p>}
      </section>

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
          <button type="submit" disabled={isLoading || !prompt.trim()}>
            {isLoading ? "Thinking..." : "Ask"}
          </button>
        </form>

      </section>
    </main>
  );
}

export default App;
