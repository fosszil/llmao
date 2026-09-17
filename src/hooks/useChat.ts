import { useState } from "react";

import type { Message } from "../types/message";
import { sendPrompt } from "../services/chatService";

export function useChat() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function sendMessage() {
    const trimmedPrompt = prompt.trim();

    if (!trimmedPrompt || isLoading) {
      return;
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedPrompt,
    };

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
    ]);

    setPrompt("");
    setError(null);
    setIsLoading(true);

    try {
      const response = await sendPrompt(trimmedPrompt);

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response,
      };

      setMessages((currentMessages) => [
        ...currentMessages,
        assistantMessage,
      ]);
    } catch (error) {
      console.error("Failed to generate response:", error);

      setError("Something went wrong while generating the response.");
    } finally {
      setIsLoading(false);
    }
  }

  function clearChat() {
    setMessages([]);
    setPrompt("");
    setError(null);
  }

  return {
    prompt,
    setPrompt,
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
  };
}
