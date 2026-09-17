import ChatMessage from "./ChatMessage";

import type { Message } from "../../types/message";

type ChatMessagesProps = {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
};

function ChatMessages({
  messages,
  isLoading,
  error,
}: ChatMessagesProps) {
  return (
    <section
      className="chat-messages"
      aria-live="polite"
    >
      {messages.length === 0 && (
        <article>
          <strong>Local LLM</strong>

          <p>
            Load a model and start a conversation.
          </p>
        </article>
      )}

      {messages.map((message) => (
        <ChatMessage
          key={message.id}
          message={message}
        />
      ))}

      {isLoading && (
        <article aria-busy="true">
          Thinking...
        </article>
      )}

      {error && (
        <p role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

export default ChatMessages;
