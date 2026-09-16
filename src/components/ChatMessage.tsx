import type { Message } from "../types/message";

type ChatMessageProps = {
  message: Message;
};

function ChatMessage({ message }: ChatMessageProps) {
  return (
    <article>
      <strong>{message.role === "user" ? "You" : "AI"}</strong>
      <p>{message.content}</p>
    </article>
  );
}

export default ChatMessage;
