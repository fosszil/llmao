import type { Message } from "../../types/message";

type ChatMessageProps = {
  message: Message;
};

function ChatMessage({ message }: ChatMessageProps) {
  return (
    <article
      className={
        message.role === "user"
          ? "message user-message"
          : "message assistant-message"
      }
    >
      <strong>
        {message.role === "user"
          ? "You"
          : "Local LLM"}
      </strong>

      <p>{message.content}</p>
    </article>
  );
}

export default ChatMessage;
