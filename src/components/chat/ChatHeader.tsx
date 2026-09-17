type ChatHeaderProps = {
  modelName?: string;
  onNewChat: () => void;
};

function ChatHeader({
  modelName,
  onNewChat,
}: ChatHeaderProps) {
  return (
    <header className="chat-header">
      <div>
        <strong>Local LLM</strong>
        <small>Private, local AI</small>
      </div>

      <div className="chat-header-actions">
        <span data-badge>
          {modelName ?? "No model"}
        </span>

        <button
          type="button"
          onClick={onNewChat}
        >
          New chat
        </button>
      </div>
    </header>
  );
}

export default ChatHeader;
