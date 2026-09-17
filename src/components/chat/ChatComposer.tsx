type ChatComposerProps = {
  prompt: string;
  isLoading: boolean;
  onPromptChange: (value: string) => void;
  onSubmit: () => void;
};

function ChatComposer({
  prompt,
  isLoading,
  onPromptChange,
  onSubmit,
}: ChatComposerProps) {
  return (
    <form
      className="chat-composer"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <textarea
        value={prompt}
        onChange={(event) =>
          onPromptChange(event.currentTarget.value)
        }
        placeholder="Ask something..."
        rows={3}
        disabled={isLoading}
      />

      <button
        type="submit"
        disabled={
          isLoading || !prompt.trim()
        }
      >
        {isLoading ? "Thinking..." : "Send"}
      </button>
    </form>
  );
}

export default ChatComposer;
