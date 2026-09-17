import ChatComposer from "./components/chat/ChatComposer";
import ChatHeader from "./components/chat/ChatHeader";
import ChatMessages from "./components/chat/ChatMessages";

import { useChat } from "./hooks/useChat";

import "./styles/chat.css";

function App() {
  const {
    prompt,
    setPrompt,
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
  } = useChat();

  return (
    <main className="chat-shell">
      <ChatHeader
        onNewChat={clearChat}
      />

      <ChatMessages
        messages={messages}
        isLoading={isLoading}
        error={error}
      />

      <ChatComposer
        prompt={prompt}
        isLoading={isLoading}
        onPromptChange={setPrompt}
        onSubmit={sendMessage}
      />
    </main>
  );
}

export default App;
