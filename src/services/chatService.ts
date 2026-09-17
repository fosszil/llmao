  import { invoke } from "@tauri-apps/api/core";

  export async function sendPrompt(prompt: string): Promise<string> {
    return invoke<string>("greet", {
      name: prompt,
    });
  }
