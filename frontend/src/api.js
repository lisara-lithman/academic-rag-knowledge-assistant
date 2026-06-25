// frontend/src/api.js
// This file handles all communication with the FastAPI backend.

const BASE_URL = "http://localhost:8000";

/**
 * sendMessage - Sends a message to the backend and streams the response.
 *
 * @param {string}   message    - The user's question
 * @param {Array}    history    - Past messages: [["user msg", "bot reply"], ...]
 * @param {Function} onToken    - Called every time a new word/token arrives
 * @param {Function} onMetadata - Called once with { decision, chunks }
 * @param {Function} onDone     - Called when streaming is complete
 */
export async function sendMessage(message, history, onToken, onMetadata, onDone) {
  // 1. Make a POST request to /chat
  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });

  // 2. Get a "reader" so we can read the stream chunk by chunk
  const reader = response.body.getReader();
  const decoder = new TextDecoder(); // Converts raw bytes → text
  let buffer = "";

  // 3. Loop: keep reading until the stream ends
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // 4. Decode the bytes and add to our buffer
    buffer += decoder.decode(value, { stream: true });

    // 5. Split on double newline — each SSE event ends with \n\n
    const parts = buffer.split("\n\n");

    // 6. Process all complete events (keep the last incomplete one in buffer)
    for (let i = 0; i < parts.length - 1; i++) {
      const line = parts[i].trim();

      // SSE events start with "data: "
      if (line.startsWith("data: ")) {
        const jsonStr = line.slice(6); // Remove "data: " prefix
        try {
          const event = JSON.parse(jsonStr);

          if (event.type === "metadata") {
            onMetadata(event);       // Pass decision + chunks to UI
          } else if (event.type === "token") {
            onToken(event.content);  // Pass each word to UI
          } else if (event.type === "done") {
            onDone();                // Signal completion
          }
        } catch (e) {
          // Ignore parse errors on malformed chunks
        }
      }
    }

    // Keep the incomplete last part in the buffer
    buffer = parts[parts.length - 1];
  }
}