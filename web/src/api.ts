/**
 * API client.
 *
 * The interesting part is streamAnswer. EventSource cannot be used because it
 * only issues GET requests and /answer takes a JSON body, so the SSE frames are
 * parsed by hand off a fetch ReadableStream.
 *
 * Frames arrive split across arbitrary chunk boundaries — a single read can
 * contain half an event — so a buffer is carried between reads and only
 * complete frames (terminated by a blank line) are dispatched.
 */

export interface TrialHit {
  nct_id: string;
  title: string;
  conditions: string;
  status: string;
  url: string;
  score: number;
}

export interface Groundedness {
  grounded: boolean;
  overlap: number;
  cited_ids: string[];
  hallucinated_ids: string[];
  unsupported_terms: string[];
}

export interface StreamHandlers {
  onSources: (sources: TrialHit[]) => void;
  onToken: (text: string) => void;
  onFinal: (g: Groundedness, tookMs: number) => void;
  onError: (message: string) => void;
}

export async function search(query: string, topK = 5, openOnly = true) {
  const res = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, open_only: openOnly }),
  });
  if (!res.ok) throw new Error(`search failed: ${res.status}`);
  return (await res.json()) as { query: string; took_ms: number; results: TrialHit[] };
}

export async function streamAnswer(
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  topK = 5,
) {
  const res = await fetch("/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, open_only: true }),
    signal,
  });

  if (!res.ok || !res.body) {
    handlers.onError(`request failed: ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // A blank line terminates an SSE frame. Anything after the last one is a
    // partial frame and stays in the buffer for the next read.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const eventLine = frame.split("\n").find((l) => l.startsWith("event:"));
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const event = eventLine.slice(6).trim();
      const payload = JSON.parse(dataLine.slice(5).trim());

      switch (event) {
        case "sources":
          handlers.onSources(payload as TrialHit[]);
          break;
        case "token":
          handlers.onToken(payload.t as string);
          break;
        case "final":
          handlers.onFinal(payload.groundedness as Groundedness, payload.took_ms);
          break;
        case "error":
          handlers.onError(payload.error as string);
          break;
      }
    }
  }
}
