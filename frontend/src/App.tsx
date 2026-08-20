import { FormEvent, KeyboardEvent, useEffect, useState } from "react";

type Conversation = {
  thread_id: string;
  title: string;
  created_at: string;
};
type Message = { role: "user" | "assistant" | "decision"; content: string };
type Interrupt = { kind: string; action?: string };
type ThreadState = {
  conversation: Conversation;
  messages: Message[];
  interrupt: Interrupt | null;
};
type ChatResponse = {
  thread_id: string;
  reply: string | null;
  interrupt: Interrupt | null;
};
type RecognitionEvent = Event & {
  results: { [index: number]: { [index: number]: { transcript: string } } };
};
type Recognition = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  onresult: ((event: RecognitionEvent) => void) | null;
  onerror: (() => void) | null;
  onend: (() => void) | null;
};
type RecognitionConstructor = new () => Recognition;
type SpeechLanguage = "es-MX" | "en-US";
type SpeechWindow = Window & {
  SpeechRecognition?: RecognitionConstructor;
  webkitSpeechRecognition?: RecognitionConstructor;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

function titleForFirstMessage(message: string): string {
  const firstLine = message.replace(/\s+/g, " ").trim();
  const preview =
    firstLine.length > 48 ? `${firstLine.slice(0, 48)}…` : firstLine;
  const time = new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
  return `${preview}\n${time}`;
}

export default function App() {
  const [threads, setThreads] = useState<Conversation[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [interrupt, setInterrupt] = useState<Interrupt | null>(null);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechLanguage, setSpeechLanguage] = useState<SpeechLanguage>(() =>
    navigator.language.startsWith("es") ? "es-MX" : "en-US",
  );
  const [error, setError] = useState("");

  const loadThreads = async (): Promise<void> => {
    const data = await request<Conversation[]>("/threads");
    setThreads(data);
    setThreadId((current) => current ?? data[0]?.thread_id ?? null);
  };

  useEffect(() => {
    void loadThreads().catch((cause: Error) => setError(cause.message));
  }, []);

  useEffect(() => {
    if (!threadId || pending) return;
    let cancelled = false;
    void request<ThreadState>(`/threads/${threadId}`)
      .then((state) => {
        if (!cancelled) {
          setConversation(state.conversation);
          setMessages(state.messages);
          setInterrupt(state.interrupt);
        }
      })
      .catch((cause: Error) => !cancelled && setError(cause.message));
    return () => {
      cancelled = true;
    };
  }, [threadId, pending]);

  const createThread = async (firstMessage: string): Promise<string> => {
    const created = await request<Conversation>("/threads", {
      method: "POST",
      body: JSON.stringify({ title: titleForFirstMessage(firstMessage) }),
    });
    setThreads((current) => [created, ...current]);
    setConversation(created);
    setThreadId(created.thread_id);
    setMessages([]);
    setInterrupt(null);
    return created.thread_id;
  };

  const selectThread = (selectedId: string): void => {
    setThreadId(selectedId);
    setConversation(
      threads.find((thread) => thread.thread_id === selectedId) ?? null,
    );
    setMessages([]);
    setInterrupt(null);
    setError("");
  };

  const startNewConversation = (): void => {
    setThreadId(null);
    setConversation(null);
    setMessages([]);
    setInterrupt(null);
    setDraft("");
    setError("");
  };

  // Shared by sendMessage and resume: both just fire one graph turn and
  // apply the same interrupt/reply/error handling to its response.
  const runTurn = async (
    action: () => Promise<ChatResponse>,
  ): Promise<void> => {
    setPending(true);
    setError("");
    try {
      const result = await action();
      setInterrupt(result.interrupt);
      if (result.reply) {
        setMessages((current) => [
          ...current,
          { role: "assistant", content: result.reply as string },
        ]);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Request failed");
    } finally {
      setPending(false);
    }
  };

  const sendMessage = (event: FormEvent): Promise<void> => {
    event.preventDefault();
    const message = draft.trim();
    if (!message || pending || interrupt) return Promise.resolve();
    return runTurn(async () => {
      const activeThreadId = threadId ?? (await createThread(message));
      setMessages((current) => [
        ...current,
        { role: "user", content: message },
      ]);
      setDraft("");
      return request<ChatResponse>(`/threads/${activeThreadId}/messages`, {
        method: "POST",
        body: JSON.stringify({ message }),
      });
    });
  };

  const resume = (approved: boolean): Promise<void> => {
    if (!threadId) return Promise.resolve();
    return runTurn(() =>
      request<ChatResponse>(`/threads/${threadId}/resume`, {
        method: "POST",
        body: JSON.stringify({ approved }),
      }),
    );
  };

  const startSpeechToText = (): void => {
    const RecognitionApi =
      (window as SpeechWindow).SpeechRecognition ??
      (window as SpeechWindow).webkitSpeechRecognition;
    if (!RecognitionApi) {
      setError("Speech recognition is not supported by this browser.");
      return;
    }
    const recognition = new RecognitionApi();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = speechLanguage;
    recognition.onresult = (event) => {
      const transcript = event.results[0]?.[0]?.transcript.trim();
      if (transcript) setDraft((current) => `${current} ${transcript}`.trim());
    };
    recognition.onerror = () =>
      setError("Speech recognition could not transcribe that audio.");
    recognition.onend = () => setListening(false);
    setListening(true);
    recognition.start();
  };

  const sendOnEnter = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 md:grid md:grid-cols-[17rem_minmax(0,1fr)]">
      <aside className="border-b border-slate-800 bg-slate-900 p-4 md:border-r md:border-b-0">
        <div className="flex items-start justify-between gap-3 md:block">
          <div>
            <p className="font-mono text-xs tracking-widest text-slate-400 uppercase">
              LangGraph + Vertex AI
            </p>
            <h1 className="mt-1 text-lg font-semibold">Chat</h1>
          </div>
          <button
            className="rounded-md bg-emerald-400 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-300 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white md:mt-5 md:w-full"
            onClick={startNewConversation}
            disabled={pending}
          >
            New
          </button>
        </div>
        <nav
          className="mt-4 flex gap-2 overflow-x-auto pb-1 md:block md:space-y-2"
          aria-label="Saved conversations"
          role="tablist"
        >
          {threads.map((thread) => (
            <button
              className={`min-w-48 rounded-md border px-3 py-2 text-left text-sm transition md:block md:w-full ${thread.thread_id === threadId ? "border-emerald-400 bg-emerald-400/10 text-white" : "border-slate-700 text-slate-300 hover:border-slate-500 hover:text-white"}`}
              key={thread.thread_id}
              onClick={() => selectThread(thread.thread_id)}
              disabled={pending}
              aria-current={thread.thread_id === threadId ? "page" : undefined}
              role="tab"
            >
              <span className="block whitespace-pre-line">{thread.title}</span>
            </button>
          ))}
        </nav>
      </aside>

      <section
        className="mx-auto grid min-h-[calc(100vh-9.5rem)] w-full max-w-4xl grid-rows-[auto_minmax(0,1fr)_auto_auto] gap-4 p-4 md:min-h-screen md:p-8"
        aria-label="Chat"
      >
        <header>
          <p className="font-mono text-xs tracking-widest text-slate-400 uppercase">
            Persistent thread
          </p>
          <h2 className="mt-1 text-xl font-semibold">
            {threadId ? "Conversation" : "Start a conversation"}
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            Each tab loads only its selected thread. Ask for UTC time or request
            an approval.
          </p>
        </header>

        <div
          className="grid content-start gap-3 overflow-y-auto"
          aria-live="polite"
          aria-busy={pending}
        >
          {messages.length === 0 && (
            <p className="text-sm text-slate-400">
              Your thread is stored in PostgreSQL.
            </p>
          )}
          {messages.map((message, index) =>
            message.role === "decision" ? (
              <p
                className={`justify-self-center rounded-full border px-4 py-1.5 text-xs font-semibold ${message.content.startsWith("Approved") ? "border-emerald-400/60 bg-emerald-400/10 text-emerald-300" : "border-slate-600 bg-slate-800 text-slate-300"}`}
                key={`decision-${index}`}
              >
                {message.content}
              </p>
            ) : (
              <article
                className={`max-w-2xl rounded-lg border px-4 py-3 text-sm leading-6 ${message.role === "user" ? "justify-self-end border-slate-600 bg-slate-800" : "justify-self-start border-slate-800 bg-slate-900"}`}
                key={`${message.role}-${index}`}
              >
                <p className="font-mono text-xs text-slate-400">
                  {message.role === "user" ? "You" : "Agent"}
                </p>
                <p className="mt-1 whitespace-pre-wrap">{message.content}</p>
              </article>
            ),
          )}
          {pending && <p className="text-sm text-slate-400">Working…</p>}
        </div>

        {interrupt && (
          <section
            className="border-l-4 border-amber-400 bg-amber-400/10 p-4"
            aria-label="Approval needed"
          >
            <strong className="text-sm">Approval needed</strong>
            <p className="mt-1 text-sm text-slate-300">
              {interrupt.action ?? "The agent needs your input."}
            </p>
            <div className="mt-3 flex gap-2">
              <button
                className="rounded-md bg-emerald-400 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-300"
                onClick={() => void resume(true)}
                disabled={pending}
              >
                Approve
              </button>
              <button
                className="rounded-md border border-slate-500 px-3 py-2 text-sm font-semibold hover:border-slate-300"
                onClick={() => void resume(false)}
                disabled={pending}
              >
                Reject
              </button>
            </div>
          </section>
        )}
        {error && (
          <p className="text-sm text-red-300" role="alert">
            {error}
          </p>
        )}

        <form onSubmit={sendMessage}>
          <label className="mb-2 block text-sm font-medium" htmlFor="message">
            Message
          </label>
          <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto_auto]">
            <textarea
              className="min-h-24 w-full rounded-md border border-slate-700 bg-slate-900 p-3 text-sm outline-none placeholder:text-slate-500 focus:border-emerald-400 disabled:opacity-50"
              id="message"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={sendOnEnter}
              placeholder="Message the agent"
              disabled={pending || Boolean(interrupt)}
              rows={2}
            />
            <label className="sr-only" htmlFor="speech-language">
              Speech language
            </label>
            <select
              className="rounded-md border border-slate-600 bg-slate-900 px-2 text-sm disabled:opacity-50"
              id="speech-language"
              value={speechLanguage}
              onChange={(event) =>
                setSpeechLanguage(event.target.value as SpeechLanguage)
              }
              disabled={pending || Boolean(interrupt) || listening}
            >
              <option value="es-MX">Español</option>
              <option value="en-US">English</option>
            </select>
            <button
              className="rounded-md border border-slate-600 px-3 py-2 text-sm font-semibold hover:border-slate-300 disabled:opacity-50"
              type="button"
              onClick={startSpeechToText}
              disabled={pending || Boolean(interrupt) || listening}
            >
              {listening ? "Listening…" : "Speak"}
            </button>
            <button
              className="rounded-md bg-emerald-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-300 disabled:opacity-50"
              type="submit"
              disabled={pending || !draft.trim() || Boolean(interrupt)}
            >
              Send
            </button>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Enter sends · Shift+Enter adds a line · Speak uses your browser's
            speech recognition and sends only its transcript to this app.
          </p>
        </form>
      </section>
    </main>
  );
}
