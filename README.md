# LangGraph + Docker POC

A deliberately small, explainable quick-start stack:

- Python 3.13 and `uv`
- async FastAPI routes, kept together in one file
- Gemini on Vertex AI with Google ADC
- `create_agent`, which is a compiled LangGraph state graph
- PostgreSQL for durable LangGraph checkpoints and a SQLModel conversation list
- a single React chat page, served by FastAPI after the Docker build

## Python layout

```text
app/settings.py   # environment validation and URLs
app/db.py         # SQLModel Conversation and async session setup
app/langgraph.py  # model, tools, graph config, interrupt helpers
app/main.py       # lifespan, typed DB dependency, and FastAPI routes
```

This is intentionally the full backend structure—there are no service or repository layers.

## RPI: research → plan → implementation

### Decisions

| Concern | Decision | Why |
| --- | --- | --- |
| Agent API | `create_agent` | It is already a LangGraph graph, accepts a checkpointer, and avoids Deep Agents' planning/filesystem/subagent overhead. |
| Vertex chat model | `ChatGoogleGenerativeAI(..., vertexai=True)` | It is the current Gemini-on-Vertex integration; `ChatVertexAI` is deprecated. |
| Checkpoint DB | PostgreSQL + `AsyncPostgresSaver` | First-party async persistence, durable interrupts, and a natural home for a tiny `conversations` table. |
| Thread identity | UUID generated at `POST /threads` | Opaque, collision-resistant, returned to the UI, and used as LangGraph's `configurable.thread_id`. |
| UI | one React + Tailwind component | Responsive tabs load only the selected conversation; browser speech-to-text is opt-in and local. |

PostgreSQL wins over MongoDB here. MongoDB has a supported saver and is a good choice when it is already the application's operational database; it adds no advantage to this POC. PostgreSQL makes thread listing and future relational application data simple, while LangGraph owns its checkpoint tables. Do not query those internal tables directly.

### Compact plan

1. Start PostgreSQL and the API with one Compose command.
2. Create a UUID thread on its first message and save its title/metadata through SQLModel.
3. Invoke the compiled `create_agent` graph with that UUID in `configurable.thread_id`.
4. Let `AsyncPostgresSaver` persist every state update; return either a reply or an interrupt.
5. Validate and persist the selected approval value with its custom action, then resume with `Command(resume=...)` and the exact same thread ID.

## Run it

1. Enable Vertex AI, then create local Application Default Credentials:

   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud auth application-default login
   ```

2. Copy the environment template, then replace the project ID and ADC path with real values:

   ```bash
   cp .env.example .env
   ```

3. Start the whole demo:

   ```bash
   docker compose up --build
   ```

Open `http://localhost:8000`. The API docs are at `/docs`.

This Compose setup intentionally binds both the API and PostgreSQL to `127.0.0.1`: it is a local demo, not an authenticated multi-user deployment. The PostgreSQL binding lets `uv run python clear_db.py --yes` connect from your host. Add authentication and pagination before exposing it on a network.

For local backend work, start PostgreSQL with Compose, set `DATABASE_URL` to its exposed host address, and run `uv run uvicorn app.main:app --reload`. For frontend hot reload, run `npm run dev` in `frontend/`; Vite proxies API requests to port 8000.

## The flow to explain

```text
React UI → FastAPI route → create_agent (LangGraph) → Vertex AI
                 │                  │
                 └── SQLModel ──────┴── AsyncPostgresSaver
                     conversation        messages + checkpoints + interrupts
                     metadata only
```

The `thread_id` deliberately lives in LangGraph config, not mutable graph state. State is loaded only after LangGraph knows which durable thread to retrieve. A UUID is created once, persisted as app metadata, returned to the browser, and reused for every message and resume.

## API surface

| Route | Purpose |
| --- | --- |
| `POST /threads` | Create and return a UUID-backed conversation. |
| `GET /threads` | List saved conversation metadata. |
| `GET /threads/{thread_id}` | Load that thread's persisted messages and pending interrupt. |
| `POST /threads/{thread_id}/messages` | Add a turn and run the graph. |
| `POST /threads/{thread_id}/resume` | Continue a paused graph with `{ "approved": true | false }`. |

## Pattern cheat sheet

The running demo uses `ToolStrategy(FinalResponse)` plus an interrupting approval tool. That makes every normal answer a validated tool call. Keep the following patterns separate; combining a forced ordinary tool with a structured-final tool makes two tools compete for the same model turn.

### 1. End immediately from a tool

```python
from langchain.tools import tool

@tool(return_direct=True)
def lookup_status(job_id: str) -> str:
    """Return the status of one job."""
    return "complete"
```

`return_direct=True` ends the agent run with the tool result. It is useful when a raw tool result is the HTTP response, but it intentionally bypasses a structured final response.

### 2. Return validated structured output

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class FinalResponse(BaseModel):
    answer: str
    confidence: float

agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(FinalResponse),
    checkpointer=checkpointer,
)
result = await agent.ainvoke({"messages": [("user", "Summarize this")]}, config)
final = result["structured_response"]
```

`ToolStrategy` uses tool calling, validates the schema, and returns `structured_response`. This is the pattern used by `app/langgraph.py`: every non-interrupted turn ends via `FinalResponse`, so the API reads its `answer` field predictably.

### 3. Require a tool call

```python
required_model = model.bind_tools(tools, tool_choice="required")
tool_only_agent = create_agent(required_model, tools=tools, checkpointer=checkpointer)
```

With Gemini, `tool_choice="required"` (or `"any"`) makes the model choose a tool. Use this as a dedicated mode, not beside the structured-final pattern above.

### 4. Pause and resume safely

```python
from langchain.tools import tool
from langgraph.types import Command, interrupt
from pydantic import BaseModel, StrictBool

class InterruptBoolValidator(BaseModel):
    action: str
    approved: StrictBool

@tool
def request_approval(action: str) -> str:
    approved = interrupt({"kind": "approval", "action": action})
    return InterruptBoolValidator(action=action, approved=approved).model_dump_json()

config = {"configurable": {"thread_id": thread_id}}
await agent.ainvoke(Command(resume=True), config)
```

The action is supplied dynamically by the model; the resume value is strict boolean. The application saves both on the `Conversation` row before resuming, so the selected decision remains visible after refresh. An interrupted node restarts from its beginning on resume. Keep work before `interrupt()` free of side effects, or make it idempotent.

## LangSmith traces

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in `.env`. LangChain automatically emits traces to that project—no tracing code or wrapper is needed. Do not commit `.env`.

## Voice input and audio uploads

**Speak** uses the browser's Web Speech API: this app receives only the transcript as chat text and does not upload or store audio. Browser implementations may use vendor speech services, and support varies (Chrome/Edge work best).

Sending an audio file to the model is a separate, small feature: add a multipart upload route, store/stream the bytes, then use a model/provider with audio input (or transcribe server-side first). For this concise chat POC I would keep the current browser transcription; it has no storage, upload, or audio-provider concerns.

## Gemini on Vertex vs. literal Model Garden

The default `VERTEX_MODEL=gemini-2.5-flash` uses Gemini through Vertex AI and ADC, the current reliable choice for async chat, tool calling, and structured output. Vertex AI Model Garden also hosts deployed/open-model endpoints. Those are model-specific: do not assume a generic endpoint supports `bind_tools` or `ToolStrategy`.

To experiment with a deployed Model Garden endpoint, install the optional integration and keep it as a separate provider experiment:

```bash
uv sync --extra model-garden
```

```python
from langchain_google_vertexai import VertexAIModelGarden

model = VertexAIModelGarden(project="your-project", endpoint_id="your-endpoint")
```

Smoke-test the chosen model's tool-calling ability before using it with `create_agent`.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run pyright app tests
```

The focused tests verify final-response/interrupt mapping, restored structured history, and custom interrupt decisions. They intentionally do not mock Vertex AI or duplicate LangGraph's own persistence tests.

Manual Compose smoke test: start a conversation by sending a message (its title is the first message plus the local time), ask for the UTC time, then ask the agent to approve a deployment. Use **Approve** or **Reject**, refresh the page, and select the same conversation; its checkpointed history should still be present.

To reset this local demo, run the intentionally-confirmed script below. It truncates only `conversation` and the LangGraph checkpoint tables in the database configured by `DATABASE_URL`; it does not remove the Docker volume or database schema. It intentionally fails if those tables have external dependencies, rather than cascading into other application data.

```bash
uv run python clear_db.py --yes
```

Run it from your host terminal; there is no need to enter the API container. The project installs SQLAlchemy's `asyncio` extra (including `greenlet`) for this local command.

## Sources

- [LangGraph persistence and thread memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph interrupts and resume](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangChain structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Tailwind CSS with Vite](https://tailwindcss.com/docs/installation/using-vite)
- [LangSmith environment configuration](https://docs.langchain.com/langsmith/smith-python-sdk)
- [Google's current Gemini-on-Vertex integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
- [Google Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc)
