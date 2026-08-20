# StoreMate: Deep Agents + Docker POC

A deliberately compact retail-sales demo: FastAPI, `create_deep_agent`, Gemini
on Vertex AI, PostgreSQL persistence, and a small React/Tailwind chat UI.

## Why this shape

PostgreSQL is both the LangGraph checkpointer and the cross-session store. It
keeps the demo to two containers while providing durable conversation threads,
interrupts, skills, and per-user memory in the `postgres_data` Docker volume.
MongoDB would be reasonable if it were already the application database, but
adds no benefit to this relational, thread-listing POC.

```text
React → FastAPI → Deep Agent → Gemini on Vertex AI
                 │      ├── AsyncPostgresSaver (per-thread state + interrupts)
                 │      └── AsyncPostgresStore (skills + cross-session memory)
                 └── SQLModel (conversations, products, orders, order items)
```

`thread_id` is an opaque UUID in LangGraph config. `LOCAL_USER_ID` is a local
demo identity used to namespace customer data; replace it with an authenticated
user ID in a real app.

## Retail data and tools

Startup seeds three demo products into `products` (SKU, name, description,
price, and stock). `orders` and `order_items` record completed purchases with
the runtime `user_id`.

`find_articles` uses SQLModel to list the catalog or filter it by a
case-insensitive name match. It returns the agent enough internal data to make a sale, but the prompt
and tool description tell it to show customers only product name, description,
and price. It should reveal an exact inventory amount only when a customer
explicitly asks for the purchase limit.

`create_order` pauses for customer approval, then decrements inventory and
inserts an order in one ORM transaction. If the requested quantity is too high,
it returns “There is not enough inventory for that purchase” without exposing
the remaining count. `get_my_recent_orders` stays scoped to the runtime user.

## Layout

```text
app/settings.py       environment configuration
app/db.py             SQLModel conversation metadata and async sessions
app/bootstrap.py       database, checkpointer/store, and agent wiring for the lifespan
app/agent/prompt.py    the Deep Agent's system prompt
app/agent/tools/       small, independently understandable agent tools
app/agent/graph.py     Deep Agent, durable backend, skills
app/agent/responses.py adapters between LangGraph values and the API response models
app/main.py            FastAPI routes
skills/*/SKILL.md      local instructions exposed to the agent
frontend/              one responsive React/Tailwind chat page
```

There are deliberately no service or repository layers.

## Run

1. Authenticate locally with Vertex AI ADC and enable the API:

   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud auth application-default login
   cp .env.example .env
   ```

2. Set `GOOGLE_CLOUD_PROJECT` and the absolute `GCP_ADC_PATH` in `.env`, then:

   ```bash
   docker compose up --build
   ```

Open `http://localhost:8000`; API documentation is at `/docs`.

The Compose ports bind to `127.0.0.1`. PostgreSQL is therefore reachable from
your host by `clear_db.py`; there is no need to enter the API container.

This POC deliberately has no migration framework or Docker initialization
scripts. At API startup:

1. `SQLModel.metadata.create_all()` creates missing application tables only. It
   never alters existing tables or columns; change the schema manually (or add
   Alembic) when the model changes.
2. `seed_catalog()` upserts the three demo products by SKU. It refreshes their
   name, description, price, and reset-stock value, but preserves current stock
   so a completed sale survives an API restart.
3. `clear_db.py --yes` clears orders and restores each product's current stock
   from its seeded reset-stock value.

Restart or rebuild the API after changing `CATALOG`; the catalog fields above
will update without deleting conversations or orders. Reset the local data only
when you want a completely fresh demo.

## Deep Agent pieces to explain

The app uses `create_deep_agent`, which adds an agent harness (planning and a
filesystem-style backend) on top of LangGraph. Its backend routes are small:

- scratch files: thread-scoped `StateBackend()`;
- `/skills/`: read-only, durable SKILL.md files shared by the local agent;
- `/memories/`: durable PostgreSQL data scoped to `LOCAL_USER_ID`, usable in a
  new conversation after restart.

The selling flow is concrete: search a product by name, answer with its live
price, confirm the product and quantity, then call `create_order`.

`skills/retail-sales/SKILL.md` is seeded into the durable store at API
startup. Add another directory containing `SKILL.md`, rebuild/restart, then
start a new conversation: a thread snapshots its skill metadata when it is
created. The agent is denied writes to `/skills/`.

`StoreBackend` only makes `/memories/` durable; it does not decide what is worth
remembering. StoreMate gives the AI two store-backed tools instead:
`save_customer_memory` and `read_customer_memory`. The system prompt and retail
skill direct the agent to save an explicitly stated, non-sensitive customer fact
before answering, and to read memory before answering a personal question. Thus
“my name is Justin” can be saved in one chat and retrieved in another without
hardcoded phrase matching. The model is responsible for choosing the tools, so
it remains an agentic workflow rather than API-side extraction.

The exact structured-output pattern remains supported by Deep Agents:

```python
agent = create_deep_agent(
    model=model,
    tools=[current_utc_time, read_customer_memory, save_customer_memory, *retail_tools(session_factory)],
    backend=backend,
    skills=["/skills/"],
    response_format=ToolStrategy(FinalResponse),
    checkpointer=checkpointer,
    store=store,
)
```

Every completed normal turn yields `FinalResponse(answer: str)`, so FastAPI can
return a predictable `reply`. Keep this separate from an end tool:

```python
@tool(return_direct=True)
def status(job_id: str) -> str:
    return "complete"
```

`return_direct=True` intentionally ends the turn with raw tool output. It is
useful for a response tool, but it bypasses the structured final-answer tool.

`create_order` calls `interrupt()` with its custom action. `resume_thread`
records the customer's decision as a `HumanMessage` via `agent.aupdate_state`
— the same durable message history everything else lives in, not a separate
database column that could disagree with it — then resumes the same
`thread_id` with `Command(resume=...)`. Keep work before `interrupt()`
idempotent because the interrupted node executes again on resume.

## API

| Route | Purpose |
| --- | --- |
| `POST /threads` | Create a UUID-backed conversation. |
| `GET /threads` | List conversation metadata. |
| `GET /threads/{thread_id}` | Load one selected thread and its pending interrupt. |
| `POST /threads/{thread_id}/messages` | Add a turn. |
| `POST /threads/{thread_id}/resume` | Resume with `{ "approved": true }` or `false`. |

The UI only fetches the selected conversation; threads are sorted by creation
time and a new title displays its first message, then its creation date and
hour on a separate line.

## Speech input

The UI has an explicit **Español** (`es-MX`) / **English** (`en-US`) selector.
It invokes the browser's Web Speech API (`SpeechRecognition` or Chrome's
`webkitSpeechRecognition`) and sends only the transcript to FastAPI.

This app does **not** bundle, call, or choose a speech-to-text model. Chrome
provides the recognition implementation and may use its own services; exact
availability and processing depend on the browser and device. Chrome and Edge
are the most practical choices for this demo. Audio file uploads would be a
different feature: receive multipart audio and transcribe it server-side or
use a multimodal model.

## LangSmith

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in
`.env`. LangChain automatically emits traces; no application wrapper is needed.

## Reset local data

With Compose running, this clears conversations, checkpoints, local customer
memories, and orders, then restores catalog inventory. It does not delete the
Docker volume, schema, or repository skills:

```bash
uv run python clear_db.py --yes
```

Do not run it while the API is handling requests. The script intentionally does
not use `CASCADE`, so it fails rather than unexpectedly clearing unrelated
application data.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run pyright app tests
```

## Sources

- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
- [Deep Agent backends and memory](https://docs.langchain.com/oss/python/deepagents/backends)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Google Gemini on Vertex AI](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
