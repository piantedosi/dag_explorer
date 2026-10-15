# dag-explorer

Discover the arrows of a hidden causal DAG. Node values update from their parents; you guess the directed edges, and the program tells you how many true edges are still missing and how many guessed edges are wrong.

Two ways to play:

- `dag_explorer.py` is a pygame window. Drag between nodes to add or remove a hypothesized edge, and freeze a node to a chosen value.
- `agent.py` is a LangGraph agent with the same actions as tools: `show_status`, `draw_edge`, `freeze_node`, and `unfreeze_node`.

The causal model lives in `dag.py`.

## Setup

Python 3.12. Dependencies are listed in `pyproject.toml` and locked in `uv.lock`.

```bash
uv sync
```

## Play the game

```bash
uv run python dag_explorer.py
```

The setup screen sets the node count (2–26) and an easy mode. In the graph, drag from one node to another to hypothesize an edge, click a node to freeze or unfreeze it, and watch the bar for missing and wrong edges.

## Run the agent

The agent calls a Hugging Face inference endpoint (`Qwen/Qwen3.8-27B`). Log in once so `huggingface_hub` can authenticate:

```bash
uv run hf auth login
uv run python agent.py
```

The script sends “Start the exploration.” and prints the model’s tool calls and final reply.

## Langfuse

Traces are optional. With project keys in the root `.env`, `agent.py` sends each run to a local Langfuse instance.

```bash
cp .env.example .env
# fill LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY from
# http://localhost:3000 -> Project Settings -> API Keys
```

To start a fresh local stack, copy `langfuse/.env.example` to `langfuse/.env`, set the secrets (including `ENCRYPTION_KEY` from `openssl rand -hex 32`), then:

```bash
docker compose --env-file langfuse/.env -f langfuse/docker-compose.yml up
```

The UI is at http://localhost:3000. A stack already running on that port ignores `langfuse/.env`.
