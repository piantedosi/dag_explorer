import inspect
import os
import random
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from dag import Dag, edge_report


##### Setup #####
NODE_COUNT = 4
EASY = False


def _load_dotenv() -> None:
    """Load a project-root .env without overriding variables already set."""
    path = Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

dag = Dag.generate(random.Random(0), NODE_COUNT)


##### Tools #####

def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def _default_values() -> dict[str, float | None]:
    """Evaluate with freezes cleared, then restore them."""
    saved = dict(dag.hypothesis().frozen)
    try:
        for name in saved:
            dag.force_node(name, None)
        return dag.evaluate()
    finally:
        for name, value in saved.items():
            dag.force_node(name, value)


def _status() -> str:
    names = [node.name for node in dag.nodes]
    hypothesized = set(dag.hypothesis().edges)
    frozen = dag.hypothesis().frozen
    values = dag.evaluate()
    defaults = _default_values()
    header = "\t".join(["", *names, "value"])
    rows = [header]
    for src, src_name in enumerate(names):
        cells = [
            "1" if (src, dst) in hypothesized else "0" for dst in range(len(names))
        ]
        shown = _fmt(values[src_name])
        marks = []
        if src_name in frozen:
            marks.append("F")
        elif shown != _fmt(defaults[src_name]):
            marks.append("C")
        if marks:
            shown = f"{shown} {' '.join(marks)}"
        rows.append("\t".join([src_name, *cells, shown]))
    missing, wrong = dag.edge_counts()
    if EASY:
        rows.append(edge_report(missing, wrong))
    elif missing == 0 and wrong == 0:
        rows.append("CORRECT GRAPH :)")
    return "\n".join(rows)


def draw_edge(src: str, dst: str) -> str:
    """Add a hypothesized edge from one node to another.

    Args:
        src: Name of the source node.
        dst: Name of the target node.

    Returns:
        Whether the edge was added, already present, or ignored.
    """
    by_name = {node.name: node.index for node in dag.nodes}
    src_i = by_name.get(src)
    dst_i = by_name.get(dst)
    if src_i is None or dst_i is None:
        return f"ignored edge {src} -> {dst}."
    before = set(dag.hypothesis().edges)
    dag.add_edge(src_i, dst_i)
    if (src_i, dst_i) in before:
        return f"edge {src} -> {dst} already present."
    if (src_i, dst_i) not in set(dag.hypothesis().edges):
        return f"ignored edge {src} -> {dst}."
    return f"added edge {src} -> {dst}."


def freeze_node(name: str, value: float) -> str:
    """Freeze a node to a numeric value.

    Args:
        name: Name of the node to freeze.
        value: Finite value to assign.

    Returns:
        Confirmation, or a message that the node could not be frozen.
    """
    dag.force_node(name, value)
    if name not in dag.hypothesis().frozen:
        return f"could not freeze {name!r}."
    return f"froze {name} at {dag.hypothesis().frozen[name]}."


def unfreeze_node(name: str) -> str:
    """Clear the freeze on a node.

    Args:
        name: Name of the node to unfreeze.

    Returns:
        Confirmation that the freeze was cleared.
    """
    dag.force_node(name, None)
    return f"unfroze {name}."


def _show_status_doc() -> str:
    """Docstring for show_status. Hard mode hides the edge counts."""
    feedback = (
        "The last line reports how many true edges are missing and how many guessed edges are wrong."
        if EASY
        else "The last line is CORRECT GRAPH :) only when the hypothesized edges match the hidden graph."
    )
    example_tail = (
        "\n            there are 4 missing edges and 0 wrong edges"
        if EASY
        else ""
    )
    return f"""Show the hypothesized adjacency matrix and each node value.

    Rows and columns are node names. A cell is 1 when that edge is hypothesized and 0 otherwise.
    Values use two decimal places. F marks a frozen node. C marks a node that is not frozen but whose shown value differs from the unfrozen default.
    {feedback}

    Returns:
        A tab-separated matrix. in this format:
        ```
                    node1   node2   node3   node4   value   
            node1   0       1       0       0       1.50 F
            node2   0       0       0       0       0.48 C
            node3   0       0       0       0       10.44 C
            node4   0       0       0       0       1.15 C{example_tail}
        ```
    """


def show_status() -> str:
    """Show the hypothesized adjacency matrix and each node value."""
    return _status()


show_status.__doc__ = _show_status_doc()


def finish(note: str) -> str:
    """Stop the exploration.

    Call this only when you are ready to stop, instead of answering in plain text.
    This does not change the hypothesized graph.

    Args:
        note: Short summary of the edges you are committing to.

    Returns:
        Confirmation that the exploration has finished.
    """
    return f"finished: {note}"


tools = [
    draw_edge,
    freeze_node,
    unfreeze_node,
    show_status,
    finish,
]


def tool_prompt() -> str:
    """Format each tool signature and docstring for the system prompt."""
    blocks = []
    for fn in tools:
        doc = inspect.cleandoc(fn.__doc__ or "")
        blocks.append(f"{fn.__name__}{inspect.signature(fn)}\n{doc}")
    return "\n\n".join(blocks)


##### Model #####

_llm_with_tools = None
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _reasoning_text(message) -> str:
    """Read a provider reasoning field. ChatHuggingFace does not copy it."""
    if not hasattr(message, "get"):
        return ""
    raw = message.get("reasoning") or message.get("reasoning_content") or ""
    return raw.strip() if isinstance(raw, str) else ""


def _split_think(content: str) -> tuple[str, str]:
    parts = [part.strip() for part in _THINK_RE.findall(content) if part.strip()]
    if not parts:
        return content, ""
    visible = _THINK_RE.sub("", content).strip()
    return visible, "\n\n".join(parts)


def _attach_reasoning(result, response) -> None:
    """Keep the thinking trace on the assistant message.

    The router returns it as `reasoning` (or `reasoning_content`). Langfuse stores
    `additional_kwargs`, and the shell prints `reasoning_content` from there.
    The text stays off `content` when the provider already split it, so later
    turns do not resend the trace.
    """
    choices = response.get("choices", []) if hasattr(response, "get") else []
    for generation, choice in zip(result.generations, choices):
        message = generation.message
        if not isinstance(message, AIMessage):
            continue
        raw_message = choice.get("message") if hasattr(choice, "get") else None
        reasoning = _reasoning_text(raw_message)
        if isinstance(message.content, str) and _THINK_RE.search(message.content):
            visible, tagged = _split_think(message.content)
            message.content = visible
            reasoning = reasoning or tagged
        if reasoning:
            message.additional_kwargs["reasoning_content"] = reasoning


class _ReasoningChat(ChatHuggingFace):
    def _create_chat_result(self, response):
        result = super()._create_chat_result(response)
        _attach_reasoning(result, response)
        return result


def get_llm_with_tools():
    """Build the llm with tools on first use so tool tests do not log in."""
    global _llm_with_tools
    if _llm_with_tools is None:
        from huggingface_hub import login

        login()
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen3.8-27B",
            task="text-generation",
            temperature=0.1,
            max_new_tokens=8000,
        )
        chat = _ReasoningChat(llm=llm_endpoint)
        _llm_with_tools = chat.bind_tools(tools)
    return _llm_with_tools


##### Graph #####

_EMPTY_NUDGE = (
    "Thought too long and nothing was decided. Think for less time, then call one tool."
)
_MAX_NUDGES = 3


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    loop_count: int


def assistant(state: AgentState):
    if EASY:
        feedback = "You will be able to see the number of missing and wrong edges."
    else:
        feedback = (
            "show_status ends with CORRECT GRAPH :) only when the hypothesized graph is correct."
        )
    sys_msg = SystemMessage(
        content="\n".join(
            [
                "You will try to discover the directed edges of a hidden causal DAG.",
                "You can create counterfactuals by freezing nodes to new values.",
                feedback,
                "You will have access to the following tools:",
                "Tools:",
                tool_prompt(),
                "",
                "Call show_status to read the matrix. Do not invent node names or values.",
                "Do not stop by writing a summary. Call finish when the exploration is done.",
            ]
        )
    )

    message = get_llm_with_tools().invoke([sys_msg] + state["messages"])
    if getattr(message, "tool_calls", None):
        return {"messages": [message], "loop_count": 0}
    count = state.get("loop_count", 0) + 1
    if count >= _MAX_NUDGES:
        return {
            "messages": [message, AIMessage(content=_status())],
            "loop_count": count,
        }
    return {"messages": [message], "loop_count": count}


def nudge(state: AgentState) -> dict:
    """Append the hint that asks for one tool call."""
    return {"messages": [HumanMessage(content=_EMPTY_NUDGE)]}


def _route_after_assistant(state: AgentState) -> str:
    """Send a tool call to tools, an empty reply to nudge, and the third empty reply to the end."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_call"
    if not isinstance(last, AIMessage):
        raise RuntimeError(
            f"assistant left {type(last).__name__}, expected a tool call or an assistant message"
        )
    if state.get("loop_count", 0) >= _MAX_NUDGES:
        return "gave_up"
    return "empty_reply"


def _route_after_tools(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            if any(call["name"] == "finish" for call in message.tool_calls):
                return "finish"
            return "continue"
    return "continue"


def _retry_rate_limit(exc: Exception) -> bool:
    """Retry only a Hugging Face 429. The hub error subclasses OSError, which the default policy skips."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def build_graph():
    """Compile the assistant/tools loop once."""
    builder = StateGraph(AgentState)
    builder.add_node(
        "assistant",
        assistant,
        retry_policy=RetryPolicy(
            initial_interval=5.0,
            backoff_factor=2.0,
            max_interval=60.0,
            max_attempts=6,
            retry_on=_retry_rate_limit,
        ),
    )
    builder.add_node("nudge", nudge)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_edge(START, "assistant")
    builder.add_edge("nudge", "assistant")
    builder.add_conditional_edges(
        "assistant",
        _route_after_assistant,
        {"tool_call": "tools", "empty_reply": "nudge", "gave_up": END},
    )
    builder.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"continue": "assistant", "finish": END},
    )
    return builder.compile()


##### Run #####

def _format_message(message) -> str:
    reasoning = ""
    if isinstance(message, AIMessage):
        raw = message.additional_kwargs.get("reasoning_content") or ""
        reasoning = raw if isinstance(raw, str) else ""
    text = message.pretty_repr()
    if reasoning:
        head, _, body = text.partition("\n\n")
        text = f"{head}\n\nThinking:\n{reasoning}"
        if body:
            text = f"{text}\n\n{body}"
    return text


def _print_message(message, log_file=None) -> None:
    text = _format_message(message)
    print(text)
    if log_file is not None:
        print(text, file=log_file)


def _open_trace_log():
    """Create logs/agent-YYYYMMDD-HHMMSS.log for this run."""
    directory = Path(__file__).resolve().parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"agent-{stamp}.log"
    suffix = 1
    while path.exists():
        path = directory / f"agent-{stamp}-{suffix}.log"
        suffix += 1
    return path.open("w", encoding="utf-8"), path


def _write_graph_png(graph) -> None:
    graph_png = Path(__file__).resolve().parent / "react_graph.png"
    graph.get_graph().draw_mermaid_png(output_file_path=str(graph_png))
    print(graph_png)


def _langfuse_handler():
    """Return a Langfuse callback when local project keys are configured."""
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not public.startswith("pk-lf-") or not secret.startswith("sk-lf-"):
        return None
    if "..." in public or "..." in secret:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


class Agent:
    def __init__(self):
        self.graph = build_graph()

    def invoke(self, input_message):
        log_file, log_path = _open_trace_log()
        print(f"trace written to {log_path.relative_to(Path(__file__).resolve().parent)}")
        try:
            handler = _langfuse_handler()
            payload = {
                "messages": [HumanMessage(content=input_message)],
                "loop_count": 0,
            }
            config = {"recursion_limit": 80}
            if handler is not None:
                config["callbacks"] = [handler]
                config["run_name"] = "dag-agent"
            if handler is None:
                trace = nullcontext()
            else:
                from langfuse import get_client, propagate_attributes

                trace = propagate_attributes(trace_name="dag-explorer", tags=["langgraph"])
            with trace:
                result = self.graph.invoke(payload, config=config)
            if handler is not None:
                get_client().flush()
            for m in result.get("messages", []):
                _print_message(m, log_file)
            last_msg = result["messages"][-1]
            if hasattr(last_msg, "content"):
                return str(last_msg.content)
            return str(last_msg)
        finally:
            log_file.close()

    def __call__(self, input_message):
        return self.invoke(input_message)


if __name__ == "__main__":
    agent = Agent()
    _write_graph_png(agent.graph)
    print(agent.invoke("Start the exploration."))
