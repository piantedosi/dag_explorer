"""A causal DAG with fixed base values and one evaluation method."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

EXOGENOUS_LOW = 0.6
EXOGENOUS_HIGH = 2.2
PROBE_DELTA = 0.5
PROBE_MIN = 0.05
VALUE_CAP = 40.0
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"


@dataclass(frozen=True)
class Term:
    op: str
    parents: tuple[int, ...]
    coef: float = 1.0


@dataclass
class Node:
    index: int
    name: str
    parents: tuple[int, ...]
    terms: tuple[Term, ...]
    base: float


@dataclass(frozen=True)
class Hypothesis:
    edges: tuple[tuple[int, int], ...]
    frozen: dict[str, float]


class Dag:
    def __init__(self, nodes: list[Node], edges: list[tuple[int, int]]) -> None:
        self.nodes = nodes
        self.edges = edges
        self._hypothesis_edges: set[tuple[int, int]] = set()
        self._frozen: dict[str, float] = {}

    @classmethod
    def generate(cls, rng: random.Random, count: int) -> Dag:
        if not 2 <= count <= 26:
            raise ValueError("node count must be from 2 to 26")
        for _ in range(400):
            dag = _try_generate(rng, count)
            if dag is not None and _is_directed_acyclic(dag):
                return dag
        raise RuntimeError("could not generate a causal DAG")

    def hypothesis(self) -> Hypothesis:
        """Return a copy of the current guessed edges and frozen nodes."""
        return Hypothesis(tuple(self._hypothesis_edges), dict(self._frozen))

    def add_edge(self, src: int, dst: int) -> None:
        if src == dst:
            return
        if not 0 <= src < len(self.nodes) or not 0 <= dst < len(self.nodes):
            return
        self._hypothesis_edges.add((src, dst))

    def remove_edge(self, src: int, dst: int) -> None:
        self._hypothesis_edges.discard((src, dst))

    def force_node(self, name: str, value: float | None) -> None:
        """Store a freeze, or clear it when value is None.

        Unknown names and non-finite values are ignored.
        """
        node = next((node for node in self.nodes if node.name == name), None)
        if node is None:
            return
        if value is None:
            self._frozen.pop(name, None)
            return
        if not math.isfinite(value):
            return
        self._frozen[name] = value

    def edge_counts(self) -> tuple[int, int]:
        """Return (missing, wrong) against the true edges."""
        true = set(self.edges)
        guessed = set(self.hypothesis().edges)
        return len(true - guessed), len(guessed - true)

    def evaluate(self) -> dict[str, float | None]:
        """Return every node's value under the stored freezes.

        An empty freeze map is the base state.
        """
        by_name = {node.name: node for node in self.nodes}
        frozen_index: dict[int, float] = {}
        for name, value in self._frozen.items():
            node = by_name.get(name)
            if node is None or not math.isfinite(value):
                continue
            frozen_index[node.index] = value
        values = _values(self.nodes, frozen_index)
        return {node.name: values[node.index] for node in self.nodes}


def edge_report(missing: int, wrong: int) -> str:
    """Sentence for the missing-edge and wrong-edge counts."""

    def phrase(count: int, kind: str) -> str:
        noun = kind if count == 1 else f"{kind}s"
        return f"{count} {noun}"

    return f"there are {phrase(missing, 'missing edge')} and {phrase(wrong, 'wrong edge')}"


def _eval_term(term: Term, values: dict[int, float]) -> float | None:
    try:
        if term.op == "lin":
            result = term.coef * values[term.parents[0]]
        else:
            total = sum(values[parent] for parent in term.parents)
            if term.op == "sin":
                result = math.sin(total)
            elif term.op == "cos":
                result = math.cos(total)
            elif term.op == "exp":
                if total > 20:
                    return None
                result = math.exp(total)
            elif term.op == "log":
                if total <= 0:
                    return None
                result = math.log(total)
            else:
                return None
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result) or abs(result) > 1e6:
        return None
    return result


def _eval_terms(terms: tuple[Term, ...], values: dict[int, float]) -> float | None:
    total = 0.0
    for term in terms:
        part = _eval_term(term, values)
        if part is None:
            return None
        total += part
    if not math.isfinite(total) or abs(total) > 1e6:
        return None
    return total


def _values(nodes: list[Node], frozen: dict[int, float]) -> list[float | None]:
    values: list[float | None] = [None] * len(nodes)
    for node in nodes:
        if node.index in frozen:
            values[node.index] = frozen[node.index]
            continue
        if not node.parents:
            values[node.index] = node.base
            continue
        if any(values[parent] is None for parent in node.parents):
            values[node.index] = None
            continue
        parent_values = {parent: values[parent] for parent in node.parents}
        values[node.index] = _eval_terms(node.terms, parent_values)
        if values[node.index] is not None and abs(values[node.index]) > 1e6:
            values[node.index] = None
    return values


def _random_terms(rng: random.Random, parents: tuple[int, ...]) -> tuple[Term, ...] | None:
    parent_list = list(parents)
    single_ops = ("lin", "sin", "cos", "exp", "log")
    target = rng.randint(1, 3)
    terms: list[Term] = []
    used: set[int] = set()
    for _ in range(12):
        if len(terms) >= target and used == set(parent_list):
            break
        if len(terms) >= 3:
            break
        if len(parent_list) >= 2 and rng.random() < 0.45:
            count = rng.randint(2, len(parent_list))
            subset = tuple(rng.sample(parent_list, count))
            terms.append(Term(rng.choice(("sin", "cos", "exp", "log")), subset))
            used.update(subset)
            continue
        parent = rng.choice(parent_list)
        op = rng.choice(single_ops)
        if op == "lin":
            terms.append(Term("lin", (parent,), float(rng.choice((-2, -1, 1, 2)))))
        else:
            terms.append(Term(op, (parent,)))
        used.add(parent)
    for parent in parent_list:
        if parent not in used and len(terms) < 3:
            terms.append(Term("lin", (parent,), float(rng.choice((-2, -1, 1, 2)))))
            used.add(parent)
    if used != set(parent_list) or not 1 <= len(terms) <= 3:
        return None
    return tuple(terms)


def _equation_ok(terms: tuple[Term, ...], parent_values: dict[int, float]) -> bool:
    base = _eval_terms(terms, parent_values)
    if base is None or abs(base) > 20:
        return False
    for parent in parent_values:
        moves: list[float] = []
        for delta in (PROBE_DELTA, -PROBE_DELTA):
            bumped = dict(parent_values)
            bumped[parent] = parent_values[parent] + delta
            alt = _eval_terms(terms, bumped)
            if alt is None or abs(alt) > VALUE_CAP:
                return False
            moves.append(abs(alt - base))
        if max(moves) < PROBE_MIN:
            return False
    return True


def _base_span_ok(nodes: list[Node]) -> bool:
    root = next(node for node in nodes if not node.parents)
    saved = root.base
    try:
        for base in (0.6, 1.0, 1.4, 1.8, 2.2):
            root.base = base
            values = _values(nodes, {})
            if any(value is None or abs(value) > VALUE_CAP for value in values):
                return False
        return True
    finally:
        root.base = saved


def _is_directed_acyclic(dag: Dag) -> bool:
    count = len(dag.nodes)
    by_index = {node.index: node for node in dag.nodes}
    seen: set[tuple[int, int]] = set()
    for source, target in dag.edges:
        if source == target or source not in by_index or target not in by_index:
            return False
        if (source, target) in seen or (target, source) in seen:
            return False
        seen.add((source, target))
        node = by_index[target]
        if source not in node.parents:
            return False
        if not any(source in term.parents for term in node.terms):
            return False
    parent_edges = {(parent, node.index) for node in dag.nodes for parent in node.parents}
    if seen != parent_edges:
        return False
    children: dict[int, list[int]] = {node.index: [] for node in dag.nodes}
    for source, target in dag.edges:
        children[source].append(target)
    color = {node.index: 0 for node in dag.nodes}

    def visit(node_index: int) -> bool:
        color[node_index] = 1
        for child in children[node_index]:
            if color[child] == 1:
                return False
            if color[child] == 0 and not visit(child):
                return False
        color[node_index] = 2
        return True

    return all(color[node.index] != 0 or visit(node.index) for node in dag.nodes) and count == len(by_index)


def _word_name(rng: random.Random, used: set[str]) -> str:
    for _ in range(100):
        name = (
            rng.choice(CONSONANTS)
            + rng.choice(VOWELS)
            + rng.choice(CONSONANTS)
            + rng.choice(VOWELS)
        )
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError("could not generate a unique name")


def _try_generate(rng: random.Random, count: int) -> Dag | None:
    used_names: set[str] = set()
    nodes: list[Node] = [
        Node(0, _word_name(rng, used_names), (), (), rng.uniform(EXOGENOUS_LOW, EXOGENOUS_HIGH))
    ]
    edges: list[tuple[int, int]] = []
    for index in range(1, count):
        parent_count = rng.randint(1, min(2, index))
        parents = tuple(sorted(rng.sample(range(index), parent_count)))
        base_values = _values(nodes, {})
        parent_values: dict[int, float] = {}
        for parent in parents:
            parent_value = base_values[parent]
            if parent_value is None:
                return None
            parent_values[parent] = parent_value
        terms = None
        for _ in range(60):
            candidate = _random_terms(rng, parents)
            if candidate is not None and _equation_ok(candidate, parent_values):
                terms = candidate
                break
        if terms is None:
            return None
        nodes.append(Node(index, _word_name(rng, used_names), parents, terms, 0.0))
        edges.extend((parent, index) for parent in parents)
    needed = count - 1 if count < 4 else max(3, count - 2)
    if len(edges) < needed:
        return None
    if not _base_span_ok(nodes):
        return None
    if any(value is None for value in _values(nodes, {})):
        return None
    return Dag(nodes, edges)
