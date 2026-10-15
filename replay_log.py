"""Replay an agent log as a GIF: the game board on top, the trace underneath."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pygame
from PIL import Image

from dag_explorer import (
    ARROW,
    ARROW_WIN,
    BAR,
    BAR_H,
    BG,
    HEIGHT,
    INTERVENE_RING,
    MUTED,
    NODE,
    NODE_RING,
    TEXT,
    WIDTH,
    blit_fitted,
    choose_radius,
    draw_arrow,
    edge_segment,
    fmt_value,
    layout_positions,
    load_font,
)

ROOT = Path(__file__).resolve().parent
PANEL_H = 200
CHARS_PER_FRAME = 4
TYPE_MS = 40
HOLD_MS = 800
END_HOLD_MS = 1600
HEAD_MS = 20_000
TAIL_MS = 30_000
SKIP_MS = 1200

CALL_COLOR = (176, 206, 255)
RESULT_COLOR = (150, 196, 170)

_BANNER = re.compile(r"^=+\s*(Human|Ai|Tool) Message\s*=+\s*$")
_CALL = re.compile(r"^(\w+)\s+\(([^)]*)\)\s*$")
_ARG = re.compile(r"^([^:]+):\s?(.*)$")
_REPORT = re.compile(r"^there are (\d+) missing edges? and (\d+) wrong edges?$")
_WIN = "CORRECT GRAPH :)"
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?$")


@dataclass
class Matrix:
    names: list[str]
    edges: list[tuple[int, int]]
    values: dict[str, float | None]
    frozen: set[str]
    report: str
    won: bool


@dataclass
class TextSegment:
    role: str
    text: str


@dataclass
class MatrixSegment:
    matrix: Matrix


Segment = TextSegment | MatrixSegment


@dataclass
class _Layout:
    names: tuple[str, ...] = ()
    positions: list[tuple[float, float]] = field(default_factory=list)
    radius: int = 0

    def for_names(self, names: list[str]) -> tuple[list[tuple[float, float]], int]:
        key = tuple(names)
        if key != self.names:
            count = len(names)
            if count <= 1:
                self.radius = 34
                mid_y = BAR_H + (HEIGHT - BAR_H) / 2
                self.positions = [(WIDTH / 2, mid_y)] if count == 1 else []
            else:
                self.radius = choose_radius(count)
                self.positions = layout_positions(random.Random(0), count, self.radius)
            self.names = key
        return self.positions, self.radius


def newest_log() -> Path:
    logs = list((ROOT / "logs").glob("agent-*.log"))
    if not logs:
        raise SystemExit(f"no agent logs in {ROOT / 'logs'}")
    return max(logs, key=lambda path: path.stat().st_mtime)


def split_messages(text: str) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    kind: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = _BANNER.match(line)
        if match:
            if kind is not None:
                messages.append((kind, "\n".join(lines).strip()))
            kind = match.group(1)
            lines = []
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        messages.append((kind, "\n".join(lines).strip()))
    return messages


def clean_prose(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    collapsed = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return collapsed.strip()


def format_arg(value: str) -> str:
    if _NUMBER.fullmatch(value):
        return value
    return json.dumps(value)


def format_call(name: str, args: dict[str, str]) -> str:
    parts = [f"{key}={format_arg(value)}" for key, value in args.items()]
    return f"{name}({', '.join(parts)})"


def parse_calls(lines: list[str]) -> list[tuple[str, dict[str, str]]]:
    calls: list[tuple[str, dict[str, str]]] = []
    name: str | None = None
    args: dict[str, str] = {}
    in_args = False

    def flush() -> None:
        nonlocal name, args, in_args
        if name is not None:
            calls.append((name, args))
        name = None
        args = {}
        in_args = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("Call ID:"):
            continue
        if stripped == "Args:":
            in_args = True
            continue
        match = _CALL.match(stripped)
        if match:
            flush()
            name = match.group(1)
            args = {}
            in_args = False
            continue
        if in_args and name is not None:
            arg = _ARG.match(stripped)
            if arg:
                args[arg.group(1).strip()] = arg.group(2)
    flush()
    return calls


def parse_ai(body: str) -> list[TextSegment]:
    sections: list[tuple[str, list[str]]] = []
    kind = "prose"
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        sections.append((kind, buf))
        buf = []

    for line in body.splitlines():
        header = line.strip()
        if header == "Thinking:":
            flush()
            kind = "thinking"
            continue
        if header in {"Tool Calls:", "Invalid Tool Calls:"}:
            flush()
            kind = "tools"
            continue
        buf.append(line)
    flush()

    segments: list[TextSegment] = []
    for section_kind, section_lines in sections:
        if section_kind == "tools":
            for name, args in parse_calls(section_lines):
                segments.append(TextSegment("call", format_call(name, args) + "\n"))
            continue
        prose = clean_prose("\n".join(section_lines))
        if prose:
            segments.append(TextSegment(section_kind, prose + "\n"))
    return segments


def strip_duplicate_block(text: str) -> str:
    """Drop the extra copy of the last message that agent.py appends.

    Keep a leading tab. The status matrix header starts with one, and str.strip
    would delete it.
    """
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) >= 2 and len(lines) % 2 == 0:
        half = len(lines) // 2
        if lines[:half] == lines[half:]:
            lines = lines[:half]
    if len(lines) >= 2 and lines[-1] == lines[-2]:
        lines = lines[:-1]
    return "\n".join(lines)


def parse_tool(body: str) -> tuple[str, str]:
    lines = body.splitlines()
    name = ""
    if lines and lines[0].startswith("Name:"):
        name = lines[0].split(":", 1)[1].strip()
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return name, strip_duplicate_block("\n".join(lines))


def parse_value_cell(cell: str) -> tuple[float | None, bool]:
    parts = cell.split()
    if not parts or parts[0] == "--":
        return None, "F" in parts[1:]
    try:
        value = float(parts[0])
    except ValueError:
        return None, False
    return value, "F" in parts[1:]


def parse_matrix(text: str) -> Matrix | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or "\t" not in lines[0]:
        return None
    header = lines[0].split("\t")
    if len(header) < 3 or header[-1].strip() != "value":
        return None
    names = [cell.strip() for cell in header[1:-1]]
    if not names or any(not name for name in names):
        return None
    edges: list[tuple[int, int]] = []
    values: dict[str, float | None] = {}
    frozen: set[str] = set()
    report = ""
    won = False
    rows: list[list[str]] = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == _WIN:
            report = stripped
            won = True
            continue
        if _REPORT.match(stripped):
            report = stripped
            continue
        cells = line.split("\t")
        if len(cells) != len(names) + 2:
            return None
        rows.append(cells)
    if len(rows) != len(names):
        return None
    won_match = _REPORT.match(report)
    if won_match is not None:
        won = won_match.group(1) == "0" and won_match.group(2) == "0"
    for src, cells in enumerate(rows):
        name = cells[0].strip()
        if name != names[src]:
            return None
        for dst, bit in enumerate(cells[1:-1]):
            if bit.strip() == "1":
                edges.append((src, dst))
            elif bit.strip() != "0":
                return None
        value, is_frozen = parse_value_cell(cells[-1].strip())
        values[name] = value
        if is_frozen:
            frozen.add(name)
    return Matrix(names, edges, values, frozen, report, won)


def load_segments(path: Path) -> list[Segment]:
    messages = split_messages(path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for kind, body in messages:
        if kind == "Human":
            continue
        if kind == "Ai":
            segments.extend(parse_ai(body))
        elif kind == "Tool":
            name, content = parse_tool(body)
            if name == "show_status":
                matrix = parse_matrix(content)
                if matrix is not None:
                    segments.append(MatrixSegment(matrix))
                    continue
            if content:
                segments.append(TextSegment("result", content + "\n"))
    spaced: list[Segment] = []
    for segment in segments:
        if isinstance(segment, TextSegment) and segment.role in {"thinking", "prose"} and spaced:
            segment = TextSegment(segment.role, "\n" + segment.text)
        spaced.append(segment)
    return spaced


def _text_duration(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_FRAME) * TYPE_MS


def _segment_duration(segment: Segment) -> int:
    if isinstance(segment, MatrixSegment):
        return HOLD_MS
    return _text_duration(segment.text)


def _replay_duration(segments: list[Segment]) -> int:
    return sum(_segment_duration(segment) for segment in segments)


def _take_head(segments: list[Segment], budget: int) -> tuple[list[Segment], list[Segment]]:
    head: list[Segment] = []
    index = 0
    while index < len(segments) and budget >= TYPE_MS:
        segment = segments[index]
        if isinstance(segment, MatrixSegment):
            if HOLD_MS > budget:
                break
            head.append(segment)
            budget -= HOLD_MS
            index += 1
            continue
        duration = _text_duration(segment.text)
        if duration <= budget:
            head.append(segment)
            budget -= duration
            index += 1
            continue
        chars = (budget // TYPE_MS) * CHARS_PER_FRAME
        if chars <= 0:
            break
        head.append(TextSegment(segment.role, segment.text[:chars]))
        rest = [TextSegment(segment.role, segment.text[chars:])]
        rest.extend(segments[index + 1 :])
        return head, rest
    return head, segments[index:]


def _take_tail(segments: list[Segment], budget: int) -> tuple[list[Segment], list[Segment]]:
    tail: list[Segment] = []
    index = len(segments)
    while index > 0 and budget >= TYPE_MS:
        index -= 1
        segment = segments[index]
        if isinstance(segment, MatrixSegment):
            if HOLD_MS > budget:
                index += 1
                break
            tail.append(segment)
            budget -= HOLD_MS
            continue
        duration = _text_duration(segment.text)
        if duration <= budget:
            tail.append(segment)
            budget -= duration
            continue
        chars = (budget // TYPE_MS) * CHARS_PER_FRAME
        if chars <= 0:
            index += 1
            break
        tail.append(TextSegment(segment.role, segment.text[-chars:]))
        index += 1
        break
    tail.reverse()
    return segments[:index], tail


def _last_matrix(segments: list[Segment]) -> Matrix | None:
    found: Matrix | None = None
    for segment in segments:
        if isinstance(segment, MatrixSegment):
            found = segment.matrix
    return found


def trim_replay(segments: list[Segment]) -> list[Segment]:
    """Keep the opening and the ending. The middle becomes a single `[...]` hold."""
    if _replay_duration(segments) <= HEAD_MS + TAIL_MS:
        return segments
    head, rest = _take_head(segments, HEAD_MS)
    skipped, tail = _take_tail(rest, TAIL_MS)
    snap = _last_matrix(skipped)

    def needs_snap(matrix: Matrix | None, ending: list[Segment]) -> bool:
        return matrix is not None and not (
            ending and isinstance(ending[0], MatrixSegment) and ending[0].matrix is matrix
        )

    if needs_snap(snap, tail):
        skipped, tail = _take_tail(rest, TAIL_MS - HOLD_MS)
        snap = _last_matrix(skipped)
    planned: list[Segment] = list(head)
    planned.append(TextSegment("skip", "\n[...]\n"))
    if needs_snap(snap, tail) and snap is not None:
        planned.append(MatrixSegment(snap))
    planned.extend(tail)
    return planned


def draw_board(
    surface: pygame.Surface,
    matrix: Matrix | None,
    layout: _Layout,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    surface.fill(BG)
    pygame.draw.rect(surface, BAR, (0, 0, WIDTH, BAR_H))
    if matrix is None:
        return
    hint = small.render(matrix.report, True, MUTED)
    surface.blit(hint, (8, (BAR_H - hint.get_height()) // 2))
    positions, radius = layout.for_names(matrix.names)
    view = _EdgeView(positions, radius)
    color = ARROW_WIN if matrix.won else ARROW
    for src, dst in matrix.edges:
        segment = edge_segment(view, src, dst)
        if segment is not None:
            draw_arrow(surface, color, segment[0], segment[1])
    for index, name in enumerate(matrix.names):
        frozen = name in matrix.frozen
        x, y = positions[index]
        pygame.draw.circle(surface, NODE, (x, y), radius)
        pygame.draw.circle(surface, INTERVENE_RING if frozen else NODE_RING, (x, y), radius, 3 if frozen else 2)
        blit_fitted(
            surface,
            font.render(name, True, TEXT),
            small.render(fmt_value(matrix.values[name]), True, MUTED),
            (x, y),
            radius,
        )


class _EdgeView:
    def __init__(self, positions: list[tuple[float, float]], radius: int) -> None:
        self.positions = positions
        self.radius = radius


class _Trace:
    """Wrapped trace window. Only the visible lines are kept."""

    def __init__(self, font: pygame.font.Font, max_width: int, max_lines: int) -> None:
        self.font = font
        self.max_width = max_width
        self.max_lines = max_lines
        self.lines: list[list[tuple[str, tuple[int, int, int]]]] = [[]]
        self.width = 0
        self._sizes: dict[str, int] = {}

    def _size(self, text: str) -> int:
        width = self._sizes.get(text)
        if width is None:
            width = self.font.size(text)[0]
            self._sizes[text] = width
        return width

    def _newline(self) -> None:
        self.lines.append([])
        self.width = 0
        extra = len(self.lines) - self.max_lines
        if extra > 0:
            del self.lines[:extra]

    def append(self, text: str, color: tuple[int, int, int]) -> None:
        for part in re.split(r"(\n)", text):
            if part == "\n":
                self._newline()
                continue
            if not part:
                continue
            for token in re.findall(r"\S+|\s+", part):
                token_width = self._size(token)
                if token.isspace():
                    if self.width == 0 or self.width + token_width > self.max_width:
                        continue
                    self.lines[-1].append((token, color))
                    self.width += token_width
                    continue
                if token_width > self.max_width:
                    for char in token:
                        char_width = self._size(char)
                        if self.width + char_width > self.max_width and self.width > 0:
                            self._newline()
                        self.lines[-1].append((char, color))
                        self.width += char_width
                    continue
                if self.width + token_width > self.max_width and self.width > 0:
                    self._newline()
                self.lines[-1].append((token, color))
                self.width += token_width

    def draw(self, surface: pygame.Surface, caret: bool) -> None:
        surface.fill(BG)
        pygame.draw.line(surface, BAR, (0, 0), (WIDTH, 0), 2)
        lines = self.lines
        if caret:
            saved = (self.lines, self.width)
            self.lines = self.lines[:-1] + [list(self.lines[-1])]
            self.append("|", TEXT)
            lines = self.lines
            self.lines, self.width = saved
        line_h = self.font.get_linesize()
        y = 6
        for line in lines[-self.max_lines :]:
            x = 8
            for text, color in line:
                image = self.font.render(text, True, color)
                surface.blit(image, (x, y))
                x += image.get_width()
            y += line_h


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = max(total, 1)
        self.done = 0
        self._last = 0

    def advance(self, label: str) -> None:
        self.done += 1
        self._show(label)

    def _show(self, label: str) -> None:
        ratio = min(1.0, self.done / self.total)
        cells = 28
        filled = int(cells * ratio)
        bar = "#" * filled + "-" * (cells - filled)
        msg = f"\r{label:<8} {ratio * 100:5.1f}% [{bar}] {min(self.done, self.total)}/{self.total}"
        pad = " " * max(0, self._last - len(msg))
        print(msg + pad, end="", file=sys.stderr, flush=True)
        self._last = len(msg)

    def note(self, label: str) -> None:
        self._show(label)

    def finish(self) -> None:
        if self._last:
            print(file=sys.stderr)


def _frame_count(segments: list[Segment]) -> int:
    total = 0
    for segment in segments:
        if isinstance(segment, MatrixSegment) or segment.role == "skip":
            total += 1
            continue
        if segment.text:
            total += math.ceil(len(segment.text) / CHARS_PER_FRAME)
    return total


def role_color(role: str) -> tuple[int, int, int]:
    if role == "call":
        return CALL_COLOR
    if role == "result":
        return RESULT_COLOR
    if role == "prose":
        return TEXT
    return MUTED


def palette_image() -> Image.Image:
    colors: list[tuple[int, int, int]] = []

    def add(color: tuple[int, int, int]) -> None:
        if color not in colors:
            colors.append(color)

    for color in (
        BG,
        BAR,
        NODE,
        NODE_RING,
        INTERVENE_RING,
        TEXT,
        MUTED,
        ARROW,
        ARROW_WIN,
        CALL_COLOR,
        RESULT_COLOR,
    ):
        add(color)
    for fg, bg in ((TEXT, BG), (MUTED, BG), (TEXT, NODE), (MUTED, NODE), (CALL_COLOR, BG), (RESULT_COLOR, BG)):
        for step in range(1, 8):
            mix = step / 8
            add(tuple(int(fg[channel] * mix + bg[channel] * (1 - mix)) for channel in range(3)))
    level = 0
    while len(colors) < 256:
        add((level % 256, level % 256, level % 256))
        level += 3
    flat: list[int] = []
    for red, green, blue in colors[:256]:
        flat.extend((red, green, blue))
    image = Image.new("P", (1, 1))
    image.putpalette(flat[: 256 * 3])
    return image


def capture(surface: pygame.Surface) -> Image.Image:
    raw = pygame.image.tobytes(surface, "RGB")
    return Image.frombytes("RGB", surface.get_size(), raw)


def render_gif(segments: list[Segment], dest: Path) -> list[Matrix]:
    planned = trim_replay(segments)
    pygame.init()
    pygame.display.set_mode((1, 1))
    font = load_font(20, 22)
    small = load_font(14, 16)
    trace_font = load_font(14, 16)
    canvas = pygame.Surface((WIDTH, HEIGHT + PANEL_H))
    board = canvas.subsurface((0, 0, WIDTH, HEIGHT))
    panel = canvas.subsurface((0, HEIGHT, WIDTH, PANEL_H))
    layout = _Layout()
    palette = palette_image()
    frames: list[Image.Image] = []
    durations: list[int] = []
    line_h = trace_font.get_linesize()
    trace = _Trace(trace_font, WIDTH - 16, max(1, (PANEL_H - 8) // line_h))
    matrix: Matrix | None = None
    matrices: list[Matrix] = []
    progress = _Progress(_frame_count(planned))

    def push(duration: int, caret: bool) -> None:
        draw_board(board, matrix, layout, font, small)
        trace.draw(panel, caret)
        frames.append(capture(canvas).quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(duration)
        progress.advance("replay")

    try:
        for segment in planned:
            if isinstance(segment, MatrixSegment):
                matrix = segment.matrix
                matrices.append(matrix)
                push(HOLD_MS, caret=False)
                continue
            if segment.role == "skip":
                trace.append(segment.text, MUTED)
                push(SKIP_MS, caret=False)
                continue
            color = role_color(segment.role)
            pending = segment.text
            index = 0
            while index < len(pending):
                nxt = min(len(pending), index + CHARS_PER_FRAME)
                trace.append(pending[index:nxt], color)
                index = nxt
                push(TYPE_MS, caret=True)
        if frames:
            durations[-1] = max(durations[-1], END_HOLD_MS)
            draw_board(board, matrix, layout, font, small)
            trace.draw(panel, caret=False)
            frames[-1] = capture(canvas).quantize(palette=palette, dither=Image.Dither.NONE)
        if not frames:
            push(END_HOLD_MS, caret=False)
        progress.note("writing")
        frames[0].save(
            dest,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
        )
    finally:
        progress.finish()
        pygame.quit()
    return matrices


def describe(matrix: Matrix) -> str:
    edges = ", ".join(f"{matrix.names[src]}->{matrix.names[dst]}" for src, dst in matrix.edges) or "none"
    frozen = ", ".join(name for name in matrix.names if name in matrix.frozen) or "none"
    values = ", ".join(f"{name}={fmt_value(matrix.values[name])}" for name in matrix.names)
    return f"edges={edges} frozen={frozen} values={values} won={matrix.won} | {matrix.report}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an agent log as a stacked GIF.")
    parser.add_argument("log", nargs="?", type=Path, help="Agent log. Defaults to the newest logs/agent-*.log.")
    args = parser.parse_args()
    path = (args.log if args.log is not None else newest_log()).resolve()
    if not path.is_file():
        raise SystemExit(f"log not found: {path}")
    segments = load_segments(path)
    dest = path.with_suffix(".gif")
    matrices = render_gif(segments, dest)
    print(f"wrote {dest.relative_to(ROOT)} ({len(matrices)} matrices)")
    for index, matrix in enumerate(matrices, start=1):
        print(f"  {index}. {describe(matrix)}")


if __name__ == "__main__":
    main()
