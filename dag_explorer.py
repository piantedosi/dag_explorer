"""Discover the arrows of a hidden causal DAG."""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass

import pygame

from dag import Dag, edge_report

NODE_COUNT = 2
MAX_NODES = 26
WIDTH = 640
HEIGHT = 360

BAR_H = 28
DRAG_THRESHOLD = 6
EDGE_HIT = 8

BG = (22, 24, 28)
BAR = (16, 18, 22)
NODE = (46, 50, 58)
NODE_RING = (186, 192, 202)
INTERVENE_RING = (232, 168, 64)
FOCUS_RING = (244, 244, 248)
TEXT = (236, 238, 242)
MUTED = (176, 182, 192)
ARROW = (168, 176, 188)
ARROW_WIN = (88, 196, 120)
PREVIEW = (120, 128, 140)
BUTTON = (58, 86, 138)
BUTTON_HOT = (78, 112, 172)
BUTTON_TEXT = (244, 246, 250)
INPUT_BG = (28, 30, 36)
CONFETTI_COLORS = (
    (240, 92, 92),
    (88, 196, 120),
    (92, 148, 232),
    (240, 196, 72),
    (196, 112, 220),
)


@dataclass
class Game:
    dag: Dag
    positions: list[tuple[float, float]]
    radius: int
    values: dict[str, float | None]
    drag_from: int | None
    drag_pos: tuple[int, int] | None
    drag_moved: bool
    press_pos: tuple[int, int] | None
    armed: str | None
    focus: int | None
    buffer: str
    won: bool
    revealed: bool
    easy: bool
    particles: list[dict]


def choose_radius(count: int) -> int:
    sine = math.sin(math.pi / count)
    avail = min(WIDTH / 2 - 8, (HEIGHT - BAR_H) / 2 - 6)
    gap = 10.0
    raw = (avail * sine - gap / 2) / (sine + 1)
    return max(13, min(34, int(raw)))


def layout_positions(rng: random.Random, count: int, node_radius: int) -> list[tuple[float, float]]:
    center_x = WIDTH / 2
    center_y = BAR_H + (HEIGHT - BAR_H) / 2
    limit = min(WIDTH / 2 - 8, (HEIGHT - BAR_H) / 2 - 6)
    ring = limit - node_radius
    order = list(range(count))
    rng.shuffle(order)
    positions = [(0.0, 0.0)] * count
    angle_jitter = 0.08 if count <= 8 else 0.0
    radial_jitter = 0.02 if count <= 8 else 0.0
    for slot, index in enumerate(order):
        angle = 2 * math.pi * slot / count - math.pi / 2
        angle += rng.uniform(-angle_jitter, angle_jitter)
        orbit = ring * rng.uniform(1 - radial_jitter, 1 + radial_jitter)
        positions[index] = (
            center_x + orbit * math.cos(angle),
            center_y + orbit * math.sin(angle),
        )
    return positions


def fmt_value(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def make_game(rng: random.Random, count: int = NODE_COUNT, easy: bool = False) -> Game:
    dag = Dag.generate(rng, count)
    radius = choose_radius(count)
    return Game(
        dag=dag,
        positions=layout_positions(rng, count, radius),
        radius=radius,
        values=dag.evaluate(),
        drag_from=None,
        drag_pos=None,
        drag_moved=False,
        press_pos=None,
        armed=None,
        focus=None,
        buffer="",
        won=False,
        revealed=False,
        easy=easy,
        particles=[],
    )


def recompute(game: Game) -> None:
    game.values = game.dag.evaluate()


def spawn_confetti(rng: random.Random) -> list[dict]:
    particles = []
    for _ in range(90):
        angle = rng.uniform(0, math.tau)
        speed = rng.uniform(50, 220)
        particles.append(
            {
                "x": WIDTH / 2 + rng.uniform(-24, 24),
                "y": (BAR_H + HEIGHT) / 2,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - 70,
                "color": CONFETTI_COLORS[rng.randrange(len(CONFETTI_COLORS))],
                "life": rng.uniform(0.7, 1.5),
                "w": rng.randint(3, 5),
                "h": rng.randint(3, 6),
            }
        )
    return particles


def check_win(game: Game, rng: random.Random) -> None:
    if game.won:
        return
    if set(game.dag.hypothesis().edges) == set(game.dag.edges):
        game.won = True
        game.focus = None
        game.buffer = ""
        game.drag_from = None
        game.particles = spawn_confetti(rng)


def add_edge(game: Game, src: int | None, dst: int | None, rng: random.Random) -> None:
    if game.won or src is None or dst is None or src == dst:
        return
    edge = (src, dst)
    if edge in game.dag.hypothesis().edges:
        return
    game.dag.add_edge(src, dst)
    check_win(game, rng)


def accept_char(buffer: str, char: str) -> str:
    if char.isdigit():
        return buffer + char
    if char == "-" and buffer == "":
        return buffer + char
    if char == "." and "." not in buffer:
        return buffer + char
    return buffer


def apply_buffer(game: Game) -> None:
    if game.focus is None:
        return
    try:
        value = float(game.buffer)
    except ValueError:
        return
    if not math.isfinite(value):
        return
    game.dag.force_node(game.dag.nodes[game.focus].name, value)
    game.focus = None
    game.buffer = ""
    recompute(game)


def edges_locked(game: Game) -> bool:
    return game.won or game.revealed


def reveal_solution(game: Game) -> None:
    if game.won or game.revealed:
        return
    game.revealed = True
    game.focus = None
    game.buffer = ""
    game.drag_from = None
    game.drag_moved = False


def click_node(game: Game, index: int) -> None:
    if game.won:
        return
    name = game.dag.nodes[index].name
    if name in game.dag.hypothesis().frozen:
        game.dag.force_node(name, None)
        if game.focus == index:
            game.focus = None
            game.buffer = ""
        recompute(game)
        return
    game.focus = index
    game.buffer = ""


def node_at(game: Game, pos: tuple[int, int]) -> int | None:
    px, py = pos
    limit = game.radius * game.radius
    for index, (x, y) in enumerate(game.positions):
        if (x - px) ** 2 + (y - py) ** 2 <= limit:
            return index
    return None


def edge_segment(
    game: Game, src: int, dst: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x1, y1 = game.positions[src]
    x2, y2 = game.positions[dst]
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1:
        return None
    ux, uy = dx / dist, dy / dist
    start = (x1 + ux * (game.radius + 1), y1 + uy * (game.radius + 1))
    end = (x2 - ux * (game.radius + 2), y2 - uy * (game.radius + 2))
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 4:
        return None
    return start, end


def point_segment_distance(
    px: float, py: float, start: tuple[float, float], end: tuple[float, float]
) -> float:
    ax, ay = start
    bx, by = end
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def edge_at(game: Game, pos: tuple[int, int]) -> tuple[int, int] | None:
    best = None
    best_dist = EDGE_HIT
    for edge in game.dag.hypothesis().edges:
        segment = edge_segment(game, edge[0], edge[1])
        if segment is None:
            continue
        dist = point_segment_distance(pos[0], pos[1], segment[0], segment[1])
        if dist <= best_dist:
            best = edge
            best_dist = dist
    return best


def reveal_rect() -> pygame.Rect:
    return pygame.Rect(WIDTH - 84, 4, 76, BAR_H - 8)


def easy_rect() -> pygame.Rect:
    return pygame.Rect(WIDTH - 176, 4, 84, BAR_H - 8)


def setup_easy_rect() -> pygame.Rect:
    return pygame.Rect(WIDTH // 2 - 58, 256, 116, 28)


def restart_rect() -> pygame.Rect:
    return pygame.Rect(8, 4, 76, BAR_H - 8)


def next_rect() -> pygame.Rect:
    return pygame.Rect(90, 4, 52, BAR_H - 8)


def setup_minus_rect() -> pygame.Rect:
    return pygame.Rect(WIDTH // 2 - 92, 156, 36, 32)


def setup_plus_rect() -> pygame.Rect:
    return pygame.Rect(WIDTH // 2 + 56, 156, 36, 32)


def setup_start_rect() -> pygame.Rect:
    return pygame.Rect(WIDTH // 2 - 58, 214, 116, 34)


def on_mouse_down(game: Game, event: pygame.event.Event, rng: random.Random) -> None:
    if event.button == 3:
        if not edges_locked(game):
            edge = edge_at(game, event.pos)
            if edge is not None:
                game.dag.remove_edge(*edge)
                check_win(game, rng)
        return
    if event.button != 1:
        return
    if restart_rect().collidepoint(event.pos):
        game.armed = "restart"
        game.drag_from = None
        return
    if game.won and next_rect().collidepoint(event.pos):
        game.armed = "next"
        game.drag_from = None
        return
    if easy_rect().collidepoint(event.pos):
        game.armed = "easy"
        game.drag_from = None
        return
    if not edges_locked(game) and reveal_rect().collidepoint(event.pos):
        game.armed = "reveal"
        game.drag_from = None
        return
    game.armed = None
    if game.won:
        return
    game.drag_from = node_at(game, event.pos)
    game.drag_moved = False
    game.press_pos = event.pos
    game.drag_pos = event.pos


def on_mouse_up(game: Game, event: pygame.event.Event, rng: random.Random) -> str | None:
    if event.button != 1:
        return
    armed = game.armed
    game.armed = None
    if armed == "restart" and restart_rect().collidepoint(event.pos):
        return "setup"
    if armed == "next" and game.won and next_rect().collidepoint(event.pos):
        return "next"
    if armed == "easy" and easy_rect().collidepoint(event.pos):
        game.easy = not game.easy
        game.drag_from = None
        return None
    if armed == "reveal" and reveal_rect().collidepoint(event.pos):
        reveal_solution(game)
        game.drag_from = None
        return None
    if game.drag_from is not None and not game.won:
        if game.drag_moved and not game.revealed:
            add_edge(game, game.drag_from, node_at(game, event.pos), rng)
        elif not game.drag_moved:
            click_node(game, game.drag_from)
    elif not game.won and game.drag_from is None:
        game.focus = None
        game.buffer = ""
    game.drag_from = None
    game.drag_moved = False
    return None


def on_motion(game: Game, event: pygame.event.Event) -> None:
    game.drag_pos = event.pos
    if game.drag_from is None or game.press_pos is None or game.drag_moved:
        return
    dx = event.pos[0] - game.press_pos[0]
    dy = event.pos[1] - game.press_pos[1]
    if dx * dx + dy * dy >= DRAG_THRESHOLD * DRAG_THRESHOLD:
        game.drag_moved = True
        game.focus = None
        game.buffer = ""


def on_key(game: Game, event: pygame.event.Event) -> None:
    if game.focus is None or game.won:
        return
    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        apply_buffer(game)
    elif event.key == pygame.K_ESCAPE:
        game.focus = None
        game.buffer = ""
    elif event.key == pygame.K_BACKSPACE:
        game.buffer = game.buffer[:-1]
    elif event.unicode:
        game.buffer = accept_char(game.buffer, event.unicode)


def update_particles(game: Game, dt: float) -> None:
    alive = []
    for particle in game.particles:
        particle["life"] -= dt
        if particle["life"] <= 0:
            continue
        particle["vy"] += 420 * dt
        particle["x"] += particle["vx"] * dt
        particle["y"] += particle["vy"] * dt
        alive.append(particle)
    game.particles = alive


def draw_arrow(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    pygame.draw.line(surface, color, start, end, 2)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 8
    spread = 0.45
    left = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    right = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )
    pygame.draw.polygon(surface, color, (end, left, right))


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    hot: bool,
) -> None:
    pygame.draw.rect(surface, BUTTON_HOT if hot else BUTTON, rect, border_radius=4)
    text = font.render(label, True, BUTTON_TEXT)
    surface.blit(text, text.get_rect(center=rect.center))


def blit_fitted(
    surface: pygame.Surface,
    name_image: pygame.Surface,
    value_image: pygame.Surface,
    center: tuple[float, float],
    node_radius: int,
) -> None:
    gap = 1
    block_w = max(name_image.get_width(), value_image.get_width())
    block_h = name_image.get_height() + gap + value_image.get_height()
    corner = math.hypot(block_w / 2, block_h / 2)
    scale = min(1.0, (node_radius - 5) / max(corner, 1))
    if scale < 0.99:
        name_image = pygame.transform.smoothscale(
            name_image,
            (max(1, int(name_image.get_width() * scale)), max(1, int(name_image.get_height() * scale))),
        )
        value_image = pygame.transform.smoothscale(
            value_image,
            (max(1, int(value_image.get_width() * scale)), max(1, int(value_image.get_height() * scale))),
        )
    top = center[1] - (name_image.get_height() + gap + value_image.get_height()) / 2
    surface.blit(name_image, name_image.get_rect(midtop=(center[0], top)))
    surface.blit(
        value_image,
        value_image.get_rect(midtop=(center[0], top + name_image.get_height() + gap)),
    )


def draw_setup(
    surface: pygame.Surface,
    node_count: int,
    easy: bool,
    armed: str | None,
    font: pygame.font.Font,
    small: pygame.font.Font,
) -> None:
    surface.fill(BG)
    title = font.render("Causal DAG", True, TEXT)
    surface.blit(title, title.get_rect(center=(WIDTH / 2, 92)))
    caption = small.render("Nodes", True, MUTED)
    surface.blit(caption, caption.get_rect(center=(WIDTH / 2, 136)))
    draw_button(surface, setup_minus_rect(), "-", font, armed == "minus")
    draw_button(surface, setup_plus_rect(), "+", font, armed == "plus")
    count = font.render(str(node_count), True, TEXT)
    surface.blit(count, count.get_rect(center=(WIDTH / 2, setup_minus_rect().centery)))
    draw_button(surface, setup_start_rect(), "Start", small, armed == "start")
    draw_button(surface, setup_easy_rect(), "Easy: on" if easy else "Easy: off", small, easy or armed == "setup-easy")
    hint = small.render("2 to 26", True, MUTED)
    surface.blit(hint, hint.get_rect(center=(WIDTH / 2, 304)))


def draw(surface: pygame.Surface, game: Game, font: pygame.font.Font, small: pygame.font.Font) -> None:
    surface.fill(BG)
    pygame.draw.rect(surface, BAR, (0, 0, WIDTH, BAR_H))
    draw_button(surface, restart_rect(), "Restart", small, game.armed == "restart")
    if game.won:
        draw_button(surface, next_rect(), "Next", small, game.armed == "next")
    elif not game.revealed:
        if game.easy:
            missing, wrong = game.dag.edge_counts()
            hint = small.render(edge_report(missing, wrong), True, MUTED)
            hint_x = restart_rect().right + 8
            surface.blit(hint, (hint_x, (BAR_H - hint.get_height()) // 2))
        draw_button(surface, reveal_rect(), "Reveal", small, game.armed == "reveal")
    draw_button(surface, easy_rect(), "Easy: on" if game.easy else "Easy: off", small, game.easy or game.armed == "easy")

    show_solution = game.won or game.revealed
    true_edges = set(game.dag.edges)
    hypothesis = game.dag.hypothesis()
    edge_set = game.dag.edges if show_solution else hypothesis.edges
    nodes = game.dag.nodes
    node_radius = game.radius
    for src, dst in edge_set:
        segment = edge_segment(game, src, dst)
        if segment is None:
            continue
        correct = show_solution or (game.easy and (src, dst) in true_edges)
        draw_arrow(surface, ARROW_WIN if correct else ARROW, segment[0], segment[1])
    if game.drag_from is not None and game.drag_moved and game.drag_pos is not None and not edges_locked(game):
        origin = game.positions[game.drag_from]
        pygame.draw.line(surface, PREVIEW, origin, game.drag_pos, 2)

    hover = None
    if game.drag_from is not None and game.drag_moved and game.drag_pos is not None:
        hover = node_at(game, game.drag_pos)
    for node in nodes:
        ring = NODE_RING
        width = 2
        if node.name in hypothesis.frozen:
            ring = INTERVENE_RING
            width = 3
        elif game.focus == node.index or hover == node.index:
            ring = FOCUS_RING
        x, y = game.positions[node.index]
        pygame.draw.circle(surface, NODE, (x, y), node_radius)
        pygame.draw.circle(surface, ring, (x, y), node_radius, width)
        blit_fitted(
            surface,
            font.render(node.name, True, TEXT),
            small.render(fmt_value(game.values[node.name]), True, MUTED),
            (x, y),
            node_radius,
        )

    if game.focus is not None and not game.won:
        x, y = game.positions[game.focus]
        box = pygame.Rect(0, 0, 72, 18)
        box.midleft = (int(x + node_radius + 6), int(y))
        if box.right > WIDTH - 4:
            box.midright = (int(x - node_radius - 6), int(y))
        pygame.draw.rect(surface, INPUT_BG, box, border_radius=3)
        pygame.draw.rect(surface, FOCUS_RING, box, 1, border_radius=3)
        caret = small.render((game.buffer or "") + "|", True, TEXT)
        surface.blit(caret, caret.get_rect(midleft=(box.x + 4, box.centery)))

    for particle in game.particles:
        rect = pygame.Rect(int(particle["x"]), int(particle["y"]), particle["w"], particle["h"])
        pygame.draw.rect(surface, particle["color"], rect)


def load_font(size: int, fallback: int) -> pygame.font.Font:
    font_path = pygame.font.match_font("dejavusans") or pygame.font.match_font("freesans")
    if font_path:
        try:
            return pygame.font.Font(font_path, size)
        except (FileNotFoundError, OSError):
            pass
    return pygame.font.Font(None, fallback)


async def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Causal DAG Explorer")
    clock = pygame.time.Clock()
    font = load_font(20, 22)
    small = load_font(14, 16)
    rng = random.Random()
    node_count = NODE_COUNT
    easy = False
    game: Game | None = None
    armed: str | None = None
    running = True
    while running:
        dt = clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif game is None:
                node_count, armed, game, easy = handle_setup(event, node_count, easy, armed, game, rng)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_mouse_down(game, event, rng)
            elif event.type == pygame.MOUSEBUTTONUP:
                action = on_mouse_up(game, event, rng)
                if action == "setup":
                    game = None
                    armed = None
                elif action == "next":
                    count = min(MAX_NODES, len(game.dag.nodes) + 1)
                    game = make_game(rng, count, game.easy)
            elif event.type == pygame.MOUSEMOTION:
                on_motion(game, event)
            elif event.type == pygame.KEYDOWN:
                on_key(game, event)
        if game is None:
            draw_setup(screen, node_count, easy, armed, font, small)
        else:
            update_particles(game, dt)
            draw(screen, game, font, small)
        pygame.display.flip()
        await asyncio.sleep(0)
    pygame.quit()


def handle_setup(
    event: pygame.event.Event,
    node_count: int,
    easy: bool,
    armed: str | None,
    game: Game | None,
    rng: random.Random,
) -> tuple[int, str | None, Game | None, bool]:
    if event.type == pygame.KEYDOWN:
        if event.key in (pygame.K_LEFT, pygame.K_DOWN):
            node_count = max(2, node_count - 1)
        elif event.key in (pygame.K_RIGHT, pygame.K_UP):
            node_count = min(MAX_NODES, node_count + 1)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            game = make_game(rng, node_count, easy)
        return node_count, armed, game, easy
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        if setup_minus_rect().collidepoint(event.pos):
            armed = "minus"
        elif setup_plus_rect().collidepoint(event.pos):
            armed = "plus"
        elif setup_start_rect().collidepoint(event.pos):
            armed = "start"
        elif setup_easy_rect().collidepoint(event.pos):
            armed = "setup-easy"
        else:
            armed = None
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        choice = armed
        armed = None
        if choice == "minus" and setup_minus_rect().collidepoint(event.pos):
            node_count = max(2, node_count - 1)
        elif choice == "plus" and setup_plus_rect().collidepoint(event.pos):
            node_count = min(MAX_NODES, node_count + 1)
        elif choice == "setup-easy" and setup_easy_rect().collidepoint(event.pos):
            easy = not easy
        elif choice == "start" and setup_start_rect().collidepoint(event.pos):
            game = make_game(rng, node_count, easy)
    return node_count, armed, game, easy


if __name__ == "__main__":
    asyncio.run(main())
