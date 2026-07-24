"""Render `docs/media/reproduce.gif`, the terminal walkthrough shown in the README.

The recording replays `scripts/media/readme_walkthrough.txt`, a storyboard whose output lines are
real stdout/stderr captured by running each command in a clean checkout. That file documents the
only two edits applied to the capture: the clone's absolute path is rewritten so no home directory
is recorded, and long runs of progress output are cut with the size of each cut stated on screen.

Deterministic: no randomness, no clock, no network. Re-running overwrites the GIF byte for byte on
the same Pillow and font.

    uv run python scripts/media/make_readme_gif.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
STORYBOARD = Path(__file__).with_name("readme_walkthrough.txt")
DEFAULT_OUT = REPO_ROOT / "docs" / "media" / "reproduce.gif"

# Terminal geometry. COLS is set by the widest real output line in the storyboard (pytest's
# 99-character progress rows); anything longer soft-wraps the way a terminal would.
COLS = 100
ROWS = 28
FONT_SIZE = 16
LINE_H = 21
PAD = 14
HEADER_H = 34
TITLE = "pretrain-or-prompt  \u00b7  reproduce the study on CPU"
PROMPT = "PS C:\\code\\Strats-for-Bug-Fixing>"

BG = (13, 17, 23)
HEADER_BG = (22, 27, 34)
INK = {
    "out": (201, 209, 217),
    "dim": (110, 118, 129),
    "hi": (63, 185, 80),
    # Amber, deliberately unlike any of the output inks: `~` lines are this storyboard talking,
    # not the program.
    "note": (219, 154, 44),
    "prompt": (88, 166, 255),
    "cmd": (230, 237, 243),
    "title": (139, 148, 158),
    "dot_r": (255, 95, 86),
    "dot_y": (255, 189, 46),
    "dot_g": (39, 201, 63),
}

# Per-frame durations in milliseconds. Typing runs at 25 fps, output reveals at 14-25 fps, and
# each scroll is interpolated over SCROLL_STEPS sub-frames at 50 fps. The holds after a command
# are long frames with a blinking cursor over them.
#
# Every value is a multiple of 10 and never below 20: GIF stores delays in centiseconds, so
# anything finer is truncated, and browsers inflate a 1-centisecond delay to a full 100 ms.
TYPE_MS = 40
TYPE_SPACE_MS = 90
CMD_BEAT_MS = 380
REVEAL_MS = {"out": 70, "dim": 50, "hi": 110, "note": 620}
BLANK_MS = 40
BLINK_MS = 470
PROMPT_MS = 320
TAIL_MS = 900
SCROLL_STEPS = 3
SCROLL_MS = 20

FONT_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Menlo.ttc",
)

Line = tuple[str, str]  # (style, text)
Screen = tuple[tuple[Line, ...], bool, int]  # (visible lines, cursor on, scroll offset in px)


def load_font() -> ImageFont.FreeTypeFont:
    """Return a real monospace TrueType face; the bitmap default is unreadable at this size."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, FONT_SIZE)
    raise SystemExit(f"no monospace TTF found; looked for {', '.join(FONT_CANDIDATES)}")


def parse_storyboard(path: Path) -> list[tuple[str, str]]:
    """Parse the storyboard into (directive, payload) steps."""
    steps: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        directive, _, payload = raw.partition(" ")
        if directive not in {"$", ">", "-", "*", "~", "."}:
            raise SystemExit(f"unknown storyboard directive {directive!r} in: {raw}")
        steps.append((directive, payload))
    return steps


def wrap(text: str, cols: int) -> list[str]:
    """Hard-wrap at the terminal width, keeping empty lines visible."""
    if not text:
        return [""]
    return [text[i : i + cols] for i in range(0, len(text), cols)]


def build_screens(steps: list[tuple[str, str]]) -> list[tuple[Screen, int]]:
    """Turn the storyboard into a (screen, duration) timeline."""
    buffer: list[Line] = []
    timeline: list[tuple[Screen, int]] = []

    def snapshot(cursor: bool, ms: int) -> None:
        timeline.append(((tuple(buffer[-ROWS:]), cursor, 0), ms))

    def append(line: Line, cursor: bool, ms: int) -> None:
        """Add a line, interpolating the scroll when the viewport is already full."""
        scrolls = len(buffer) >= ROWS
        buffer.append(line)
        if scrolls:
            window = tuple(buffer[-(ROWS + 1) :])
            for step in range(1, SCROLL_STEPS + 1):
                offset = round(LINE_H * step / (SCROLL_STEPS + 1))
                timeline.append(((window, cursor, offset), SCROLL_MS))
        snapshot(cursor, ms)

    def hold(ms: int) -> None:
        """Spend `ms` on the current screen, blinking the cursor so the terminal stays alive."""
        remaining, cursor = ms, True
        while remaining > BLINK_MS:
            snapshot(cursor, BLINK_MS)
            remaining -= BLINK_MS
            cursor = not cursor
        snapshot(cursor, max(remaining, 1))

    styles = {">": "out", "-": "dim", "*": "hi", "~": "note"}

    for directive, payload in steps:
        if directive == "$":
            if buffer:
                append(("out", ""), True, BLANK_MS)
            append(("prompt", ""), True, PROMPT_MS)
            for i, char in enumerate(payload, start=1):
                buffer[-1] = ("prompt", payload[:i])
                snapshot(True, TYPE_SPACE_MS if char == " " else TYPE_MS)
            hold(CMD_BEAT_MS)
        elif directive == ".":
            hold(int(payload))
        else:
            style = styles[directive]
            for chunk in wrap(payload, COLS):
                append((style, chunk), True, BLANK_MS if not chunk else REVEAL_MS[style])

    hold(TAIL_MS)
    return timeline


def build_palette() -> Image.Image:
    """A fixed palette of every ink colour blended over both backgrounds.

    Antialiased glyph edges are exactly such blends, so quantizing against this palette keeps the
    text crisp while every frame shares one palette, which is what lets the GIF delta-encode.
    """
    colors: list[tuple[int, int, int]] = []
    for bg in (BG, HEADER_BG):
        for fg in INK.values():
            for step in range(8):
                t = step / 7
                r, g, b = (
                    round(back + (fore - back) * t) for fore, back in zip(fg, bg, strict=True)
                )
                colors.append((r, g, b))
    unique = list(dict.fromkeys(colors))[:256]
    unique += [unique[-1]] * (256 - len(unique))
    palette = Image.new("P", (1, 1))
    palette.putpalette([channel for color in unique for channel in color])
    return palette


def render(screen: Screen, font: ImageFont.FreeTypeFont, size: tuple[int, int]) -> Image.Image:
    """Draw one terminal frame."""
    lines, cursor, offset = screen
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, size[0], HEADER_H), fill=HEADER_BG)
    for i, key in enumerate(("dot_r", "dot_y", "dot_g")):
        cx = PAD + 6 + i * 17
        draw.ellipse((cx - 5, HEADER_H // 2 - 5, cx + 5, HEADER_H // 2 + 5), fill=INK[key])
    title_w = draw.textlength(TITLE, font=font)
    draw.text(
        ((size[0] - title_w) / 2, HEADER_H / 2 - FONT_SIZE / 2 - 1), TITLE, INK["title"], font
    )

    # The body is drawn one row taller than the viewport and then cropped, so a partly scrolled
    # top line is clipped instead of bleeding into the title bar.
    body = Image.new("RGB", (size[0], LINE_H * (ROWS + 1)), BG)
    body_draw = ImageDraw.Draw(body)
    char_w = draw.textlength("M", font=font)
    for row, (style, text) in enumerate(lines):
        y = row * LINE_H
        if style == "prompt":
            body_draw.text((PAD, y), PROMPT, INK["prompt"], font)
            x = PAD + char_w * (len(PROMPT) + 1)
            body_draw.text((x, y), text, INK["cmd"], font)
            end = x + char_w * len(text)
        else:
            body_draw.text((PAD, y), text, INK[style], font)
            end = PAD + char_w * len(text)
        if cursor and row == len(lines) - 1:
            body_draw.rectangle((end + 1, y + 2, end + char_w, y + LINE_H - 4), fill=INK["cmd"])
    img.paste(body.crop((0, offset, size[0], offset + LINE_H * ROWS)), (0, HEADER_H + PAD))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    font = load_font()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    char_w = probe.textlength("M", font=font)
    size = (round(PAD * 2 + char_w * COLS), HEADER_H + PAD * 2 + LINE_H * ROWS)

    timeline = build_screens(parse_storyboard(STORYBOARD))
    palette = build_palette()
    frames = [
        render(screen, font, size).quantize(palette=palette, dither=Image.Dither.NONE)
        for screen, _ in timeline
    ]
    durations = [ms for _, ms in timeline]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    total = sum(durations) / 1000
    kib = args.out.stat().st_size / 1024
    print(f"wrote {args.out}")
    print(f"  {size[0]}x{size[1]}, {len(frames)} frames, {total:.1f}s, {kib / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
