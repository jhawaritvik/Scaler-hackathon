"""One-off generator for submission_artifacts/eval_comparison.{png,svg}.

Plots untrained vs trained mean grader score per task (easy/medium/hard +
overall) as a grouped bar chart. Pure-PIL + hand-written SVG, matching the
no-matplotlib convention used by plot_training_curves.py.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
trained = json.loads((ROOT / "eval_trained.json").read_text())
untrained = json.loads((ROOT / "eval_untrained.json").read_text())

TASKS = ["easy", "medium", "hard"]
LABELS = [*TASKS, "overall"]
U = [untrained[t]["mean"] for t in TASKS] + [untrained["overall_mean"]]
T = [trained[t]["mean"] for t in TASKS] + [trained["overall_mean"]]

WIDTH, HEIGHT = 960, 520
PAD_L, PAD_R, PAD_T, PAD_B = 80, 40, 70, 70
PLOT_W = WIDTH - PAD_L - PAD_R
PLOT_H = HEIGHT - PAD_T - PAD_B
Y_MAX = 1.0

GROUP_W = PLOT_W / len(LABELS)
BAR_W = GROUP_W * 0.36
GAP = GROUP_W * 0.06

COLOR_BG = (255, 255, 255)
COLOR_AXIS = (60, 60, 60)
COLOR_GRID = (220, 220, 220)
COLOR_TEXT = (40, 40, 40)
COLOR_U = (170, 170, 170)
COLOR_U_EDGE = (90, 90, 90)
COLOR_T = (217, 74, 61)
COLOR_T_EDGE = (122, 31, 23)
COLOR_DELTA_POS = (27, 122, 53)
COLOR_DELTA_NEG = (160, 32, 32)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def y_to_px(v: float) -> float:
    return PAD_T + (1.0 - v / Y_MAX) * PLOT_H


def render_png(out_path: Path) -> None:
    img = Image.new("RGBA", (WIDTH, HEIGHT), COLOR_BG + (255,))
    draw = ImageDraw.Draw(img)
    f_title = _font(20)
    f_label = _font(14)
    f_tick = _font(12)
    f_value = _font(12)
    f_delta = _font(13)

    title = "Trained vs untrained — held-out OpenEnv eval (5 seeds/task)"
    tw, th = _text_size(draw, title, f_title)
    draw.text(((WIDTH - tw) / 2, 18), title, fill=COLOR_TEXT, font=f_title)

    for i in range(6):
        v = i / 5.0
        y = y_to_px(v)
        draw.line([(PAD_L, y), (WIDTH - PAD_R, y)], fill=COLOR_GRID, width=1)
        tick = f"{v:.1f}"
        tw, th = _text_size(draw, tick, f_tick)
        draw.text((PAD_L - tw - 6, y - th / 2), tick, fill=COLOR_TEXT, font=f_tick)

    draw.line([(PAD_L, PAD_T), (PAD_L, PAD_T + PLOT_H)], fill=COLOR_AXIS, width=2)
    draw.line([(PAD_L, PAD_T + PLOT_H), (WIDTH - PAD_R, PAD_T + PLOT_H)],
              fill=COLOR_AXIS, width=2)

    yl = "Grader score (0–1)"
    yl_img = Image.new("RGBA", _text_size(draw, yl, f_label), (0, 0, 0, 0))
    yd = ImageDraw.Draw(yl_img)
    yd.text((0, 0), yl, fill=COLOR_TEXT, font=f_label)
    yl_img = yl_img.rotate(90, expand=True)
    img.paste(yl_img, (12, PAD_T + (PLOT_H - yl_img.height) // 2), yl_img)

    for i, label in enumerate(LABELS):
        cx = PAD_L + GROUP_W * (i + 0.5)
        u, t = U[i], T[i]
        x_u = cx - BAR_W - GAP / 2
        x_t = cx + GAP / 2

        for x, v, fill, edge in [
            (x_u, u, COLOR_U, COLOR_U_EDGE),
            (x_t, t, COLOR_T, COLOR_T_EDGE),
        ]:
            y_top = y_to_px(v)
            y_bot = PAD_T + PLOT_H
            draw.rectangle([x, y_top, x + BAR_W, y_bot], fill=fill, outline=edge, width=1)
            txt = f"{v:.3f}"
            tw, th = _text_size(draw, txt, f_value)
            draw.text((x + BAR_W / 2 - tw / 2, y_top - th - 4),
                      txt, fill=COLOR_TEXT, font=f_value)

        delta = t - u
        d_color = COLOR_DELTA_POS if delta >= 0 else COLOR_DELTA_NEG
        d_text = f"Δ {delta:+.3f}"
        dtw, dth = _text_size(draw, d_text, f_delta)
        y_anchor = min(y_to_px(u), y_to_px(t)) - 32
        draw.text((cx - dtw / 2, y_anchor), d_text, fill=d_color, font=f_delta)

        lw, lh = _text_size(draw, label, f_label)
        draw.text((cx - lw / 2, PAD_T + PLOT_H + 10),
                  label, fill=COLOR_TEXT, font=f_label)

    legend_y = HEIGHT - 28
    box = 14
    lx = PAD_L
    draw.rectangle([lx, legend_y, lx + box, legend_y + box],
                   fill=COLOR_U, outline=COLOR_U_EDGE)
    draw.text((lx + box + 6, legend_y - 1),
              "Untrained (zero-shot Qwen3-4B)", fill=COLOR_TEXT, font=f_label)
    lx2 = lx + 240
    draw.rectangle([lx2, legend_y, lx2 + box, legend_y + box],
                   fill=COLOR_T, outline=COLOR_T_EDGE)
    draw.text((lx2 + box + 6, legend_y - 1),
              "Trained (GRPO LoRA, iter 17)", fill=COLOR_TEXT, font=f_label)

    img.convert("RGB").save(out_path, "PNG")


def render_svg(out_path: Path) -> None:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Arial, sans-serif">'
    )
    parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>')
    parts.append(
        f'<text x="{WIDTH / 2}" y="34" text-anchor="middle" font-size="20" '
        f'fill="#282828">{escape("Trained vs untrained — held-out OpenEnv eval (5 seeds/task)")}</text>'
    )

    for i in range(6):
        v = i / 5.0
        y = y_to_px(v)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{WIDTH - PAD_R}" y2="{y:.1f}" '
            f'stroke="#dcdcdc" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#282828">{v:.1f}</text>'
        )

    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + PLOT_H}" '
        f'stroke="#3c3c3c" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + PLOT_H}" x2="{WIDTH - PAD_R}" '
        f'y2="{PAD_T + PLOT_H}" stroke="#3c3c3c" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="22" y="{PAD_T + PLOT_H / 2}" text-anchor="middle" font-size="14" '
        f'fill="#282828" transform="rotate(-90 22 {PAD_T + PLOT_H / 2})">Grader score (0–1)</text>'
    )

    for i, label in enumerate(LABELS):
        cx = PAD_L + GROUP_W * (i + 0.5)
        u, t = U[i], T[i]
        x_u = cx - BAR_W - GAP / 2
        x_t = cx + GAP / 2

        for x, v, fill, edge in [
            (x_u, u, "#aaaaaa", "#5a5a5a"),
            (x_t, t, "#d94a3d", "#7a1f17"),
        ]:
            y_top = y_to_px(v)
            h = (PAD_T + PLOT_H) - y_top
            parts.append(
                f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{BAR_W:.1f}" '
                f'height="{h:.1f}" fill="{fill}" stroke="{edge}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x + BAR_W / 2:.1f}" y="{y_top - 6:.1f}" '
                f'text-anchor="middle" font-size="12" fill="#282828">{v:.3f}</text>'
            )

        delta = t - u
        d_color = "#1b7a35" if delta >= 0 else "#a02020"
        y_anchor = min(y_to_px(u), y_to_px(t)) - 26
        parts.append(
            f'<text x="{cx:.1f}" y="{y_anchor:.1f}" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="{d_color}">Δ {delta:+.3f}</text>'
        )

        parts.append(
            f'<text x="{cx:.1f}" y="{PAD_T + PLOT_H + 24}" text-anchor="middle" '
            f'font-size="14" fill="#282828">{label}</text>'
        )

    legend_y = HEIGHT - 28
    parts.append(
        f'<rect x="{PAD_L}" y="{legend_y}" width="14" height="14" '
        f'fill="#aaaaaa" stroke="#5a5a5a"/>'
    )
    parts.append(
        f'<text x="{PAD_L + 22}" y="{legend_y + 11}" font-size="14" '
        f'fill="#282828">Untrained (zero-shot Qwen3-4B)</text>'
    )
    parts.append(
        f'<rect x="{PAD_L + 240}" y="{legend_y}" width="14" height="14" '
        f'fill="#d94a3d" stroke="#7a1f17"/>'
    )
    parts.append(
        f'<text x="{PAD_L + 262}" y="{legend_y + 11}" font-size="14" '
        f'fill="#282828">Trained (GRPO LoRA, iter 17)</text>'
    )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    out_png = ROOT / "eval_comparison.png"
    out_svg = ROOT / "eval_comparison.svg"
    render_png(out_png)
    render_svg(out_svg)
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
