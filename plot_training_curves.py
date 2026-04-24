#!/usr/bin/env python3
r"""Generate lightweight SVG training plots from GRPO JSONL logs.

This script has no plotting-library dependency. It reads the JSONL metrics
emitted by ``train_grpo.py`` and writes judge-friendly SVG plots plus a short
markdown summary that can be embedded into the README.

Typical use:

    .\.venv\Scripts\python.exe plot_training_curves.py
    .\.venv\Scripts\python.exe plot_training_curves.py ^
        --log grpo_wildfire/log.jsonl ^
        --out-dir submission_artifacts
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    color: str
    value_format: str = "{value:.3f}"


TASK_COLORS = {
    "easy": "#d9f99d",
    "medium": "#fde68a",
    "hard": "#fecaca",
}


def load_training_log(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "iter" in record:
            records.append(record)
    records.sort(key=lambda item: int(item["iter"]))
    if not records:
        raise ValueError(f"No JSONL training records found in {path}")
    return records


def _task_spans(records: list[dict]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    current_task = str(records[0]["task_id"])
    start_iter = int(records[0]["iter"])
    prev_iter = start_iter
    for record in records[1:]:
        task_id = str(record["task_id"])
        iteration = int(record["iter"])
        if task_id != current_task:
            spans.append((current_task, start_iter, prev_iter))
            current_task = task_id
            start_iter = iteration
        prev_iter = iteration
    spans.append((current_task, start_iter, prev_iter))
    return spans


def _value_bounds(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        pad = 1.0 if math.isclose(low, 0.0) else abs(low) * 0.1
        return low - pad, high + pad
    pad = (high - low) * 0.1
    return low - pad, high + pad


def _x_to_svg(value: int, x_min: int, x_max: int, plot_left: int, plot_width: int) -> float:
    if x_max == x_min:
        return plot_left + plot_width / 2.0
    return plot_left + ((value - x_min) / (x_max - x_min)) * plot_width


def _y_to_svg(value: float, y_min: float, y_max: float, top: int, height: int) -> float:
    if math.isclose(y_max, y_min):
        return top + height / 2.0
    return top + height - ((value - y_min) / (y_max - y_min)) * height


def _polyline_points(
    records: list[dict],
    spec: MetricSpec,
    x_min: int,
    x_max: int,
    y_min: float,
    y_max: float,
    plot_left: int,
    plot_width: int,
    top: int,
    height: int,
) -> str:
    points = []
    for record in records:
        x = _x_to_svg(int(record["iter"]), x_min, x_max, plot_left, plot_width)
        y = _y_to_svg(float(record[spec.key]), y_min, y_max, top, height)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_chart(
    records: list[dict],
    spec: MetricSpec,
    top: int,
    chart_height: int,
    width: int,
    plot_left: int,
    plot_width: int,
) -> str:
    x_values = [int(record["iter"]) for record in records]
    y_values = [float(record[spec.key]) for record in records]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = _value_bounds(y_values)

    pieces: list[str] = []
    plot_top = top + 28
    plot_height = chart_height - 54

    pieces.append(
        f'<text x="{plot_left}" y="{top + 14}" font-size="14" font-weight="700" '
        f'fill="#0f172a">{escape(spec.title)}</text>'
    )

    for task_id, start_iter, end_iter in _task_spans(records):
        x0 = _x_to_svg(start_iter, x_min, x_max, plot_left, plot_width)
        x1 = _x_to_svg(end_iter, x_min, x_max, plot_left, plot_width)
        span_width = max(1.0, x1 - x0)
        pieces.append(
            f'<rect x="{x0:.2f}" y="{plot_top}" width="{span_width:.2f}" '
            f'height="{plot_height}" fill="{TASK_COLORS.get(task_id, "#e2e8f0")}" '
            'fill-opacity="0.25" />'
        )

    pieces.append(
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" '
        'fill="white" stroke="#cbd5e1" />'
    )

    for tick in range(5):
        fraction = tick / 4.0
        y_value = y_min + (y_max - y_min) * (1.0 - fraction)
        y = plot_top + fraction * plot_height
        pieces.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}" '
            'stroke="#e2e8f0" stroke-dasharray="4 4" />'
        )
        pieces.append(
            f'<text x="{plot_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="10" '
            f'fill="#475569">{escape(spec.value_format.format(value=y_value))}</text>'
        )

    for record in records:
        x = _x_to_svg(int(record["iter"]), x_min, x_max, plot_left, plot_width)
        pieces.append(
            f'<line x1="{x:.2f}" y1="{plot_top + plot_height}" x2="{x:.2f}" y2="{plot_top + plot_height + 5}" '
            'stroke="#94a3b8" />'
        )

    pieces.append(
        f'<polyline points="{_polyline_points(records, spec, x_min, x_max, y_min, y_max, plot_left, plot_width, plot_top, plot_height)}" '
        f'fill="none" stroke="{spec.color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
    )

    for record in records:
        x = _x_to_svg(int(record["iter"]), x_min, x_max, plot_left, plot_width)
        y = _y_to_svg(float(record[spec.key]), y_min, y_max, plot_top, plot_height)
        pieces.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{spec.color}" />'
        )

    pieces.append(
        f'<text x="{plot_left + plot_width / 2:.2f}" y="{plot_top + plot_height + 24}" '
        'text-anchor="middle" font-size="11" fill="#475569">training iteration</text>'
    )

    legend_x = plot_left + plot_width - 160
    pieces.append(
        f'<rect x="{legend_x}" y="{top + 2}" width="152" height="18" fill="white" stroke="#e2e8f0" rx="4" />'
    )
    for idx, task_id in enumerate(("easy", "medium", "hard")):
        x = legend_x + 8 + idx * 48
        pieces.append(
            f'<rect x="{x}" y="{top + 7}" width="10" height="8" fill="{TASK_COLORS[task_id]}" />'
        )
        pieces.append(
            f'<text x="{x + 14}" y="{top + 14}" font-size="10" fill="#334155">{task_id}</text>'
        )

    return "\n".join(pieces)


def render_svg_dashboard(title: str, records: list[dict], specs: list[MetricSpec]) -> str:
    width = 920
    chart_height = 220
    height = 54 + len(specs) * chart_height
    plot_left = 84
    plot_width = width - 120

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc" />',
        (
            f'<text x="32" y="30" font-size="20" font-weight="700" fill="#0f172a">'
            f'{escape(title)}</text>'
        ),
        (
            '<text x="32" y="48" font-size="11" fill="#475569">'
            'Generated from train_grpo.py JSONL logs by plot_training_curves.py'
            '</text>'
        ),
    ]

    for idx, spec in enumerate(specs):
        top = 54 + idx * chart_height
        body.append(_render_chart(records, spec, top, chart_height, width, plot_left, plot_width))

    body.append("</svg>")
    return "\n".join(body)


def write_svg(path: Path, title: str, records: list[dict], specs: list[MetricSpec]) -> None:
    path.write_text(render_svg_dashboard(title, records, specs), encoding="utf-8")


def build_summary_markdown(records: list[dict]) -> str:
    best_by_task: dict[str, dict] = {}
    for record in records:
        task_id = str(record["task_id"])
        current = best_by_task.get(task_id)
        if current is None or float(record["mean_grader_score"]) > float(current["mean_grader_score"]):
            best_by_task[task_id] = record

    final_record = records[-1]
    lines = [
        "# Training Summary",
        "",
        f"- Iterations logged: `{len(records)}`",
        f"- Final iteration: `{final_record['iter']}` on task `{final_record['task_id']}`",
        f"- Final mean return: `{float(final_record['mean_return']):.3f}`",
        f"- Final mean grader: `{float(final_record['mean_grader_score']):.3f}`",
        f"- Final parse success rate: `{100 * float(final_record['action_parse_success_rate']):.1f}%`",
        "",
        "## Best Mean Grader By Task",
        "",
        "| Task | Iter | Mean grader | Mean return | Parse success |",
        "|---|---:|---:|---:|---:|",
    ]
    for task_id in ("easy", "medium", "hard"):
        record = best_by_task.get(task_id)
        if record is None:
            continue
        lines.append(
            "| "
            f"{task_id} | {record['iter']} | {float(record['mean_grader_score']):.3f} | "
            f"{float(record['mean_return']):.3f} | {100 * float(record['action_parse_success_rate']):.1f}% |"
        )

    return "\n".join(lines) + "\n"


def write_summary(path: Path, records: list[dict]) -> None:
    path.write_text(build_summary_markdown(records), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG training plots from GRPO JSONL logs.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("grpo_wildfire") / "log.jsonl",
        help="Path to train_grpo.py JSONL metrics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("submission_artifacts"),
        help="Directory for SVG plots and summary markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log.exists():
        raise FileNotFoundError(
            f"Training log not found at {args.log}. Run train_grpo.py first or pass --log."
        )

    records = load_training_log(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reward_specs = [
        MetricSpec("mean_return", "Mean Trajectory Return", "#2563eb"),
        MetricSpec("mean_grader_score", "Mean Final Grader Score", "#16a34a"),
    ]
    loss_specs = [
        MetricSpec("policy_loss", "Policy Loss", "#dc2626"),
        MetricSpec("kl_divergence", "KL Divergence", "#7c3aed"),
        MetricSpec("clip_fraction", "Clip Fraction", "#ea580c"),
    ]

    write_svg(args.out_dir / "training_reward_curve.svg", "Wildfire GRPO Reward Curves", records, reward_specs)
    write_svg(args.out_dir / "training_loss_curve.svg", "Wildfire GRPO Optimisation Curves", records, loss_specs)
    write_summary(args.out_dir / "training_summary.md", records)

    print(f"Wrote {args.out_dir / 'training_reward_curve.svg'}")
    print(f"Wrote {args.out_dir / 'training_loss_curve.svg'}")
    print(f"Wrote {args.out_dir / 'training_summary.md'}")


if __name__ == "__main__":
    main()
