from __future__ import annotations

import os
from typing import Iterable

try:
    from termcolor import colored as _term_colored
except Exception:  # pragma: no cover
    _term_colored = None


def ctext(text: object, color: str | None = None, attrs: Iterable[str] | None = None) -> str:
    message = str(text)
    if _term_colored is None:
        return message
    return _term_colored(message, color=color, attrs=list(attrs) if attrs else None)


def _tag(level: str, color: str, attrs: Iterable[str] | None = None) -> str:
    return ctext(f"[{level}]", color=color, attrs=attrs or ("bold",))


def _is_rank0_or_single_process() -> bool:
    world_size_raw = os.getenv("WORLD_SIZE")
    if world_size_raw is None:
        return True
    try:
        world_size = int(world_size_raw)
    except ValueError:
        return True
    if world_size <= 1:
        return True
    rank_raw = os.getenv("RANK")
    if rank_raw is None:
        return True
    try:
        return int(rank_raw) == 0
    except ValueError:
        return True


def rank0_print(*args, **kwargs) -> None:
    if _is_rank0_or_single_process():
        print(*args, **kwargs)


def log_info(message: object) -> None:
    if _is_rank0_or_single_process():
        print(f"{_tag('INFO', 'cyan')} {message}")


def log_success(message: object) -> None:
    if _is_rank0_or_single_process():
        print(f"{_tag('OK', 'green')} {message}")


def log_warn(message: object) -> None:
    if _is_rank0_or_single_process():
        print(f"{_tag('WARN', 'yellow')} {message}")


def log_error(message: object) -> None:
    if _is_rank0_or_single_process():
        print(f"{_tag('ERR', 'red')} {message}")


def log_debug(message: object) -> None:
    if _is_rank0_or_single_process():
        print(f"{_tag('DBG', 'magenta')} {message}")


def progress_color(step: int, total: int) -> str:
    if total <= 0:
        return ctext(str(step), "cyan", attrs=("bold",))
    ratio = max(0.0, min(1.0, step / total))
    if ratio < 0.25:
        color = "red"
    elif ratio < 0.5:
        color = "yellow"
    elif ratio < 0.9:
        color = "cyan"
    else:
        color = "green"
    return ctext(f"{step}/{total}", color=color, attrs=("bold",))


def metric_color(value: float, *, lower_is_better: bool, good: float, bad: float) -> str:
    if lower_is_better:
        if value <= good:
            color = "green"
        elif value >= bad:
            color = "red"
        else:
            color = "yellow"
    else:
        if value >= good:
            color = "green"
        elif value <= bad:
            color = "red"
        else:
            color = "yellow"
    return ctext(f"{value:.4f}", color=color, attrs=("bold",))
