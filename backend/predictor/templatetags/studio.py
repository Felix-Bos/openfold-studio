"""Template filters for OpenFold Studio pages."""

from datetime import timedelta

from django import template

register = template.Library()


@register.filter
def duration(value) -> str:
    """timedelta or seconds -> "8m 12s" / "1h 04m" / "—"."""
    if value is None or value == "":
        return "—"
    seconds = int(value.total_seconds() if isinstance(value, timedelta) else float(value))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@register.filter
def percent(value, digits: int = 0) -> str:
    """0.873 -> "87%"."""
    if value is None:
        return "—"
    return f"{float(value) * 100:.{int(digits)}f}%"
