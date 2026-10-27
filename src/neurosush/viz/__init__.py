"""Atlas visualization engine: traces, step schedules, and self-contained HTML plates."""

from __future__ import annotations

from neurosush.viz.export import available_plates, render_plate, write_html
from neurosush.viz.schedule import schedule_lanes, schedule_table
from neurosush.viz.trace import Trace, TraceRecorder, pack_array, unpack_array

__all__ = [
    "Trace",
    "TraceRecorder",
    "available_plates",
    "pack_array",
    "render_plate",
    "schedule_lanes",
    "schedule_table",
    "unpack_array",
    "write_html",
]
