"""Self-contained HTML exporter for neuroSush visualization plates."""

from __future__ import annotations

import html
import importlib.resources
import json
from pathlib import Path
from typing import Any


def available_plates() -> list[str]:
    """Return sorted list of available plate IDs found in the package.

    Returns:
        List of plate identifier strings.
    """
    plates_dir = importlib.resources.files("neurosush.viz.web.plates")
    plates: list[str] = []
    for item in plates_dir.iterdir():
        if item.name.endswith(".js") and not item.name.startswith("__"):
            plates.append(item.name[:-3])
    return sorted(plates)


def _escape_script_json(obj: Any) -> str:
    """Serialize an object to JSON safe for embedding inside an HTML script tag.

    Args:
        obj: Serializable object.

    Returns:
        JSON string safe against </script> injection and line separator issues.
    """
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    # Escape </ to prevent breaking out of script tags; escape U+2028/U+2029 for JS compatibility
    return (
        raw.replace("</", "<\\u002f")
        .replace(chr(0x2028), "\\u2028")
        .replace(chr(0x2029), "\\u2029")
    )


def render_plate(
    plate_id: str,
    trace_dict: dict[str, Any],
    *,
    title: str | None = None,
    source: str = "replay",
    notes: dict[str, Any] | None = None,
    theme: str = "auto",
) -> str:
    """Render a self-contained HTML document for a visualization plate.

    Args:
        plate_id: Identifier of the plate (must match a file in web/plates/<id>.js).
        trace_dict: Trace data dictionary containing steps, series, and meta.
        title: Optional page title (2-4 words). Defaults to "neuroSush · <PlateId>".
        source: Data provenance mode ('live', 'replay', or 'sweep').
        notes: Optional dictionary of margin notes or metadata.
        theme: Initial theme mode ('auto', 'light', or 'dark').

    Returns:
        Self-contained HTML string with inlined styles, scripts, and trace data.

    Raises:
        FileNotFoundError: If plate_id is not in available_plates().
        ValueError: If source or theme is outside its documented set.
    """
    valid_themes = ("auto", "light", "dark")
    if theme not in valid_themes:
        raise ValueError(f"theme must be one of {valid_themes}, got {theme!r}")

    valid_sources = ("live", "replay", "sweep")
    if source not in valid_sources:
        raise ValueError(f"source must be one of {valid_sources}, got {source!r}")

    known_plates = available_plates()
    if plate_id not in known_plates:
        raise FileNotFoundError(f"Plate '{plate_id}' not found. Available plates: {known_plates}")

    web_res = importlib.resources.files("neurosush.viz.web")
    css_content = web_res.joinpath("atlas.css").read_text(encoding="utf-8")
    js_content = web_res.joinpath("atlas.js").read_text(encoding="utf-8")
    plate_js = web_res.joinpath(f"plates/{plate_id}.js").read_text(encoding="utf-8")

    strings_json = web_res.joinpath("i18n/en.json").read_text(encoding="utf-8")
    strings_data = json.loads(strings_json)

    theme_attr = (
        f' data-theme="{html.escape(theme, quote=True)}"' if theme in ("light", "dark") else ""
    )

    default_title = f"neuroSush · {plate_id.title()}"
    page_title = title if title is not None else default_title
    escaped_title = html.escape(page_title, quote=True)

    escaped_trace_json = _escape_script_json(trace_dict)
    escaped_strings_json = _escape_script_json(strings_data)
    escaped_notes_json = _escape_script_json(notes if notes is not None else {})

    is_live = bool(source == "live" or trace_dict.get("meta", {}).get("live"))
    root_live_attr = ' data-live="true"' if is_live else ""

    bootstrap_spec = {
        "plateId": plate_id,
        "source": source,
        "theme": theme,
        "live": is_live,
    }
    escaped_spec_json = _escape_script_json(bootstrap_spec)

    return f"""<!DOCTYPE html>
<html lang="en" dir="ltr"{theme_attr}>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escaped_title}</title>
  <style>
{css_content}
  </style>
</head>
<body>
  <div id="atlas-root"{root_live_attr}></div>
  <script type="application/json" id="atlas-strings">{escaped_strings_json}</script>
  <script type="application/json" id="atlas-trace">{escaped_trace_json}</script>
  <script type="application/json" id="atlas-notes">{escaped_notes_json}</script>
  <script>
{js_content}
  </script>
  <script>
{plate_js}
  </script>
  <script>
    (function() {{
      const strings = JSON.parse(document.getElementById('atlas-strings').textContent);
      const trace = JSON.parse(document.getElementById('atlas-trace').textContent);
      const notes = JSON.parse(document.getElementById('atlas-notes').textContent);
      const spec = {escaped_spec_json};
      spec.trace = trace;
      spec.notes = notes;
      spec.root = document.getElementById('atlas-root');
      Atlas.setStrings(strings);
      Atlas.start(spec);
    }})();
  </script>
</body>
</html>
"""


def write_html(
    path: str | Path,
    plate_id: str,
    trace_dict: dict[str, Any],
    *,
    title: str | None = None,
    source: str = "replay",
    notes: dict[str, Any] | None = None,
    theme: str = "auto",
) -> Path:
    """Render and write a self-contained plate HTML file to disk.

    Args:
        path: Output file path.
        plate_id: Identifier of the plate.
        trace_dict: Trace data dictionary.
        title: Optional page title.
        source: Data provenance mode.
        notes: Optional margin notes.
        theme: Initial theme mode ('auto', 'light', or 'dark').

    Returns:
        Path object pointing to the written file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_plate(
        plate_id=plate_id,
        trace_dict=trace_dict,
        title=title,
        source=source,
        notes=notes,
        theme=theme,
    )
    out_path.write_text(content, encoding="utf-8")
    return out_path
