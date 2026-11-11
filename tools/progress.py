"""A self-report progress board for the course.

You tick the sections you have worked through and write notes; this renders the
result as a track you can read at a glance.

The section list is not stored anywhere. It is parsed from the module articles
each time, for the same reason the figures are generated from the numbers the
articles quote: a hand-maintained copy drifts, and a derived one cannot.

Your ticks and notes live in ``progress.json`` at the course root, so they
survive restarts and are the only thing you own.

Usage from a notebook::

    from tools import progress
    progress.checklist()     # tick boxes and write notes
    progress.render()        # draw the board

Usage from a shell::

    python tools/progress.py            # print a text summary
    python tools/progress.py --html out.html
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "progress.json"

# Status palette. The first two are the reserved status steps (good, warning);
# "partial" borrows categorical slot 1, and "none" is a deliberate neutral.
# Every state is drawn with a dot AND a label, so colour never carries meaning
# on its own -- the mitigation that lets warning be used on a dark surface.
STATUSES = {
    "done":     ("Completed",         "#0ca30c"),
    "active":   ("In progress",       "#fab219"),
    "partial":  ("Partially covered", "#3987e5"),
    "none":     ("Not started",       "#6b6a65"),
}

# Dark surface, selected rather than flipped: the board carries its own
# background so it looks the same in a light or a dark notebook theme.
SURFACE_0 = "#141413"
SURFACE_1 = "#1a1a19"
SURFACE_2 = "#232322"
INK = "#f0efec"
INK_DIM = "#c3c2b7"
INK_MUTED = "#8a8982"
BORDER = "rgba(255,255,255,0.10)"


@dataclass
class Module:
    key: str                      # "04"
    folder: str                   # "04-ml-system-design"
    title: str                    # "Planning the Training of a Very Large Model"
    blurb: str                    # the italic line under the heading
    items: list[tuple[str, str]] = field(default_factory=list)  # (key, label)


def _article(folder: Path) -> Path | None:
    """The v2 article if there is one, else the original."""
    for name in ("README-v2.md", "README.md"):
        if (folder / name).exists():
            return folder / name
    return None


def discover() -> list[Module]:
    """Read the module articles and build the checklist from their sections."""
    modules: list[Module] = []
    for folder in sorted(ROOT.glob("[0-9][0-9]-*")):
        if not folder.is_dir():
            continue
        path = _article(folder)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")

        heading = re.search(r"^#\s+(.*)$", text, re.M)
        title = heading.group(1) if heading else folder.name
        # Articles are titled "04 - Planning ..."; keep only the part after the dash.
        title = re.sub(r"^\d+\s*[-–—]\s*", "", title).strip()

        # The blurb is the first italic paragraph under the heading, and only
        # that one: a lazy match without an anchored end runs on to the next
        # italic in the file, which is the reading-time line or worse.
        blurb = ""
        m = re.search(r"^#[^\n]*\n+\*(.+?)\*[ \t]*\n[ \t]*\n", text, re.S)
        if m:
            blurb = " ".join(m.group(1).split())

        items = [(num, f"{num}. {name.strip()}")
                 for num, name in re.findall(r"^##\s+(\d+)\.\s+(.*)$", text, re.M)]
        if (folder / "exercise.py").exists():
            items.append(("exercise", "The exercise"))

        modules.append(Module(folder.name[:2], folder.name, title, blurb, items))
    return modules


def load() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"done": {}, "status": {}, "notes": {}}


def save(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def status_of(mod: Module, state: dict) -> str:
    """Derived from the ticks, unless you have overridden it by hand.

    Derivation cannot tell "in progress" from "partially covered" -- both are
    some-but-not-all -- so that distinction is yours to make, and an override
    is how you make it.
    """
    override = state.get("status", {}).get(mod.key)
    if override in STATUSES:
        return override
    done = set(state.get("done", {}).get(mod.key, []))
    hit = len([k for k, _ in mod.items if k in done])
    if not mod.items or hit == 0:
        return "none"
    return "done" if hit == len(mod.items) else "partial"


def totals(modules: list[Module], state: dict) -> dict:
    done_map = state.get("done", {})
    ticked = total = 0
    counts = {k: 0 for k in STATUSES}
    for mod in modules:
        d = set(done_map.get(mod.key, []))
        ticked += len([k for k, _ in mod.items if k in d])
        total += len(mod.items)
        counts[status_of(mod, state)] += 1
    return {"ticked": ticked, "total": total,
            "pct": (100.0 * ticked / total) if total else 0.0,
            "counts": counts}


# ----------------------------------------------------------------- rendering
def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _clip(text: str, n: int) -> str:
    """Trim to a word boundary. A hard slice cuts mid-word, which looks broken."""
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"


def _column(mod: Module, state: dict) -> str:
    st = status_of(mod, state)
    label, colour = STATUSES[st]
    done = set(state.get("done", {}).get(mod.key, []))

    rows = []
    for key, text in mod.items:
        on = key in done
        mark = "&#10003;" if on else "&#9675;"      # check, or hollow circle
        col = colour if on else INK_MUTED
        weight = "500" if on else "400"
        rows.append(
            f'<li style="display:flex;gap:7px;align-items:flex-start;'
            f'margin:0 0 5px 0;color:{INK_DIM if on else INK_MUTED};'
            f'font-weight:{weight}">'
            f'<span aria-hidden="true" style="color:{col};flex:none;'
            f'line-height:1.35">{mark}</span>'
            f'<span>{_esc(text)}</span></li>')

    note = state.get("notes", {}).get(mod.key, "").strip()
    note_html = ""
    if note:
        note_html = (
            f'<div style="margin-top:10px;padding-top:9px;'
            f'border-top:1px solid {BORDER};color:{INK_MUTED};'
            f'font-size:11.5px;line-height:1.45;white-space:pre-wrap">'
            f'{_esc(note)}</div>')

    return f"""
<div style="flex:0 0 212px;background:{SURFACE_1};border:1px solid {BORDER};
     border-radius:10px;padding:13px 13px 14px;display:flex;flex-direction:column">
  <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px">
    <span style="flex:none;width:26px;height:26px;border-radius:50%;
          background:{colour};color:#0b0b0b;font-weight:700;font-size:12.5px;
          display:flex;align-items:center;justify-content:center">{_esc(mod.key)}</span>
    <span style="color:{INK};font-weight:650;font-size:13px;line-height:1.25">
      {_esc(mod.title)}</span>
  </div>
  <div style="color:{INK_MUTED};font-size:11px;line-height:1.45;margin-bottom:10px;
       min-height:46px">{_esc(_clip(mod.blurb, 135))}</div>
  <div style="background:{colour};color:#0b0b0b;font-weight:650;font-size:11.5px;
       text-align:center;border-radius:6px;padding:5px 0;margin-bottom:11px">
    {_esc(label)}</div>
  <ul style="list-style:none;margin:0;padding:0;font-size:11.5px;line-height:1.35">
    {''.join(rows)}
  </ul>{note_html}
</div>"""


def _bar(modules: list[Module], state: dict, t: dict) -> str:
    """One stacked bar: section ticks grouped by their module's status.

    A 2px surface gap separates adjacent fills, so the segments read as
    separate quantities rather than one smeared ribbon.
    """
    done_map = state.get("done", {})
    seg = {k: 0 for k in STATUSES}
    for mod in modules:
        d = set(done_map.get(mod.key, []))
        seg[status_of(mod, state)] += len([k for k, _ in mod.items if k in d])
    total = t["total"] or 1

    parts = []
    for key in ("done", "active", "partial", "none"):
        n = seg[key]
        if not n:
            continue
        parts.append(
            f'<div title="{_esc(STATUSES[key][0])}: {n} sections" '
            f'style="width:{100.0 * n / total:.3f}%;background:{STATUSES[key][1]};'
            f'border-radius:3px"></div>')
    # Everything not ticked, in the same grey as the "Not started" legend chip,
    # so the segment reads as a tracked quantity rather than as empty background.
    rest = total - sum(seg.values())
    if rest > 0:
        parts.append(f'<div title="Not ticked: {rest} sections" '
                     f'style="width:{100.0 * rest / total:.3f}%;'
                     f'background:{STATUSES["none"][1]};border-radius:3px"></div>')
    return (f'<div style="display:flex;gap:2px;height:17px;margin:9px 0 7px">'
            f'{"".join(parts)}</div>')


def _legend() -> str:
    chips = []
    for key, (label, colour) in STATUSES.items():
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'color:{INK_DIM};font-size:11.5px">'
            f'<span aria-hidden="true" style="width:10px;height:10px;border-radius:50%;'
            f'background:{colour};display:inline-block"></span>{_esc(label)}</span>')
    return ('<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">'
            + "".join(chips) + "</div>")


def _panel(title: str, body: str) -> str:
    return f"""
<div style="flex:1 1 260px;background:{SURFACE_1};border:1px solid {BORDER};
     border-radius:10px;padding:13px 15px">
  <div style="color:{INK};font-weight:650;font-size:12.5px;margin-bottom:9px">
    {_esc(title)}</div>
  {body}
</div>"""


def build_html(modules=None, state=None) -> str:
    modules = modules if modules is not None else discover()
    state = state if state is not None else load()
    t = totals(modules, state)

    columns = "".join(_column(m, state) for m in modules)

    counts = []
    for key, (label, colour) in STATUSES.items():
        counts.append(
            f'<li style="display:flex;align-items:center;gap:8px;margin-bottom:6px;'
            f'color:{INK_DIM};font-size:12px">'
            f'<span aria-hidden="true" style="width:10px;height:10px;border-radius:50%;'
            f'background:{colour};flex:none"></span>'
            f'<span style="flex:1">{_esc(label)}</span>'
            f'<span style="color:{INK};font-weight:650">{t["counts"][key]}</span></li>')

    left = [f"{m.key} {m.title}" for m in modules if status_of(m, state) != "done"]
    left_html = "".join(
        f'<li style="color:{INK_DIM};font-size:11.5px;margin-bottom:4px">'
        f'{_esc(x)}</li>' for x in left[:12]) or \
        f'<li style="color:{INK_DIM};font-size:11.5px">Nothing. The course is done.</li>'

    note = state.get("notes", {}).get("_global", "").strip()
    note_html = (f'<div style="color:{INK_DIM};font-size:11.5px;line-height:1.5;'
                 f'white-space:pre-wrap">{_esc(note)}</div>') if note else \
                (f'<div style="color:{INK_MUTED};font-size:11.5px">'
                 f'No notes yet.</div>')

    return f"""
<div style="background:{SURFACE_0};border-radius:14px;padding:20px 22px 22px;
     font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:{INK}">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;
       gap:20px;flex-wrap:wrap;margin-bottom:16px">
    <div>
      <div style="font-size:23px;font-weight:700;letter-spacing:-0.3px">
        Course progress</div>
      <div style="color:{INK_MUTED};font-size:12.5px;margin-top:3px">
        Self-reported, {len(modules)} modules, {t['total']} sections</div>
    </div>
    {_legend()}
  </div>

  <div style="display:flex;gap:11px;overflow-x:auto;padding-bottom:8px;
       margin-bottom:15px">{columns}</div>

  <div style="display:flex;gap:11px;flex-wrap:wrap">
    {_panel("Overall progress", f'''
      {_bar(modules, state, t)}
      <div style="display:flex;align-items:baseline;gap:9px">
        <span style="font-size:30px;font-weight:700;letter-spacing:-0.5px">
          {t['pct']:.0f}%</span>
        <span style="color:{INK_MUTED};font-size:12px">
          {t['ticked']} of {t['total']} sections</span>
      </div>''')}
    {_panel("Segment status", f'<ul style="list-style:none;margin:0;padding:0">'
                              f'{"".join(counts)}</ul>')}
    {_panel("What is left", f'<ul style="list-style:none;margin:0;padding:0">'
                            f'{left_html}</ul>')}
    {_panel("Notes", note_html)}
  </div>
</div>"""


def render(modules=None, state=None):
    """Draw the board in a notebook."""
    from IPython.display import HTML, display
    return display(HTML(build_html(modules, state)))


# ---------------------------------------------------------------- the ticks
def checklist(modules=None):
    """Tick boxes and write notes. Every change is saved immediately."""
    import ipywidgets as w
    from IPython.display import display

    modules = modules if modules is not None else discover()
    state = load()
    state.setdefault("done", {})
    state.setdefault("status", {})
    state.setdefault("notes", {})

    def on_tick(mod_key, item_key):
        def handler(change):
            done = set(state["done"].get(mod_key, []))
            done.add(item_key) if change["new"] else done.discard(item_key)
            state["done"][mod_key] = sorted(done)
            save(state)
        return handler

    def on_status(mod_key):
        def handler(change):
            if change["new"] == "auto":
                state["status"].pop(mod_key, None)
            else:
                state["status"][mod_key] = change["new"]
            save(state)
        return handler

    def on_note(mod_key):
        def handler(change):
            state["notes"][mod_key] = change["new"]
            save(state)
        return handler

    tabs = []
    for mod in modules:
        done = set(state["done"].get(mod.key, []))
        boxes = []
        for key, label in mod.items:
            cb = w.Checkbox(value=key in done, description=label,
                            indent=False, layout=w.Layout(width="99%"))
            cb.observe(on_tick(mod.key, key), names="value")
            boxes.append(cb)

        dd = w.Dropdown(
            options=[("auto (from ticks)", "auto")] +
                    [(lbl, k) for k, (lbl, _) in STATUSES.items()],
            value=state["status"].get(mod.key, "auto"),
            description="Status:", layout=w.Layout(width="330px"))
        dd.observe(on_status(mod.key), names="value")

        note = w.Textarea(value=state["notes"].get(mod.key, ""),
                          placeholder="Notes for this module",
                          layout=w.Layout(width="99%", height="72px"))
        note.observe(on_note(mod.key), names="value")

        tabs.append(w.VBox([dd, w.VBox(boxes), note]))

    glob = w.Textarea(value=state["notes"].get("_global", ""),
                      placeholder="Notes shown on the board",
                      layout=w.Layout(width="99%", height="72px"))
    glob.observe(on_note("_global"), names="value")
    tabs.append(w.VBox([glob]))

    acc = w.Tab(children=tabs)
    for i, mod in enumerate(modules):
        acc.set_title(i, mod.key)
    acc.set_title(len(modules), "Notes")
    display(acc)
    return acc


def main(argv: list[str]) -> int:
    modules = discover()
    state = load()
    if "--html" in argv:
        out = Path(argv[argv.index("--html") + 1])
        out.write_text(build_html(modules, state), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    t = totals(modules, state)
    for mod in modules:
        done = set(state.get("done", {}).get(mod.key, []))
        hit = len([k for k, _ in mod.items if k in done])
        print(f"  {mod.key}  {STATUSES[status_of(mod, state)][0]:<18s}"
              f"{hit}/{len(mod.items)}  {mod.title}")
    print(f"\n  {t['pct']:.0f}% - {t['ticked']} of {t['total']} sections")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
