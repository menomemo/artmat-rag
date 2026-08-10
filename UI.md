# Changing the interface yourself

Everything visible is in one file, [`app/main.py`](app/main.py), about 300 lines
including comments. There is no template, no component library, no build step.
You edit Python, save, and the page redraws.

## The loop

```bash
make ui          # http://localhost:8501
```

Leave it running. Every time you save `app/main.py`, Streamlit notices and the
browser offers **Rerun** (top right) — or turn on *Always rerun* in the same
menu and it just redraws on save. A syntax error shows up as a red traceback on
the page instead of a crash, so you can fix it and save again without
restarting.

One exception, and it will confuse you once: `make ui` must be restarted if you
edit anything under `rag/` or `app/db.py`. Streamlit reloads the script but not
modules it has already imported, so a changed prompt in `rag/generate.py` will
appear to have no effect. Restart, then it works.

## What is where

| Lines | What |
|---|---|
| 51–57 | `EXAMPLES` — the five buttons across the top |
| 59–64 | `LAYER_LABEL` — how the four source layers are named on screen |
| 68–107 | `bootstrap` / `warm` — connections and model warm-up. Not display. |
| 110–178 | `retrieve` / `finish_query` — search and logging. Not display. |
| 181–222 | the sidebar: blurb, three settings, layer legend |
| 224–245 | title, caption, example buttons, question box |
| 247–284 | the **Ask** branch: search, stream the answer, log, rerun |
| 286–end | rendering the finished result: answer, cost line, feedback, passages |

The split at line 247 is the thing to understand before editing. Everything
above it draws on *every* page load. The `if st.button("Ask")` block runs only
on the load where the button was clicked, and it ends with `st.rerun()` — so the
answer you see was drawn by the block at the bottom, from `st.session_state`,
not by the block that produced it.

## Making it less crowded

The most likely first change. Three independent cuts:

**Drop the sidebar blurb** — delete lines 184–191 (`st.header("What this is")`
and the `st.markdown` under it). The settings and legend stay.

**Fewer example buttons** — delete entries from `EXAMPLES` (line 51). The row
sizes itself; `st.columns(len(EXAMPLES))` adapts.

**Collapse the sidebar by default** — line 49:

```python
st.set_page_config(
    page_title="artmat — materials for making",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

**Narrower text.** `layout="wide"` makes answers run the full window width,
which is hard to read at 27 inches. Either drop `layout="wide"` entirely, or
keep it and put the answer in a column:

```python
answer_col, _ = st.columns([3, 2])
with answer_col:
    st.markdown(answer.text)
```

## Colours and fonts

Not in Python — [`.streamlit/config.toml`](.streamlit/config.toml). Editing it
requires a restart of `make ui`, not just a rerun.

```toml
[theme]
base = "light"
primaryColor = "#1f4e5f"          # buttons, focus rings, links
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f4f1ec"   # sidebar and expander fills
textColor = "#1a1a1a"
font = "serif"                    # "sans serif" | "serif" | "monospace"
```

This file is committed, so the deployed app looks the same as yours. A theme
clicked into a hosting console would not survive a redeploy.

## Common edits

**Change the wording.** Title and strapline are lines 226–230; the sidebar
explanation is 184–191; every `help=` string is a tooltip. All plain strings.

**Change the defaults.** Line 201, `st.slider("Passages retrieved", 4, 16, 8)` —
the three numbers are min, max, default. Line 197 chooses which answer style is
pre-selected.

**Show the passages before the answer.** Move the whole `st.subheader("Passages
this answer was built from")` block (line 313 to the end of the file) above
`st.markdown(answer.text)` at line 295. Nothing depends on the order; both read
from `result`.

**Add a source-layer filter.** `rag/search.py` already supports it — `search()`
takes `source_types`. In the sidebar:

```python
layers = st.multiselect(
    "Search only these layers",
    list(LAYER_LABEL), default=list(LAYER_LABEL),
    format_func=LAYER_LABEL.get,
)
```

then pass `source_types=layers` through `retrieve()` into `search(...)`. Worth
knowing what you are doing to the results: filtering to one layer removes the
disagreement the whole system is built to show, which is occasionally what you
want and usually not.

**Add a new answer style.** Do not touch `main.py`. Add an entry to `PROMPTS` in
[`rag/generate.py`](rag/generate.py) and it appears in the dropdown by itself —
the selectbox is built from `PROMPTS.keys()`. The one exclusion is
`no_context`, filtered out at line 196 because it is an evaluation control, not
something to offer a user. Restart `make ui` afterwards (see above).

## Two traps

**Widgets need stable keys.** If you add two buttons with the same label,
Streamlit raises a duplicate-ID error. Give them `key="something_unique"`.

**Anything you want to survive a click must live in `st.session_state`.** Local
variables are gone on the next rerun — which happens on *every* interaction,
including moving the slider. This is why the answer is stored in
`st.session_state.result` rather than just drawn. If you add state, follow the
same shape: initialise it near line 232, read it near line 286.

## After you have changed it

```bash
git add -u && git commit -m "ui: ..." && git push
```

Streamlit Community Cloud watches the branch and redeploys within a minute or
two. Nothing to click. If the app comes back broken, **Manage app → logs** in
the bottom-right of the deployed page shows the traceback.
