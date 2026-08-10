"""Bridge Streamlit Cloud's `st.secrets` into `os.environ`.

Everything below `app/` reads configuration from the environment, because that
is what works under docker-compose, in the Makefile, and in the offline
evaluation scripts. Streamlit Community Cloud does not set environment
variables; it mounts a TOML file and exposes it as `st.secrets`. Without a
bridge the deployed app would fall back to every local default at once --
`localhost:6333`, `localhost:5432` -- and fail with a connection error that
says nothing about the actual cause, which is that the secrets were never read.

Copying rather than rewriting the call sites is deliberate. One file knows that
Streamlit exists; the rag package stays deployable anywhere that can set
environment variables.

`setdefault`, not assignment: a real environment variable wins over a secret of
the same name, so a container that injects `POSTGRES_DSN` is not overridden by
a stale value in a committed-by-accident secrets file.

Import this *before* `app.db` or `rag.env`. It is not a no-op at import time.
"""

from __future__ import annotations

import os

# Only these are bridged. Streamlit's secrets file may hold nested sections and
# values that are not strings, and blindly flooding os.environ with them makes
# a debugging session start with "what else is in here".
BRIDGED = (
    "ANTHROPIC_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "POSTGRES_DSN",
    "GENERATE_MODEL",
)


def load() -> list[str]:
    """Copy known secrets into os.environ. Returns the names that were set."""
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        # No secrets file, or not running under Streamlit at all. Both are
        # normal: the local path gets its configuration from `.env`.
        return []

    loaded = []
    for name in BRIDGED:
        try:
            value = secrets[name]
        except Exception:
            continue
        if value is None or value == "":
            continue
        if name not in os.environ:
            os.environ[name] = str(value)
            loaded.append(name)
    return loaded


LOADED = load()
