"""Load `.env` before anything reads os.environ.

Imported for its side effect by every entry point. The alternative -- telling
the reader to `export` three variables before each command -- is the kind of
setup step that works on the author's machine and fails on a grader's, and the
whole reproducibility section is about not doing that.

`override=False` so a variable already set in the real environment wins. That
is what makes the same code run under docker-compose, where the values arrive
as container environment rather than from a file that is deliberately not
committed.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
