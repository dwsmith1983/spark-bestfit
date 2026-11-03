"""Create or update Spark canary issues through the ``gh`` CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any, Literal

from .findings import CANARY_LABEL, Finding, is_passes_outside_title

Runner = Callable[[list[str]], str]
UpsertResult = Literal["created", "commented", "unchanged", "dry-run"]

LABEL_COLOR = "FBCA04"
LIST_LIMIT = "200"
FINGERPRINT_RE = re.compile(r"<!-- canary-fingerprint: ([0-9a-f]{64}) -->")


def gh(args: list[str]) -> str:
    """Run ``gh`` and return stdout; a non-zero exit raises with gh's stderr in the message."""
    try:
        return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gh {' '.join(args[:2])} failed ({exc.returncode}): {exc.stderr.strip()}") from exc


def _find_issue(title: str, state: str, run: Runner) -> dict[str, Any] | None:
    """The issue whose title equals ``title`` exactly, preferring an open one.

    Lists by label rather than ``--search``: the search index lags new issues, which would let a second
    run create a duplicate. Title matching happens here.
    """
    listed = json.loads(
        run(
            [
                "issue",
                "list",
                "--label",
                CANARY_LABEL,
                "--state",
                state,
                "--json",
                "number,title,state",
                "--limit",
                LIST_LIMIT,
            ]
        )
    )
    matches = [issue for issue in listed if issue["title"] == title]
    matches.sort(key=lambda issue: str(issue["state"]).upper() != "OPEN")
    return matches[0] if matches else None


def _login(entry: dict[str, Any]) -> str | None:
    return (entry.get("author") or {}).get("login")


def _newest_fingerprint(issue: dict[str, Any]) -> str | None:
    """Fingerprint at the very start of the newest text the canary wrote.

    Only comments posted by this token (``viewerDidAuthor``) count. The issue body counts when its author
    is a bot or is the same login as one of those comments, so a human can't silence the canary by pasting
    a marker.
    """
    own = [c for c in issue.get("comments") or [] if c.get("viewerDidAuthor") is True]
    own_logins = {_login(c) for c in own} - {None}
    author = issue.get("author") or {}
    texts = [c.get("body") or "" for c in own]
    if author.get("is_bot") is True or _login(issue) in own_logins:
        texts.insert(0, issue.get("body") or "")
    found = [match.group(1) for text in texts if (match := FINGERPRINT_RE.match(text))]
    return found[-1] if found else None


def _with_body_file(body: str, run: Runner, args: list[str]) -> None:
    fd, path = tempfile.mkstemp(prefix="canary-issue-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
        run([*args, "--body-file", path])
    finally:
        os.unlink(path)


def upsert(f: Finding, run: Runner, dry_run: bool = False) -> UpsertResult:
    """Create the issue, comment on the open one with the same title, or do nothing when nothing changed.

    Never closes or reopens issues. "Passes outside the declared range" titles are matched across all
    states, so once someone closes that issue it stays closed.
    """
    state = "all" if is_passes_outside_title(f.title) else "open"
    if dry_run:
        print(f"[dry-run] {f.title} (labels: {', '.join(f.labels)}; dedupe over {state} issues)")
        print(f.body)
        return "dry-run"
    run(["label", "create", CANARY_LABEL, "--color", LABEL_COLOR, "--force"])
    issue = _find_issue(f.title, state, run)
    if issue is None:
        _with_body_file(f.body, run, ["issue", "create", "--title", f.title, "--label", ",".join(f.labels)])
        return "created"
    if str(issue["state"]).upper() != "OPEN":
        return "unchanged"
    number = str(issue["number"])
    current = json.loads(run(["issue", "view", number, "--json", "body,author,comments"]))
    if _newest_fingerprint(current) == f.fingerprint:
        return "unchanged"
    _with_body_file(f.body, run, ["issue", "comment", number])
    return "commented"
