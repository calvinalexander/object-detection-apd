"""Bulk labelling through `claude -p`.

The Messages API rejects this machine's subscription OAuth token with a bare 429,
so the CLI is the working path. That shapes the batching: each `claude -p`
invocation re-sends Claude Code's own system prompt, so invocation count is what
costs, not sheet size -- the opposite of the SDK, where the batch is the thing
that caches. Sheets therefore want to be as large as they can be while every crop
stays legible; measurement put that at about 12 at 300 px, since a denser 24-up
sheet at 240 px measurably increased abstention.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .schema import ATTRIBUTE_RULE, Value

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")

#: Where to look for a subscription token when the environment does not carry one.
#: `claude -p` reads CLAUDE_CODE_OAUTH_TOKEN from its environment and only falls
#: back to ~/.claude/.credentials.json -- which on this machine holds an empty
#: accessToken, so a bare subprocess fails with "OAuth session expired and could
#: not be refreshed" while the interactive session keeps working.
TOKEN_FILES = (Path.home() / "tryan/test/.env",)


def resolve_token() -> str | None:
    """Subscription OAuth token from the environment, else from a known .env."""
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return token
    for path in TOKEN_FILES:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None

SYSTEM_PROMPT = """\
You label personal protective equipment in CCTV crops from commercial kitchens.

You will be given one contact sheet image containing numbered crops. Judge each
numbered crop independently and return one verdict per slot.

{rule}

Answer `unclear` whenever the evidence is not there -- the crop is too blurred,
the relevant body part is turned away or occluded, or you cannot tell which
person the slot refers to. `unclear` is a correct and expected answer, not a
failure; guessing instead of abstaining injects noise that is worse than a
missing label.

Return ONLY a JSON object mapping slot number (as a string) to one of
"present", "absent", "unclear". No prose, no code fence, no commentary.
Example for a 3-slot sheet: {{"1":"present","2":"unclear","3":"absent"}}
"""


@dataclass
class SheetResult:
    verdicts: dict[str, str]
    cost_usd: float | None
    error: str | None = None


def render_sheet(crops: list[Path], out: Path, cell: int = 300, cols: int = 4) -> None:
    """Contact sheet with a numbered banner per crop."""
    rows = (len(crops) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 24)), (233, 237, 238))
    draw = ImageDraw.Draw(sheet)
    for n, path in enumerate(crops):
        with Image.open(path) as im:
            tile = im.convert("RGB").resize((cell, cell), Image.LANCZOS)
        x, y = (n % cols) * cell, (n // cols) * (cell + 24)
        sheet.paste(tile, (x, y + 24))
        draw.rectangle([x, y, x + cell - 1, y + 23], fill=(19, 26, 29))
        draw.text((x + 7, y + 6), f"{n + 1}", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)


def label_sheet(sheet: Path, n_slots: int, attribute: str, model: str = "sonnet",
                timeout: int = 600) -> SheetResult:
    """One `claude -p` invocation over one rendered sheet."""
    prompt = (
        f"Read the image at {sheet.resolve()} and label all {n_slots} numbered "
        f"crops for `{attribute}`. Return the JSON object and nothing else."
    )
    command = [
        "claude", "-p", "--output-format", "json", "--model", model,
        "--allowedTools", "Read",
        "--exclude-dynamic-system-prompt-sections",
        "--system-prompt-snapshot", "on",
        "--system-prompt", SYSTEM_PROMPT.format(rule=ATTRIBUTE_RULE[attribute]),
    ]
    env = dict(os.environ)
    token = resolve_token()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    try:
        proc = subprocess.run(command, input=prompt, capture_output=True,
                              text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return SheetResult({}, None, f"timeout after {timeout}s")
    if proc.returncode != 0:
        return SheetResult({}, None, f"exit {proc.returncode}: {proc.stderr.strip()[:200]}")

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return SheetResult({}, None, "envelope was not JSON")
    if envelope.get("is_error") or envelope.get("subtype") != "success":
        return SheetResult({}, envelope.get("total_cost_usd"),
                           f"{envelope.get('subtype')} {envelope.get('api_error_status') or ''}".strip())

    body = _FENCE.sub("", (envelope.get("result") or "").strip())
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", body, re.S)          # tolerate a stray sentence
        if not m:
            return SheetResult({}, envelope.get("total_cost_usd"), "no JSON in result")
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            return SheetResult({}, envelope.get("total_cost_usd"), "unparseable JSON")

    # Validate against the contract rather than trusting the model's vocabulary.
    verdicts, bad = {}, []
    for slot in range(1, n_slots + 1):
        v = raw.get(str(slot))
        if v is None:
            bad.append(f"missing {slot}"); continue
        try:
            verdicts[str(slot)] = Value(str(v).strip().lower()).value
        except ValueError:
            bad.append(f"slot {slot}={v!r}")
    err = "; ".join(bad)[:200] if bad else None
    return SheetResult(verdicts, envelope.get("total_cost_usd"), err)
