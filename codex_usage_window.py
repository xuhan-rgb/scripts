"""Read timestamped token usage for a rolling window from local Codex logs."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def summarize(root, start, end):
    models = {}
    keys = ("input_tokens", "cached_input_tokens", "output_tokens")
    for path in root.rglob("*.jsonl"):
        model = "unknown"
        previous = None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # A running session may have an incomplete final line.
                payload = event.get("payload") or {}
                if event.get("type") == "turn_context":
                    model = payload.get("model") or "unknown"
                if event.get("type") != "event_msg" or payload.get("type") != "token_count":
                    continue
                info = payload.get("info") or {}
                total = info.get("total_token_usage")
                if not total:
                    continue
                current = tuple(total.get(key, 0) for key in keys)
                if current == previous:
                    continue  # Quota updates can repeat the same cumulative usage.
                usage = info.get("last_token_usage")
                if usage is None:
                    baseline = previous or (0, 0, 0)
                    usage = dict(zip(keys, (
                        max(0, value - old) for value, old in zip(current, baseline)
                    )))
                previous = current  # Keep the baseline even outside the window.
                try:
                    timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if timestamp.tzinfo is None or not start <= timestamp <= end:
                    continue
                cached = usage.get("cached_input_tokens", 0)
                input_tokens = max(0, usage.get("input_tokens", 0) - cached)
                output = usage.get("output_tokens", 0)
                row = models.setdefault(model, dict.fromkeys(
                    ("inputTokens", "cachedInputTokens", "outputTokens", "totalTokens"), 0
                ))
                for key, value in zip(row, (input_tokens, cached, output, input_tokens + cached + output)):
                    row[key] += value
    return {"daily": [{"models": models}]}


if __name__ == "__main__":
    end = datetime.now(timezone.utc)
    duration = sys.argv[2]
    seconds = int(duration[:-1]) * {"h": 3600, "m": 60}[duration[-1]]
    start = end - timedelta(seconds=seconds)
    report = summarize(Path(sys.argv[1]) / "sessions", start, end)
    report["window"] = f"{start.astimezone().isoformat(timespec='seconds')} → {end.astimezone().isoformat(timespec='seconds')}"
    print(json.dumps(report))
