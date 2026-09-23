"""Group a Codex session with its recursive subagents for usage reports."""

import json
import sys
from pathlib import Path

sessions = json.load(sys.stdin)["sessions"]
if not sessions:
    print(json.dumps({"session": None, "daily": []}))
    sys.exit(0)

root = Path(sys.argv[1]) / "sessions"
nodes = {}
thread_ids = {}
for path in root.rglob("*.jsonl"):
    try:
        with path.open(encoding="utf-8") as stream:
            event = json.loads(stream.readline())
    except (OSError, ValueError):
        continue
    if event.get("type") != "session_meta":
        continue
    meta = event["payload"]
    identity = meta.get("id") or meta.get("session_id")
    if not identity:
        continue
    parent = meta.get("parent_thread_id")
    if not parent:
        try:
            parent = meta["source"]["subagent"]["thread_spawn"]["parent_thread_id"]
        except (KeyError, TypeError):
            pass
    session_id = str(path.relative_to(root).with_suffix(""))
    thread_ids[session_id] = identity
    nodes[identity] = {"sessionId": session_id, "parent": parent}

for session in sessions:
    session_id = session["sessionId"]
    identity = thread_ids.setdefault(session_id, session_id)
    nodes.setdefault(identity, {"sessionId": session_id, "parent": None})

def root_thread(session):
    identity = thread_ids[session["sessionId"]]
    seen = {identity}
    while nodes[identity]["parent"] in nodes:
        parent = nodes[identity]["parent"]
        if parent in seen:
            raise ValueError("会话父子关系存在循环：" + session["sessionId"])
        seen.add(parent)
        identity = parent
    return identity

latest = max(sessions, key=lambda session: session["lastActivity"])
identity = root_thread(latest)
members = [session for session in sessions if root_thread(session) == identity]
session_id = nodes[identity]["sessionId"]
print(json.dumps({"session": {
    "sessionId": session_id,
    "lastActivity": latest["lastActivity"],
    "subagentCount": sum(session["sessionId"] != session_id for session in members),
}, "daily": members}))
